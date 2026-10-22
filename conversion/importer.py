import bpy

import pathlib
import collections
from typing import Any, cast

import numpy as np
import numpy.linalg as la

from es3 import nif
from es3.utils.math import ID44, decompose

from .. import nif_switch
from . import import_animation, import_geometry, import_materials, import_skeletons
from .bone_axes import import_axis_correction
from .options import ImportOptions
from .records import (
    ImportCreationJob,
    ImportNode,
    ImportNullSlot,
    ImportRole,
    ImportSwitchPlan,
)
from .runtime import ImportRuntime, object_edit_scope, require_object_mode, scene_frame_scope


class Importer:
    def __init__(self, filepath, options: ImportOptions, runtime: ImportRuntime):
        self.options = options
        self.runtime = runtime
        self.nodes = []
        self.discovery_order = []
        self.materials = {}
        self.mesh_data = {}
        self.history = collections.defaultdict(list)
        self.bone_candidates = collections.defaultdict(set)
        self.skeletons = []
        self.skeletons_by_source = {}
        self.colliders = collections.defaultdict(set)
        self.switches = []
        self.warnings = []
        self.animation_context = import_animation.ImportAnimationContext(self.options, self.runtime.scene, self.warn)
        self.material_context = import_materials.ImportMaterialContext(
            self.options,
            self.runtime.texture_paths,
            self.materials,
            self.animation_context,
        )
        self.filepath = pathlib.Path(filepath)
        self.world_matrices = {}
        # Blender object (or pose bone, for bones) created for each node and null slot
        self.outputs: dict[ImportNode | ImportNullSlot, Any] = {}

    def execute(self):
        require_object_mode(self.runtime.context)

        data = nif.NiStream()
        data.load(self.filepath)
        data.merge_properties()

        # fix roots list
        # every AV root is kept: the engine attaches them all under a container it synthesizes
        data.roots = list(data.roots_of_type(nif.NiAVObject)) or [nif.NiNode()]

        # fix transforms
        # several roots sit under the engine's container, so their transforms are real placement
        if self.options.discard_root_transforms and len(data.roots) == 1:
            if isinstance(data.root, nif.NiNode):
                data.root.matrix = ID44

        # attach kf file
        if self.options.attach_keyframe_data:
            self.import_keyframe_data(data)

        # copy file name
        for root in data.roots:
            if not isinstance(root, nif.NiObjectNET):
                continue
            if root.name == "":
                root.name = self.filepath.name

        # scale correction
        data.apply_scale(self.runtime.scale_correction)

        # time correction
        data.apply_time_scale(self.runtime.fps)

        # resolve heirarchy
        roots = self.resolve_nodes(data.roots_of_type(nif.NiAVObject))

        # resolve armatures
        if any(self.bone_candidates):
            self.skeletons, self.skeletons_by_source = import_skeletons.resolve_armatures(
                self.nodes,
                self.bone_candidates,
                self.get,
            )
            import_skeletons.correct_rest_positions(
                self.skeletons,
                self.nodes,
                self.get,
                self.skeletons_by_source,
            )
            import_skeletons.apply_axis_corrections(self.skeletons, self.nodes)
            import_skeletons.correct_bone_parenting(
                self.skeletons,
                self.nodes,
                self.get,
                self.skeletons_by_source,
            )

        self.world_matrices = import_skeletons.evaluate_node_worlds(self.nodes)

        creation_schedule = self.build_creation_schedule()

        # create at frame zero, then return to the current frame
        with scene_frame_scope(self.runtime.scene, 0):
            # create bl objects
            for job in creation_schedule:
                self.create_job(job)

            # switch types and selections
            self.resolve_switches()

            # unmute animations
            for skeleton in self.skeletons:
                import_animation.set_mute(self.outputs[skeleton.root], False, self.animation_context)

            # Include the active-object change in the frame-restoration evaluation.
            self.runtime.view_layer.objects.active = self.get_root_output(roots)

    # -------
    # RESOLVE
    # -------

    def resolve_nodes(self, ni_roots, parent=None):
        root_nodes = [new_node(root, parent) for root in ni_roots]

        queue: collections.deque[ImportNode | ImportNullSlot] = collections.deque(root_nodes)
        while queue:
            target = queue.popleft()
            self.discovery_order.append(target)
            if isinstance(target, ImportNullSlot):
                continue

            node = target
            if not self.process(node):
                continue

            self.nodes.append(node)
            self.history[node.source].append(node)
            if type(node.source) in (nif.NiSwitchNode, nif.NiLODNode, nif.NiFltAnimationNode):
                # null links are slots too: queue them in place so markers link in source order
                slots = tuple(
                    new_node(child, node) if isinstance(child, nif.NiAVObject) else ImportNullSlot(node)
                    for child in node.source.children
                )
                plan = node.switch_plan = ImportSwitchPlan(node, slots)
                self.switches.append(plan)
                queue.extend(slots)
            elif hasattr(node.source, "children"):
                queue.extend(new_node(child, node) for child in node.source.children_of_type(nif.NiAVObject))

        return root_nodes

    def resolve_switches(self):
        """Mark imported switch-family nodes and resolve their selections or ranges, once every child object exists."""
        for plan in self.switches:
            node = plan.owner
            name = node.source.name
            node_type = type(node.source).__name__
            output = self.outputs.get(node)
            if not (isinstance(output, bpy.types.Object) and output.type == "EMPTY"):
                self.warn(f"{node_type} '{name}' was imported as part of a skeleton; its {node_type} semantics were dropped")
                continue

            output = cast(Any, output)
            slots = [self.slot_output(node, slot) for slot in plan.slots]
            if None in slots:
                self.warn(f"{node_type} '{name}' lost a child to skeleton or collision handling; its indices shifted")

            if node_type == nif_switch.LOD:
                self.resolve_lod(node, output, slots)
                continue

            index = node.source.active_index
            if index == -1:
                continue
            if 0 <= index < len(slots) and slots[index] is not None:
                output.mw.switch_active_child = slots[index]
            else:
                output.mw.switch_invalid_index = index
                self.warn(f"{node_type} '{name}' selects child {index} of {len(slots)}; pick a valid child before export")

    def slot_output(self, owner, slot):
        """The object realizing a switch slot, or None if the slot left the switch or became a bone."""
        output = self.outputs.get(slot)
        if slot.parent is owner and isinstance(output, bpy.types.Object):
            return output
        return None

    def resolve_lod(self, node, output, slots):
        # the stored active index is ignored: the engine recomputes it every frame
        name = node.source.name
        lod = output.mw
        lod.lod_center = tuple(map(float, node.source.lod_center))
        levels = node.source.lod_levels
        # every child gets a row; one without a level keeps the default range
        rows = nif_switch.ensure_lod_rows(output, [slot for slot in slots if slot is not None])
        # rows are filled with sync off: filling one row must not rewrite its neighbors
        lod.lod_sync = False
        for slot, (near, far) in zip(slots, levels):
            if slot is not None:
                rows[slot].near, rows[slot].far = float(near), float(far)
        lod.lod_sync = len(levels) >= len(slots) and nif_switch.levels_chained(levels[: len(slots)])
        if len(levels) > len(slots):
            self.warn(
                f"NiLODNode '{name}' has {len(levels)} ranges for {len(slots)} children; the extra ranges were dropped"
            )
        elif len(levels) < len(slots):
            self.warn(
                f"NiLODNode '{name}' has {len(levels)} ranges for {len(slots)} children; the rest use the default range"
            )

    # -------
    # PROCESS
    # -------

    def process(self, node):
        source_type = type(node.source)
        if source_type in (
            nif.NiNode,
            nif.NiLODNode,
            nif.NiSwitchNode,
            nif.NiFltAnimationNode,
            nif.AvoidNode,
            nif.NiBillboardNode,
            nif.NiBSAnimationNode,
            nif.NiBSParticleNode,
            nif.NiCollisionSwitch,
            nif.NiSortAdjustNode,
        ):
            return self.process_empty(node)
        if source_type is nif.NiTriShape:
            return self.process_mesh(node)
        if source_type is nif.RootCollisionNode:
            return self.process_collision(node)

        print(f"Warning: Unhandled Type: {node.source.type}")
        # containment: an unsupported type becomes a placeholder rather than terminating
        # traversal, so its supported descendants survive. Returning True is what lets
        # resolve_nodes keep walking; deliberate skips (collision) still return False.
        node.role = ImportRole.EMPTY
        return True

    def process_empty(self, node):
        node.role = ImportRole.EMPTY

        # detect bones via name conventions
        name = node.name.lower()
        if (name == "bip01") or (name == "root bone"):
            self.bone_candidates[node.source].update()
        elif ("bip01" in name) or name.endswith(" bone"):
            self.bone_candidates[None].add(node.source)

        return True

    def process_mesh(self, node):
        node.role = ImportRole.MESH

        # track skinned meshes
        skin = node.source.skin
        if skin and skin.root and skin.bones:
            self.bone_candidates[skin.root].update(skin.bones)

        return True

    def process_collision(self, node):
        if self.options.ignore_collision_nodes:
            node.role = ImportRole.PRUNED
            return False

        node.role = ImportRole.EMPTY
        self.colliders[node.source].update(node.source.descendants())
        return True

    # -------
    # UTILITY
    # -------

    def get(self, source):
        return self.history[source][0]

    def warn(self, message):
        print(f"Warning: {message}")
        self.warnings.append(message)

    def build_creation_schedule(self):
        owned_bones = set()
        for skeleton in self.skeletons:
            duplicate_bones = owned_bones.intersection(skeleton.bones)
            if duplicate_bones:
                raise RuntimeError("Each imported bone must belong to exactly one resolved skeleton")
            owned_bones.update(skeleton.bones)

        jobs = []
        scheduled_bones = set()
        scheduled_targets = set()

        for target in self.discovery_order:
            if isinstance(target, ImportNullSlot):
                if target in scheduled_targets:
                    raise RuntimeError("An import target cannot have more than one creation job")
                jobs.append(ImportCreationJob(target, None))
                scheduled_targets.add(target)
                continue
            if target.role in (ImportRole.PRUNED, ImportRole.BONE):
                continue
            if target in scheduled_targets:
                raise RuntimeError("An import target cannot have more than one creation job")
            if target.role is ImportRole.ARMATURE:
                skeleton = self.skeletons_by_source[target.source]
                jobs.append(ImportCreationJob(target, target.role, skeleton.bones))
                scheduled_bones.update(skeleton.bones)
            else:
                jobs.append(ImportCreationJob(target, target.role))
            scheduled_targets.add(target)

        if scheduled_bones != owned_bones:
            raise RuntimeError("Each imported bone must be owned by exactly one armature creation job")
        return tuple(jobs)

    def get_root_output(self, roots):
        # a root may legitimately have no output: an ignored RootCollisionNode
        output = next((self.outputs[root] for root in roots if root in self.outputs), None)
        return output.id_data if output is not None else None

    def get_mesh_data(self, shape, name) -> bpy.types.Mesh:
        if shape.morph_targets or shape.bone_influences:
            # TODO: Support instancing of animated and skinned meshes.
            return bpy.data.meshes.new(name)

        try:
            bl_data = self.mesh_data[shape.data]
        except KeyError:
            bl_data = self.mesh_data[shape.data] = bpy.data.meshes.new(name)

        return bl_data

    def import_keyframe_data(self, data):
        kf_path = self.filepath.with_suffix(".kf")
        if not kf_path.exists():
            print(f'import_keyframe_data: "{kf_path}" does not exist')
        else:
            kf_data = nif.NiStream()
            kf_data.load(kf_path)
            data.attach_keyframe_data(kf_data)

    # ------
    # CREATE
    # ------

    def create_job(self, job):
        target = job.target
        if isinstance(target, ImportNullSlot):
            self.create_null_slot(target)
        elif job.role is ImportRole.EMPTY:
            self.create_empty(target)
        elif job.role is ImportRole.MESH:
            self.create_mesh(target)
        elif job.role is ImportRole.ARMATURE:
            self.create_armature(target, job.owned_bones)
        else:
            raise RuntimeError(f"Unsupported import creation role: {job.role}")

    def create_empty(self, node, bl_data=None):
        """Create, place and animate the object of `node`; `bl_data` makes it a mesh or armature object."""
        output: Any = self.create_object(node.name, bl_data)
        self.outputs[node] = output
        output.empty_display_size *= self.runtime.scale_correction
        if getattr(output, "type", None) == "EMPTY":
            # before the flags: picking a type resets its type-specific bits
            self.set_node_type(node.source, output)
        output.mw.object_flags = node.source.flags

        bl_parent = cast(Any, self.outputs.get(node.parent))
        try:
            output.parent = bl_parent
        except TypeError:
            # parent is an armature bone
            output.parent = bl_parent.id_data
            output.parent_type = "BONE"
            output.parent_bone = bl_parent.name
            # blender places bone children at the axis-corrected bone's tail; undoing both in the parent inverse keeps
            # matrix_basis, and the raw animation keys, in the node's own space without reading the evaluated pose
            axis = import_axis_correction(node.parent.source.name)
            tail = np.eye(4)
            tail[1, 3] = bl_parent.bone.length
            output.matrix_parent_inverse = la.inv(axis @ tail).T
            output.matrix_basis = (axis @ node.matrix_local).T
        else:
            # parent is an empty or None
            output.matrix_local = node.matrix_local.T

        if node.source in self.colliders:
            output.name = "Collision"
            output.display_type = "WIRE"

        if node.source.is_bounding_box:
            output.empty_display_size = 1.0
            output.empty_display_type = "CUBE"
            output.matrix_world = node.source.bounding_volume.matrix.T

        import_animation.import_object_animation(
            node.source,
            output,
            bone_name(node.name),
            self.animation_context,
        )

        return output

    @staticmethod
    def set_node_type(source, output):
        """Type an Empty after its source class and fill that type's payload; unsupported types stay NiNode."""
        node_type = type(source).__name__
        if node_type not in nif_switch.NODE_TYPE_FLAGS:
            return
        output.mw.node_type = node_type
        if isinstance(source, nif.NiFltAnimationNode):
            output.mw.flt_period = source.period
        elif isinstance(source, nif.NiSortAdjustNode):
            # the accumulator link carries no data; export recreates it for Subsort
            output.mw.sorting_mode = nif.NiSortAdjustNode.SortingMode(source.sorting_mode).name.removeprefix("SORTING_")

    def create_object(self, name, bl_data=None):
        bl_object = bpy.data.objects.new(name, bl_data)
        self.runtime.active_collection.objects.link(bl_object)
        bl_object.select_set(True)
        return bl_object

    def create_null_slot(self, slot):
        """Realize a null link in a switch's children as a marked Empty, so the slot keeps its index."""
        bl_parent = self.outputs.get(slot.parent)
        if not (isinstance(bl_parent, bpy.types.Object) and bl_parent.type == "EMPTY"):
            return None  # skeletal switch, diagnosed by resolve_switches

        output: Any = self.create_object(nif_switch.NULL_SLOT_NAME)
        self.outputs[slot] = output
        output.empty_display_size = bl_parent.empty_display_size * 0.5
        output.mw.is_switch_null = True
        output.parent = bl_parent
        return output

    def create_armature(self, node, bone_nodes):
        # create armature object
        bl_data = bpy.data.armatures.new(node.name)
        bl_object = self.create_empty(node, bl_data)

        # apply default settings
        bl_data.display_type = "STICK"
        bl_object.show_in_front = True

        # used for calculating armature space matrices
        placement = self.get(node.source)
        skeleton = self.skeletons_by_source[node.source]
        posed_world = skeleton.posed_world
        root_inverse = la.inv(self.world_matrices[placement])

        # bones can only be created in edit mode
        with object_edit_scope(self.runtime, bl_object):
            bones = self.create_edit_bones(bl_data, bone_nodes, root_inverse, skeleton)

        # armature-space pose matrices
        posed = {bone: (root_inverse @ posed_world[bone]).T for bone in bones}

        # record each bone's output and apply pose transforms
        for bone, name in bones.items():
            pose_bone = self.outputs[bone] = bl_object.pose.bones[name]
            cast(Any, pose_bone).mw.object_flags = bone.source.flags
            # the pose_bone.matrix setter reads the parent's evaluated pose, which would need a depsgraph
            # update per bone, so convert from the parent's known pose instead
            data_bone = pose_bone.bone
            parent = {}
            if data_bone.parent:
                parent = {"parent_matrix": posed[bone.parent], "parent_matrix_local": data_bone.parent.matrix_local}
            pose_bone.matrix_basis = data_bone.convert_local_to_pose(
                posed[bone], data_bone.matrix_local, invert=True, **parent
            )
            # create animations, preserve poses
            import_animation.import_object_animation(
                bone.source,
                pose_bone,
                bone_name(bone.name),
                self.animation_context,
            )
            import_animation.set_mute(pose_bone, True, self.animation_context)

        return bl_object

    def create_edit_bones(self, bl_data, bone_nodes, root_inverse, skeleton):
        """Create and place the edit bone of every node; returns each node's bone name."""
        rest_world = skeleton.rest_world
        posed_world = skeleton.posed_world

        # bone mappings cache
        bones = {}

        # position bone heads
        for node in bone_nodes:
            # create bone and assign its parent
            bone = bones[node] = bl_data.edit_bones.new(bone_name(node.name))
            bone.parent = bones.get(node.parent)
            bone.select = True

            # compute the armature-space matrix
            matrix = root_inverse @ rest_world[node]

            # calculate axis/roll and head/tail
            bone.matrix = matrix.T
            bone.tail = matrix[:3, 1] + matrix[:3, 3]  # axis + head

        # position bone tails
        for node, bone in bones.items():
            # edit_bones will not persist outside of edit mode
            bones[node] = bone.name

            if bone.children:
                # calculate length from children mean location
                locations = [posed_world[c][:3, 3] for c in node.children if c in bones]
                bone.length = la.norm(posed_world[node][:3, 3] - np.mean(locations, axis=0))
            elif bone.parent:
                # set length to half of the parent bone length
                bone.length = bone.parent.length / 2

            if bone.length <= 1e-5:
                print(f"Warning: Zero length bones are not supported ({bone.name})")
                # TODO figure out a proper fix for zero length bones
                bone.tail.z += 1e-5

        return bones

    def create_mesh(self, node):
        bl_data = self.get_mesh_data(node.source, node.name)
        bl_object = self.create_empty(node, bl_data)

        if len(node.source.data.vertices) == 0:
            return bl_object

        # We only need to calculate the geometry once per mesh instance.
        if bl_data.users == 1:
            self.create_geometry(node, bl_object)

        if node.parent is not None:
            try:
                parent_output = cast(Any, self.outputs.get(node.parent))
                bl_object.display_type = parent_output.display_type
            except AttributeError:
                pass

        import_materials.import_material(
            node.source,
            inherited_properties(node),
            bl_object,
            self.material_context,
        )

        return bl_object

    def create_geometry(self, node, ob):
        # the placement's world scale sizes the welding tolerance
        source = node.source
        placement = self.get(source)
        scale = decompose(self.world_matrices[placement])[-1]
        buffers = import_geometry.prepare_buffers(source, scale, self.options.vertex_precision, node.name)

        import_geometry.create_vertices(ob, buffers.vertices)
        import_geometry.create_triangles(ob, buffers.triangles)

        import_geometry.create_vertex_colors(ob, buffers.vertex_colors)
        import_geometry.create_uv_sets(ob, buffers.uv_sets)

        if buffers.skin is not None:
            root = self.outputs[self.get(source.skin.root)]
            bone_names = [self.outputs[self.get(bone)].name for bone in source.skin.bones]
            import_geometry.create_vertex_weights(ob, buffers.skin, root.id_data, bone_names)
        import_geometry.create_vertex_morphs(
            ob,
            buffers.morphs,
            source.morph_targets,
            self.runtime.scene,
            source.controller,
        )

        import_geometry.create_normals(ob, buffers.normals, self.options.ignore_custom_normals)


def new_node(source, parent=None):
    node = ImportNode(source, np.asarray(source.matrix, dtype="<f"))
    node.reparent(parent)
    return node


def bone_name(name):
    """The Blender name of a NIF bone: Bip01 sides become Blender's .L/.R suffixes."""
    if name.startswith("Bip01 L "):
        return f"Bip01 {name[8:]}.L"
    if name.startswith("Bip01 R "):
        return f"Bip01 {name[8:]}.R"
    return name


def inherited_properties(node):
    """The NIF properties in effect on `node`, each type taken from the nearest node that has it."""
    props = {type(p): p for p in node.source.properties}
    if node.parent is not None:
        return {**inherited_properties(node.parent), **props}
    return props
