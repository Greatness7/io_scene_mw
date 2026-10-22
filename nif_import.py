import bpy

import timeit
import pathlib
import collections
from typing import Any, cast

import numpy as np
import numpy.linalg as la

from es3 import nif
from es3.utils.math import ID44, decompose

from . import nif_utils
from . import nif_switch
from .conversion import import_animation, import_materials, import_skeletons
from .conversion.blender_compat import ensure_action, ensure_fcurve
from .conversion.options import ImportOptions, import_options_from_mapping
from .conversion.records import (
    ImportCreationJob,
    ImportMeshBuffers,
    ImportMorphBuffers,
    ImportNode,
    ImportNullSlot,
    ImportRole,
    ImportSkinBuffers,
    ImportSwitchPlan,
)
from .conversion.runtime import ImportRuntime

def load(context, filepath, report=None, **config):
    """load a scene from a nif file"""

    print(f"Import File: {filepath}")
    time = timeit.default_timer()

    options = import_options_from_mapping(config)
    runtime = ImportRuntime.from_context(context)
    importer = Importer(filepath, options, runtime)
    importer.execute()

    for warning in importer.warnings:
        if report:
            report({"WARNING"}, warning)

    time = timeit.default_timer() - time
    print(f"Import Done: {time:.4f} seconds")

    return {"FINISHED"}


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

    def execute(self):
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

        # discard frame pos
        frame_current = self.runtime.scene.frame_current
        self.runtime.scene.frame_set(0)

        # create bl objects
        for job in creation_schedule:
            self.create_job(job)

        # switch types and selections
        self.resolve_switches()

        # unmute animations
        for skeleton in self.skeletons:
            import_animation.set_mute(skeleton.root.output, False, self.animation_context)

        # restore frame pos
        self.runtime.scene.frame_current = frame_current

        # set active object
        self.runtime.view_layer.objects.active = self.get_root_output(roots)

    # -------
    # RESOLVE
    # -------

    def resolve_nodes(self, ni_roots, parent=None):
        root_nodes = [SceneNode(self, root, parent) for root in ni_roots]

        queue: collections.deque[SceneNode | ImportNullSlot] = collections.deque(root_nodes)
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
            # exact types: NiFltAnimationNode stays an ordinary node
            if type(node.source) in (nif.NiSwitchNode, nif.NiLODNode):
                # null links are slots too: queue them in place so markers link in source order
                slots = tuple(
                    SceneNode(self, child, node) if isinstance(child, nif.NiAVObject) else ImportNullSlot(self, node)
                    for child in node.source.children
                )
                plan = node.switch_plan = ImportSwitchPlan(node, slots)
                self.switches.append(plan)
                queue.extend(slots)
            elif hasattr(node.source, "children"):
                queue.extend(SceneNode(self, child, node) for child in node.source.children_of_type(nif.NiAVObject))

        return root_nodes

    def resolve_switches(self):
        """Mark imported switch-family nodes and resolve their selections or ranges, once every child object exists."""
        for plan in self.switches:
            node = plan.owner
            name = node.source.name
            node_type = type(node.source).__name__
            if not (isinstance(node.output, bpy.types.Object) and node.output.type == "EMPTY"):
                self.warn(f"{node_type} '{name}' was imported as part of a skeleton; its {node_type} semantics were dropped")
                continue

            output = cast(Any, node.output)
            output.mw.node_type = node_type
            slots = [
                slot.output if slot.parent is node and isinstance(slot.output, bpy.types.Object) else None
                for slot in plan.slots
            ]
            if None in slots:
                self.warn(f"{node_type} '{name}' lost a child to skeleton or collision handling; its indices shifted")

            if node_type == nif_switch.LOD:
                self.resolve_lod(node, slots)
                continue

            index = node.source.active_index
            if index == -1:
                continue
            if 0 <= index < len(slots) and slots[index] is not None:
                output.mw.switch_active_child = slots[index]
            else:
                output.mw.switch_invalid_index = index
                self.warn(f"{node_type} '{name}' selects child {index} of {len(slots)}; pick a valid child before export")

    def resolve_lod(self, node, slots):
        # the stored active index is ignored: the engine recomputes it every frame
        name = node.source.name
        output = cast(Any, node.output)
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
            self.warn(f"NiLODNode '{name}' has {len(levels)} ranges for {len(slots)} children; the extra ranges were dropped")
        elif len(levels) < len(slots):
            self.warn(f"NiLODNode '{name}' has {len(levels)} ranges for {len(slots)} children; the rest use the default range")

    def resolve_armatures(self):
        self.skeletons, self.skeletons_by_source = import_skeletons.resolve_armatures(
            self.nodes,
            self.bone_candidates,
            self.get,
        )
    def resolve_armature(self, root, bones, orphan_bones):
        posed_worlds = import_skeletons.evaluate_node_worlds(self.nodes)
        return import_skeletons.resolve_armature(root, bones, orphan_bones, self.nodes, self.get, posed_worlds)

    def correct_rest_positions(self):
        import_skeletons.correct_rest_positions(
            self.skeletons,
            self.nodes,
            self.get,
            self.skeletons_by_source,
        )
    def correct_rest_position(self, root):
        skeleton = self.skeletons_by_source[root.source]
        import_skeletons.correct_rest_positions(
            [skeleton],
            self.nodes,
            self.get,
            self.skeletons_by_source,
        )
    def apply_axis_corrections(self):
        import_skeletons.apply_axis_corrections(self.skeletons, self.nodes)

    def apply_axis_correction(self, root):
        import_skeletons.apply_axis_corrections([self.skeletons_by_source[root.source]], self.nodes)

    def correct_bone_parenting(self):
        import_skeletons.correct_bone_parenting(
            self.skeletons,
            self.nodes,
            self.get,
            self.skeletons_by_source,
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
            nif.NiBSAnimationNode,
            nif.NiCollisionSwitch,
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

    def iter_bones(self, root):
        yield from self.skeletons_by_source[root.source].bones

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

    def create_job(self, job):
        target = job.target
        if isinstance(target, ImportNullSlot):
            SwitchNull(target).create()
        elif job.role is ImportRole.EMPTY:
            Empty(target).create()
        elif job.role is ImportRole.MESH:
            Mesh(target).create()
        elif job.role is ImportRole.ARMATURE:
            Armature(target).create(job.owned_bones)
        else:
            raise RuntimeError(f"Unsupported import creation role: {job.role}")

    def get_root_output(self, roots):
        # a root may legitimately have no output: an ignored RootCollisionNode
        output = next((root.output for root in roots if root.output is not None), None)
        return output.id_data if output is not None else None

    def get_armature_node(self, source):
        return import_skeletons.get_armature_node(source, self.get, self.skeletons_by_source)

    def get_skinned_meshes(self, armature):
        yield from import_skeletons.get_skinned_meshes(
            armature,
            self.nodes,
            self.get,
            self.skeletons_by_source,
        )
    def get_skeleton_for_bone(self, bone):
        return next(skeleton for skeleton in self.skeletons if bone in skeleton.posed_world)

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

class SceneNode(ImportNode):

    def __init__(self, importer, source, parent=None):
        super().__init__(importer, source, np.asarray(source.matrix, dtype="<f"))
        self.parent = parent

    def __repr__(self):
        if not self.parent:
            return f'SceneNode("{self.name}", parent=None)'
        return f'SceneNode("{self.name}", parent={self.parent.source.name})'

    def create(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def name(self):
        return self.source.name

    @property
    def bone_name(self):
        name = self.name
        if name.startswith("Bip01 L "):
            return f"Bip01 {name[8:]}.L"
        if name.startswith("Bip01 R "):
            return f"Bip01 {name[8:]}.R"
        return name

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, node):
        if self._parent is not None:
            try:
                self._parent.children.remove(self)
            except ValueError:
                pass
        self._parent = node
        if self._parent is not None:
            self._parent.children.append(self)

    @property
    def parents(self):
        node = self.parent
        while node:
            yield node
            node = node._parent

    @property
    def properties(self):
        props = {type(p): p for p in self.source.properties}
        if self.parent:
            return {**getattr(self.parent, "properties"), **props}
        return props

class Empty(SceneNode):
    __slots__ = ()

    def __init__(self, node):
        self.__dict__ = node.__dict__

    def create(self, bl_data=None):
        self.output = self.create_object(bl_data)
        self.output.empty_display_size *= self.importer.runtime.scale_correction
        self.output.mw.object_flags = self.source.flags

        bl_parent = cast(Any, getattr(self.parent, "output", None))
        try:
            self.output.parent = bl_parent
        except TypeError:
            # parent is an armature bone
            self.output.parent = bl_parent.id_data
            self.output.parent_type = "BONE"
            self.output.parent_bone = bl_parent.name
            skeleton = self.importer.get_skeleton_for_bone(self.parent)
            self.output.matrix_world = (skeleton.posed_world[self.parent] @ self.matrix_local).T
        else:
            # parent is an empty or None
            self.output.matrix_local = self.matrix_local.T

        if self.source in self.importer.colliders:
            self.output.name = "Collision"
            self.output.display_type = "WIRE"
            if getattr(self.output, "type", None) == "EMPTY":
                self.output.mw.node_type = nif_switch.COLLISION

        if self.source.is_bounding_box:
            self.convert_to_bounding_box()

        import_animation.import_object_animation(
            self.source,
            self.output,
            self.bone_name,
            self.importer.animation_context,
        )

        return self.output

    def create_object(self, bl_data=None):
        bl_object = bpy.data.objects.new(self.name, bl_data)
        self.importer.runtime.active_collection.objects.link(bl_object)
        bl_object.select_set(True)
        return bl_object

    def convert_to_bounding_box(self):
        self.output.empty_display_size = 1.0
        self.output.empty_display_type = 'CUBE'
        self.output.matrix_world = self.source.bounding_volume.matrix.T


class SwitchNull(ImportNullSlot):
    """A null link in a switch's children, realized as a marked Empty so the slot keeps its index."""

    __slots__ = ()

    def __init__(self, slot):
        self.__dict__ = slot.__dict__

    @property
    def name(self):
        return nif_switch.NULL_SLOT_NAME

    def create(self):
        bl_parent = self.parent.output
        if not (isinstance(bl_parent, bpy.types.Object) and bl_parent.type == "EMPTY"):
            return None  # skeletal switch, diagnosed by resolve_switches

        self.output = bpy.data.objects.new(self.name, None)
        self.importer.runtime.active_collection.objects.link(self.output)
        self.output.select_set(True)
        self.output.empty_display_size = bl_parent.empty_display_size * 0.5
        self.output.mw.is_switch_null = True
        self.output.parent = bl_parent
        return self.output


class Armature(SceneNode):
    __slots__ = ()

    def __init__(self, node):
        self.__dict__ = node.__dict__

    def create(self, bone_nodes):
        # create armature object
        bl_data = bpy.data.armatures.new(self.name)
        bl_object = Empty(self).create(bl_data)

        # apply default settings
        bl_data.display_type = "STICK"
        bl_object.show_in_front = True

        # swap to edit mode to allow creation of bones
        self.importer.runtime.view_layer.objects.active = bl_object
        bpy.ops.object.mode_set(mode="EDIT")

        # used for calculating armature space matrices
        placement = self.importer.get(self.source)
        skeleton = self.importer.skeletons_by_source[self.source]
        rest_world = skeleton.rest_world
        posed_world = skeleton.posed_world
        root_inverse = la.inv(self.importer.world_matrices[placement])

        # bone mappings cache
        bones = {}

        # position bone heads
        for node in bone_nodes:
            # create bone and assign its parent
            bone = bones[node] = bl_data.edit_bones.new(node.bone_name)
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

        # back to object mode now that all bones exist
        bpy.ops.object.mode_set(mode="OBJECT")

        # assign node.output and apply pose transforms
        for node, name in bones.items():
            pose_bone = node.output = bl_object.pose.bones[name]
            # compute the armature-space matrix
            pose_bone.matrix = (root_inverse @ posed_world[node]).T
            # TODO try not to call scene update
            self.importer.runtime.view_layer.depsgraph.update()
            # create animations, preserve poses
            import_animation.import_object_animation(
                node.source,
                node.output,
                node.bone_name,
                self.importer.animation_context,
            )
            import_animation.set_mute(node.output, True, self.importer.animation_context)

        return bl_object


class Mesh(SceneNode):
    __slots__ = ()

    def __init__(self, node):
        self.__dict__ = node.__dict__

    def create(self):
        bl_data = self.importer.get_mesh_data(self.source, self.name)
        bl_object = Empty(self).create(bl_data)

        if len(self.source.data.vertices) == 0:
            return bl_object

        # We only need to calculate the geometry once per mesh instance.
        if bl_data.users == 1:
            ni_data = self.calc_geometry_data()

            self.create_vertices(bl_object, ni_data.vertices)
            self.create_triangles(bl_object, ni_data.triangles)

            self.create_vertex_colors(bl_object, ni_data.vertex_colors)
            self.create_uv_sets(bl_object, ni_data.uv_sets)

            self.create_vertex_weights(bl_object, ni_data.skin)
            self.create_vertex_morphs(bl_object, ni_data.morphs)

            self.create_normals(bl_object, ni_data.normals)

        if self.parent is not None:
            try:
                parent_output = cast(Any, self.parent.output)
                self.output.display_type = parent_output.display_type
            except AttributeError:
                pass

        import_materials.import_material(
            self.source,
            self.properties,
            self.output,
            self.importer.material_context,
        )

        return bl_object

    def create_vertices(self, ob, vertices):
        ob.data.vertices.add(len(vertices))
        ob.data.vertices.foreach_set("co", vertices.ravel())

    def create_triangles(self, ob, triangles):
        n = len(triangles)
        ob.data.loops.add(3 * n)
        ob.data.loops.foreach_set("vertex_index", triangles.ravel())

        ob.data.polygons.add(n)
        ob.data.polygons.foreach_set("loop_total", [3] * n)
        ob.data.polygons.foreach_set("loop_start", range(0, 3 * n, 3))
        ob.data.polygons.foreach_set("use_smooth", [True] * n)

        ob.data.update()

    def create_normals(self, ob, normals):
        if len(normals) == 0:
            ob.data["ignore_normals"] = True
        else:
            # Each polygon has a "use_smooth" flag that controls whether it
            # should use flat shading or smoooth shading. Our custom normals
            # will override this behavior, but the user may decide to remove
            # custom data layers at some point after importing, which would
            # make the renderer fall back to using said flags. We calculate
            # these flags as best we can by checking if the polygon's normals
            # are all equivalent, which would mean it is NOT smooth shaded.
            n0, n1, n2 = np.swapaxes(normals.reshape(-1, 3, 3), 0, 1)
            n0__eq__n1 = np.isclose(n0, n1, rtol=0, atol=1e-04)
            n1__eq__n2 = np.isclose(n1, n2, rtol=0, atol=1e-04)
            use_smooth = ~(n0__eq__n1 & n1__eq__n2).all(axis=1)
            ob.data.polygons.foreach_set("use_smooth", use_smooth)

            # apply custom normals
            if not self.importer.options.ignore_custom_normals:
                if ob.data.validate(verbose=False, clean_customdata=False):
                    print(f"Warning: Invalid mesh data, custom normals will be skipped: ({ob.name})")
                else:
                    ob.data.normals_split_custom_set(normals)
                    if bpy.app.version < (4, 1, 0):
                        ob.data.use_auto_smooth = True

    def create_uv_sets(self, ob, uv_sets):
        for uv_set in uv_sets[:8]:  # max 8 uv sets (blender limitation)
            uv = ob.data.uv_layers.new()
            uv.data.foreach_set("uv", uv_set.ravel())

    def create_vertex_colors(self, ob, vertex_colors):
        if len(vertex_colors):
            vc = ob.data.vertex_colors.new()
            vc.data.foreach_set("color", vertex_colors.ravel())

    def create_vertex_weights(self, ob, skin_data):
        if skin_data is None:
            return

        vertex_weights = skin_data.weights

        root = self.importer.get(self.source.skin.root)
        bones = map(self.importer.get, self.source.skin.bones)

        # Make Armature
        armature = ob.modifiers.new("", "ARMATURE")
        armature.object = root.output.id_data

        # Vertex Weights
        for i, node in enumerate(bones):
            vg = ob.vertex_groups.new(name=node.output.name)

            weights = vertex_weights[i]
            for j in np.flatnonzero(weights).tolist():
                vg.add((j,), weights[j], "ADD")

    def create_vertex_morphs(self, ob, morph_data):
        if morph_data is None:
            return

        vertex_morphs = morph_data.targets

        # add basis key
        ob.shape_key_add(name="Basis")

        # add anim data
        action = ensure_action(ob.data.shape_keys)

        # add morph keys
        for i, target in enumerate(self.source.morph_targets):
            # create morph targets
            shape_key = ob.shape_key_add(name="")
            shape_key.data.foreach_set("co", vertex_morphs[i].ravel())

            # create morph fcurves
            data_path = shape_key.path_from_id("value")
            fc = ensure_fcurve(action, data_path)

            # add fcurve keyframes
            fc.keyframe_points.add(len(target.keys))
            fc.keyframe_points.foreach_set("co", target.keys[:, :2].ravel())
            import_animation.create_interpolation_data(target, fc)
            fc.update()

        # update frame range
        import_animation.update_frame_range(self.importer.runtime.scene, self.source.controller)

    def calc_geometry_data(self):
        vertices = self.source.data.vertices
        normals = self.source.data.normals
        uv_sets = self.source.data.uv_sets.copy()
        vertex_colors = self.source.data.vertex_colors
        vertex_weights = self.source.vertex_weights()
        vertex_morphs = self.source.vertex_morphs()
        triangles = self.source.data.triangles

        if len(normals):
            # re-unitize, fixes landscape meshes
            normals /= la.norm(normals, axis=1, keepdims=True)
            # reconstruct as per-triangle layout
            normals = normals[triangles].reshape(-1, 3)

        if len(uv_sets):
            # convert OpenGL into Blender format
            uv_sets[..., 1] = 1 - uv_sets[..., 1]
            # reconstruct as per-triangle layout
            uv_sets = uv_sets[:, triangles].reshape(-1, triangles.size, 2)

        if len(vertex_colors):
            # reconstruct as per-triangle layout
            vertex_colors = vertex_colors[triangles].reshape(-1, 4)

        # remove doubles
        placement = self.importer.get(self.source)
        scale = decompose(self.importer.world_matrices[placement])[-1]
        indices, inverse = nif_utils.unique_rows(
            vertices * scale,
            *vertex_weights,
            *vertex_morphs,
            precision=self.importer.options.vertex_precision,
        )
        if len(vertices) > len(indices) > 3:
            vertices = vertices[indices]
            vertex_weights = vertex_weights[:, indices]
            vertex_morphs = vertex_morphs[:, indices]
            triangles = inverse[triangles]

        # '''
        # Blender does not allow two faces to use identical vertex indices, regardless of order.
        # This is problematic as such occurances are commonly found throughout most nif data sets.
        # The usual case is "double-sided" faces, which share vertex indices but differ in winding.
        # Identify the problematic faces and duplicate their vertices to ensure the indices are unique.
        uniques, indices = np.unique(np.sort(triangles, axis=1), axis=0, return_index=True)
        if len(triangles) > len(uniques):
            # boolean mask of the triangles to be updated
            target_faces = np.full(len(triangles), True)
            target_faces[indices] = False

            # indices of the vertices that must be copied
            target_verts = triangles[target_faces].ravel()

            # find the vertices used in problematic faces
            new_vertices = vertices[target_verts]
            new_vertex_weights = vertex_weights[:, target_verts]
            new_vertex_morphs = vertex_morphs[:, target_verts]
            new_vertex_indices = np.arange(len(new_vertices)) + len(vertices)

            # update our final mesh data with new geometry
            vertices = np.vstack((vertices, new_vertices))
            vertex_weights = np.hstack((vertex_weights, new_vertex_weights))
            vertex_morphs = np.hstack((vertex_morphs, new_vertex_morphs))

            # copy is needed since shapes could share data
            triangles = triangles.copy()
            triangles[target_faces] = new_vertex_indices.reshape(-1, 3)
        # '''

        return ImportMeshBuffers(
            triangles=triangles,
            vertices=vertices,
            normals=normals,
            uv_sets=uv_sets,
            vertex_colors=vertex_colors,
            skin=ImportSkinBuffers(vertex_weights) if len(vertex_weights) else None,
            morphs=ImportMorphBuffers(vertex_morphs) if len(vertex_morphs) else None,
        )
