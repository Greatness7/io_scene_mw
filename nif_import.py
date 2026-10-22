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
from . import nif_shader
from . import nif_switch
from .conversion import import_skeletons
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
            Animation(skeleton.root).set_mute(False)

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

    @property
    def animation(self):
        return Animation(self)

    @property
    def material(self):
        return Material(self)


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

        self.animation.create()

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
            node.animation.create()
            node.animation.set_mute(True)

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

        self.material.create()

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

        animation = self.animation

        # add basis key
        ob.shape_key_add(name="Basis")

        # add anim data
        action = animation.get_action(ob.data.shape_keys)

        # add morph keys
        for i, target in enumerate(self.source.morph_targets):
            # create morph targets
            shape_key = ob.shape_key_add(name="")
            shape_key.data.foreach_set("co", vertex_morphs[i].ravel())

            # create morph fcurves
            data_path = shape_key.path_from_id("value")
            fc = animation.create_fcurves(action, data_path)

            # add fcurve keyframes
            fc.keyframe_points.add(len(target.keys))
            fc.keyframe_points.foreach_set("co", target.keys[:, :2].ravel())
            animation.create_interpolation_data(target, fc)
            fc.update()

        # update frame range
        animation.update_frame_range(self.source.controller)

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


class Material(SceneNode):
    __slots__ = ()

    def __init__(self, node):
        self.__dict__ = node.__dict__

    def create(self):
        properties = self.properties
        if len(properties) == 0:
            return

        ni_alpha = properties.get(nif.NiAlphaProperty)
        ni_material = properties.get(nif.NiMaterialProperty)
        ni_stencil = properties.get(nif.NiStencilProperty)
        ni_texture = properties.get(nif.NiTexturingProperty)
        ni_wireframe = properties.get(nif.NiWireframeProperty)

        # Vertex colors need a texturing property present to be visible.
        if (ni_texture is None) and len(self.source.data.vertex_colors):
            ni_texture = nif.NiTexturingProperty()

        # Blender stores wireframe on the object rather than a material.
        if ni_wireframe and ni_wireframe.wireframe:
            self.output.display_type = "WIRE"

        # Re-Use Materials
        name = self.calc_name_from_textures(ni_texture)
        if self.apply_existing_material(name, ni_alpha):
            return

        # Merge Duplicates
        props_hash = (
            *properties.values(),
            # "use_vertex_colors" is stored on the material
            len(self.source.data.vertex_colors),
            # uv animations are also stored on the material
            self.source.controllers.find_type(nif.NiUVController),
        )
        bl_prop = None
        try:
            bl_prop = self.importer.materials[props_hash]
        except KeyError:
            bl_prop = self.importer.materials[props_hash] = nif_shader.execute(self.output)
        else:
            # material already exists, reuse it
            self.link_object_material(self.output, bl_prop.material)
            return
        finally:
            if self.importer.options.use_existing_materials and bl_prop is not None:
                bl_prop.material.name = name
        assert bl_prop is not None

        # Setup Properties
        if ni_alpha:
            self.create_alpha_property(bl_prop, ni_alpha)
        if ni_material:
            self.create_material_property(bl_prop, ni_material)
        if ni_stencil:
            self.create_stencil_property(bl_prop, ni_stencil)
        if ni_texture:
            self.create_texturing_property(bl_prop, ni_texture)
        if ni_wireframe:
            self.create_wireframe_property(bl_prop, ni_wireframe)

    def create_alpha_property(self, bl_prop, ni_prop):
        # Alpha Flags
        bl_prop.alpha_flags = ni_prop.flags
        # Alpha Threshold
        bl_prop.alpha_threshold = float(ni_prop.test_ref / 255)
        # Blending Method
        if ni_prop.alpha_blending:
            bl_prop.use_alpha_blend = True
        if ni_prop.alpha_testing:
            bl_prop.use_alpha_clip = True

    def create_material_property(self, bl_prop, ni_prop):
        # Material Name
        if not self.importer.options.use_existing_materials:
            bl_prop.material.name = ni_prop.name
        # Material Color
        bl_prop.ambient_color[:3] = ni_prop.ambient_color
        bl_prop.diffuse_color[:3] = ni_prop.diffuse_color
        bl_prop.specular_color[:3] = ni_prop.specular_color
        bl_prop.emissive_color[:3] = ni_prop.emissive_color
        # Material Shine
        bl_prop.shine = ni_prop.shine
        # Material Alpha
        bl_prop.alpha = ni_prop.alpha
        # Material Anims
        self.animation.create_color_controller(bl_prop, ni_prop)
        self.animation.create_alpha_controller(bl_prop, ni_prop)

    def create_texturing_property(self, bl_prop, ni_prop):
        # Texture Slots
        for name in nif.NiTexturingProperty.texture_keys:
            self.create_texturing_property_map(bl_prop, ni_prop, name)
        # Vertex Colors
        if self.output.data.vertex_colors:
            bl_prop.vertex_color.layer_name = self.output.data.vertex_colors[0].name
            bl_prop.create_link(bl_prop.vertex_color, bl_prop.shader, "Color", "Diffuse Color")
            bl_prop.create_link(bl_prop.vertex_color, bl_prop.shader, "Alpha", "Diffuse Alpha")
        # UV Animations
        for controller in self.source.controllers:
            if isinstance(controller, nif.NiUVController):
                self.animation.create_uv_controller(controller)

    def create_wireframe_property(self, bl_prop, ni_prop):
        if ni_prop.wireframe:
            self.output.display_type = "WIRE"

    def create_stencil_property(self, bl_prop, ni_prop):
        bl_prop.material.use_backface_culling = False
        bl_prop.material.show_transparent_back = True

    def create_texturing_property_map(self, bl_prop, ni_prop, slot_name):
        try:
            bl_slot = getattr(bl_prop, slot_name)
            ni_slot = getattr(ni_prop, slot_name)
            # only supports slots with texture image attached
            image = self.create_image(ni_slot.source.filename)
        except (AttributeError, LookupError):
            return

        # texture image
        bl_slot.image = image

        # use repeat
        if ni_slot.clamp_mode.name == 'CLAMP_S_CLAMP_T':
            bl_slot.use_repeat = False

        # use mipmaps
        if ni_slot.source.use_mipmaps.name == 'NO':
            bl_slot.use_mipmaps = False

        # uv layer
        try:
            bl_slot.layer = self.output.data.uv_layers[ni_slot.uv_set].name
        except IndexError:
            pass

    def create_image(self, filepath):
        abspath = self.resolve_texture_path(filepath)

        if abspath.exists():
            image = bpy.data.images.load(str(abspath), check_existing=True)
        else:  # placeholder
            image = bpy.data.images.new(name=abspath.name, width=1, height=1)
            image.filepath = str(abspath)
            image.source = "FILE"

        return image

    def calc_name_from_textures(self, ni_prop):
        if not self.importer.options.use_existing_materials:
            return ""

        if ni_prop is None:
            return ""

        names = {}
        for tex_key, tex_map in zip(ni_prop.texture_keys, ni_prop.texture_maps):
            try:
                names[tex_key] = pathlib.Path(tex_map.source.filename).stem.lower()
            except AttributeError:
                pass

        if names.keys() == {"base_texture"}:
            return names["base_texture"]

        return " | ".join(f"{k.rpartition('_')[0]}:{v}" for k, v in names.items())

    def apply_existing_material(self, name, ni_alpha):
        """
        Check if a material with the same name and properties already exists
        in the current blend file. If so, use it instead of making a new one.
        """
        if not self.importer.options.use_existing_materials:
            return

        use_vertex_colors = bool(len(self.source.data.vertex_colors))
        use_alpha_blend = getattr(ni_alpha, "alpha_blending", False)
        use_alpha_clip = getattr(ni_alpha, "alpha_testing", False)

        base_name, index = name, 0
        while True:
            try:
                material = cast(Any, bpy.data.materials[name])
                bl_prop = material.mw.validate()
            except (LookupError, TypeError):
                break

            if (
                bl_prop.use_vertex_colors == use_vertex_colors
                and bl_prop.use_alpha_blend == use_alpha_blend
                and bl_prop.use_alpha_clip == use_alpha_clip
            ):
                self.link_object_material(self.output, bl_prop.material)
                return True

            index += 1
            name = f"{base_name}.{index:03}"

    @staticmethod
    def link_object_material(bl_object, bl_material):
        # Use any existing empty material slot first.
        for slot in bl_object.material_slots:
            if slot.material is None:
                break
        else:
            bl_object.data.materials.append(None)
            slot = bl_object.material_slots[-1]

        slot.link = "OBJECT"
        slot.material = bl_material

    def resolve_texture_path(
        self,
        relpath,
        case_insensitive = pathlib.Path(__file__.upper()).exists()
    ):
        # get the initial filepath
        path = pathlib.Path(bpy.path.native_pathsep(relpath).lower())

        # discard "data files" prefix
        if path.parts[0] == "data files":
            path = path.relative_to("data files")

        # discard "textures" prefix
        if path.parts[0] == "textures":
            path = path.relative_to("textures")

        # potential file extensions
        suffixes = {path.suffix, ".dds", ".tga", ".bmp"}

        # evaluate final image path
        for directory in self.importer.runtime.texture_paths:
            abspath = directory / path
            for suffix in suffixes:
                abspath = abspath.with_suffix(suffix)
                if not case_insensitive:
                    abspath = pathlib.Path(bpy.path.resolve_ncase(str(abspath)))
                if abspath.exists():
                    return abspath

        return pathlib.Path("textures") / path


class Animation(SceneNode):
    __slots__ = ()

    def __init__(self, node):
        self.__dict__ = node.__dict__

    def create(self):
        if self.importer.options.ignore_animations:
            return

        bl_object = self.output.id_data

        if self.source.extra_data:
            self.create_text_keys(bl_object)

        if self.source.controller:
            self.create_kf_controller(bl_object)
            self.create_vis_controller(bl_object)

    def create_text_keys(self, bl_object):
        text_data = self.source.extra_datas.find_type(nif.NiTextKeyExtraData)
        if text_data is None:
            return

        action = self.get_action(bl_object)

        for frame, text in text_data.keys.tolist():
            for name in filter(None, text.splitlines()):
                assert len(name) < 64, f"Marker exceeds character limit ({name})"
                action.pose_markers.new(name).frame = round(frame)

    def create_kf_controller(self, bl_object):
        controller = self.source.controllers.find_type(nif.NiKeyframeController)
        if not (controller and controller.data):
            return

        # get animation action
        action = self.get_action(bl_object)

        # translation keys
        self.create_translations(controller, action)
        # rotation keys
        self.create_rotations(controller, action)
        # scale keys
        self.create_scales(controller, action)

        self.update_frame_range(controller)

    def create_translations(self, controller, action):
        data = controller.data.translations
        if len(data.keys) == 0:
            return

        # get blender data path
        data_path = self.output.path_from_id("location")

        # build blender fcurves
        for i in range(3):
            fc = self.animation.create_fcurves(action, data_path, index=i, action_group=self.bone_name)
            fc.keyframe_points.add(len(data.keys))
            fc.keyframe_points.foreach_set("co", data.keys[:, (0, i+1)].ravel())
            self.create_interpolation_data(data, fc, axis=i)
            fc.update()

    def create_rotations(self, controller, action):
        if controller.data.rotations.euler_data:
            if isinstance(self.output, bpy.types.PoseBone):
                print(f"[INFO] Euler animations on bones are not currently supported. ({self.name})")
                controller.data.rotations.convert_to_quaternions()
            else:
                self.output.rotation_mode = controller.data.rotations.euler_axis_order.name
                self.create_euler_rotations(controller, action)
                return

        self.output.rotation_mode = 'QUATERNION'
        self.create_quaternion_rotations(controller, action)

    def create_euler_rotations(self, controller, action):
        for i, data in enumerate(controller.data.rotations.euler_data):
            if len(data.keys) == 0:
                continue

            # get blender data path
            data_path = self.output.path_from_id("rotation_euler")

            # build blender fcurves
            fc = self.animation.create_fcurves(action, data_path, index=i, action_group=self.output.name)
            fc.keyframe_points.add(len(data.keys))
            fc.keyframe_points.foreach_set("co", data.keys[:, :2].ravel())
            self.create_interpolation_data(data, fc)
            fc.update()

    def create_quaternion_rotations(self, controller, action):
        data = controller.data.rotations
        if len(data.keys) == 0:
            return

        # get blender data path
        data_path = self.output.path_from_id("rotation_quaternion")

        # build blender fcurves
        for i in range(4):
            fc = self.animation.create_fcurves(action, data_path, index=i, action_group=self.output.name)
            fc.keyframe_points.add(len(data.keys))
            fc.keyframe_points.foreach_set("co", data.keys[:, (0, i+1)].ravel())
            fc.update()

    def create_scales(self, controller, action):
        data = controller.data.scales
        if len(data.keys) == 0:
            return

        # get blender data path
        data_path = self.output.path_from_id("scale")

        # build blender fcurves
        for i in range(3):
            fc = self.animation.create_fcurves(action, data_path, index=i, action_group=self.output.name)
            fc.keyframe_points.add(len(data.keys))
            fc.keyframe_points.foreach_set("co", data.keys[:, :2].ravel())
            self.create_interpolation_data(controller.data.scales, fc)
            fc.update()

    def create_vis_controller(self, bl_object):
        controller = self.source.controllers.find_type(nif.NiVisController)
        if controller is None:
            return

        data = controller.data
        if (data is None) or len(data.keys) == 0:
            return

        keys = np.empty((len(data.keys), 2), dtype=np.float32)

        # invert appculled flag
        keys[:, 1] = 1 - data.values

        # get animations action
        action = self.get_action(bl_object)

        # get blender data path
        try:
            data_path = self.output.path_from_id("hide_viewport")
        except AttributeError:
            print(f"Warning: NiVisController on bones are not supported ({self.name})")
            return

        # build blender fcurves
        fc = self.animation.create_fcurves(action, data_path, index=0, action_group=self.output.name)
        fc.keyframe_points.add(len(keys))
        fc.keyframe_points.foreach_set("co", keys.ravel())
        fc.update()

    def create_uv_controller(self, controller):
        if self.importer.options.ignore_animations:
            return

        data = controller.data
        if data is None:
            return

        # get blender property
        try:
            bl_prop = self.output.active_material.mw
        except AttributeError:
            return

        # get animation action
        action = self.get_action(bl_prop.texture_group.node_tree)

        # get the texture slot
        try:
            uv_name = self.output.data.uv_layers[controller.texture_set].name
            bl_slot = next(s for s in bl_prop.texture_slots if s.layer == uv_name)
            bl_node = bl_slot.mapping_node
        except (IndexError, StopIteration):
            print("Warning: skipping NiUVController due to invalid texture set")
            return

        channels = {
            (data.u_offset_data, data.v_offset_data):
                bl_node.inputs["Location"].path_from_id("default_value"),
            (data.u_tiling_data, data.v_tiling_data):
                bl_node.inputs["Scale"].path_from_id("default_value"),
        }

        try:
            # TODO: do these in shader instead
            data.u_offset_data.keys[:, 1] *= -1
            data.v_offset_data.keys[:, 1] *= -1
        except AttributeError:
            pass

        for sources, data_path in channels.items():
            for i, uv_data in enumerate(sources):
                # build blender fcurves
                fc = self.animation.create_fcurves(action, data_path, index=i, action_group=uv_name)
                fc.keyframe_points.add(len(uv_data.keys))
                fc.keyframe_points.foreach_set("co", uv_data.keys[:, :2].ravel())
                self.create_interpolation_data(uv_data, fc)
                fc.update()

        self.update_frame_range(controller)

    def create_color_controller(self, bl_prop, ni_prop):
        if self.importer.options.ignore_animations:
            return

        controller = ni_prop.controllers.find_type(nif.NiMaterialColorController)
        if controller is None:
            return

        data = controller.data
        if data is None:
            return

        keys = controller.data.keys
        if len(keys) == 0:
            return

        # get blender data path
        if controller.color_field == 'DIFFUSE':
            data_path = bl_prop.diffuse_input.path_from_id("default_value")
        elif controller.color_field == 'EMISSIVE':
            data_path = bl_prop.emissive_input.path_from_id("default_value")
        else:
            raise NotImplementedError(f"'{controller.color_field}' animations are not supported")

        # create blender action
        action = self.get_action(bl_prop.material.node_tree)

        # build blender fcurves
        for i in range(3):
            fc = self.animation.create_fcurves(action, data_path, index=i, action_group=bl_prop.material.name)
            fc.keyframe_points.add(len(keys))
            fc.keyframe_points.foreach_set("co", keys[:, (0, i+1)].ravel())
            self.create_interpolation_data(data, fc)
            fc.update()

        self.update_frame_range(controller)

    def create_alpha_controller(self, bl_prop, ni_prop):
        if self.importer.options.ignore_animations:
            return

        controller = ni_prop.controllers.find_type(nif.NiAlphaController)
        if controller is None:
            return

        data = controller.data
        if data is None:
            return

        keys = controller.data.keys
        if len(keys) == 0:
            return

        # create blender action
        action = self.get_action(bl_prop.material.node_tree)

        # get blender data path
        data_path = bl_prop.opacity_input.path_from_id("default_value")

        # build blender fcurves
        fc = self.animation.create_fcurves(action, data_path, index=0, action_group=bl_prop.material.name)
        fc.keyframe_points.add(len(keys))
        fc.keyframe_points.foreach_set("co", keys[:, :2].ravel())
        self.create_interpolation_data(data, fc)
        fc.update()

        self.update_frame_range(controller)

    @staticmethod
    def get_action(bl_object):
        try:
            action = bl_object.animation_data.action
        except AttributeError:
            action = bpy.data.actions.new(f"{bl_object.name}Action")
            anim_data = bl_object.animation_data_create()
            anim_data.action = action

            if bpy.app.version >= (4, 4, 0):
                anim_data.action_slot = action.slots.new(id_type=bl_object.id_type, name=bl_object.name)

        return action

    @staticmethod
    def create_fcurves(action, data_path, index=0, action_group=""):
        if bpy.app.version >= (5, 0, 0):
            from bpy_extras import anim_utils
            channelbag = cast(Any, anim_utils).action_ensure_channelbag_for_slot(action, *action.slots)
            return channelbag.fcurves.ensure(data_path, index=index, group_name=action_group)
        else:
            return action.fcurves.new(data_path, index=index, action_group=action_group)

    @staticmethod
    def create_interpolation_data(ni_data, fcurves, axis: Any = ...):
        if ni_data.key_type.name == 'NO_INTERP':
            for kp in fcurves.keyframe_points:
                kp.interpolation = 'CONSTANT'
        elif ni_data.key_type.name == 'LIN_KEY':
            for kp in fcurves.keyframe_points:
                kp.interpolation = 'LINEAR'
        else:
            for kp in fcurves.keyframe_points:
                kp.interpolation = 'BEZIER'
                kp.handle_left_type = kp.handle_right_type = 'FREE'
            handles = ni_data.get_tangent_handles()  # TODO: call this once per controller rather than per axis
            fcurves.keyframe_points.foreach_set("handle_left", handles[0, axis].ravel())
            fcurves.keyframe_points.foreach_set("handle_right", handles[1, axis].ravel())

    def update_frame_range(self, controller):
        scene = self.importer.runtime.scene
        frame_end = int(np.ceil(controller.stop_time))
        scene.frame_end = scene.frame_preview_end = max(scene.frame_end, frame_end)

    def set_mute(self, state, fcurves=None):
        if self.importer.options.ignore_animations:
            return

        if fcurves is None:
            try:
                fcurves = self.output.id_data.animation_data.action.fcurves
            except AttributeError:
                return
        fcurves.foreach_set("mute", [state] * len(fcurves))
