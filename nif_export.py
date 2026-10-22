import bpy

import timeit
import pathlib
import collections
from typing import Any, cast

import numpy as np
import numpy.linalg as la

from es3 import nif
from es3.utils import meshoptimizer
from es3.utils.math import ID44, compose, decompose

from dataclasses import replace
from functools import cached_property

from . import nif_utils
from . import nif_switch
from .conversion import export_animation, export_materials, export_skeletons
from .conversion.blender_compat import get_fcurves
from .conversion.bone_axes import export_axis_correction, import_axis_correction
from .conversion.options import ExportOptions, export_options_from_mapping
from .conversion.records import (
    ExportCreationJob,
    ExportMorphBuffers,
    ExportNode,
    ExportNullSlot,
    ExportRole,
    ExportScope,
    ExportSkinBuffers,
    ExportSwitchPlan,
    MaterialPartition,
)
from .conversion.runtime import ExportRuntime

def save(context, filepath, report=None, **config):
    """save the scene to a nif file"""

    print(f"Export File: {filepath}")
    time = timeit.default_timer()

    options = export_options_from_mapping(config)
    runtime = ExportRuntime.from_context(context)
    exporter = Exporter(filepath, options, runtime)
    exporter.execute()

    for warning in exporter.warnings:
        if report:
            report({"WARNING"}, warning)

    time = timeit.default_timer() - time
    print(f"Export Done: {time:.4f} seconds")

    return {"FINISHED"}


class Exporter:
    def __init__(self, filepath, options: ExportOptions, runtime: ExportRuntime):
        self.options = options
        self.runtime = runtime
        self.filepath = pathlib.Path(filepath)
        self.nodes = []
        self.materials = {}
        self.pixel_datas = {}
        self.material_context = export_materials.ExportMaterialContext(
            self.options,
            self.cycle_type,
            self.materials,
            self.pixel_datas,
        )
        self.root_scope = ExportScope()
        self.history = collections.defaultdict(list)
        self.armature_candidates = {}
        self.skeletons = []
        self.skeletons_by_source = {}
        self.colliders = collections.defaultdict(set)
        self.switches = []
        self.errors = []
        self.warnings = []
        self.scene_objects = []
        self.depsgraph = None
        self.world_matrices = {}

    def execute(self):
        bl_objects = self.get_source_objects()

        # resolve heirarchy
        roots = self.resolve_nodes(bl_objects, sequence=self.scene_objects, scope=self.root_scope)

        # nothing is written unless every switch can be exported as authored
        if self.errors:
            raise nif_switch.SwitchError("\n".join(dict.fromkeys(self.errors)))

        # resolve armatures
        if any(self.armature_candidates):
            self.skeletons, self.skeletons_by_source = export_skeletons.resolve_armatures(
                self.armature_candidates,
                self.nodes,
                self.get,
            )
            self.world_matrices = export_skeletons.apply_axis_corrections(self.skeletons, self.nodes)

        # resolve depsgraph
        self.resolve_depsgraph()

        # uniform scale fix
        for root in roots:
            root.ensure_uniform_scale()
        self.world_matrices = export_skeletons.evaluate_node_worlds(self.nodes)
        for skeleton in self.skeletons:
            skeleton.world_matrices = self.world_matrices

        # create ni objects
        creation_schedule = self.build_creation_schedule()
        for job in creation_schedule:
            self.create_job(job)

        # switch children and selections, once every real output exists
        self.finalize_switches()

        data = nif.NiStream()
        data.root = self.get_root_output(roots)
        self.widen_controller_ranges(data)
        data.apply_scale(self.runtime.scale_correction)
        data.apply_time_scale(1 / self.runtime.fps)
        data.merge_properties(ignore={"name", "shine", "specular_color"})
        child_orders = [(node, list(node.children)) for node in data.objects_of_type(nif.NiNode)]
        data.sort()
        for node, children in child_orders:
            node.reorder_children(children)
        self.setup_collision(data)
        self.setup_markers(data)
        data.save(self.filepath)

        # extract x/kf file
        if self.options.extract_keyframe_data:
            self.export_keyframe_data(data)

    # -------
    # RESOLVE
    # -------

    def resolve_nodes(self, bl_objects, parent=None, sequence=None, scope=None):
        if scope is None:
            scope = self.root_scope
        roots = [SceneNode(self, obj, scope, parent) for obj in bl_objects if not obj.parent]
        included = set(bl_objects)

        queue = collections.deque(roots)
        while queue:
            node = queue.popleft()
            if self.process(node):
                self.nodes.append(node)
                self.history[node.scope, node.source].append(node)

                if node.role is ExportRole.EMPTY:
                    self.discover_empty(node)
                elif node.role is ExportRole.ARMATURE:
                    self.discover_armature(node)

                if nif_switch.is_switch(node.source):
                    # slots follow the scene sequence; null-slot markers stay out of node creation
                    children = nif_switch.ordered_children(node.source, sequence or bl_objects)
                    slots = tuple(
                        ExportNullSlot(child) if nif_switch.is_null_slot(child) else SceneNode(self, child, scope, node)
                        for child in children
                        if child in included
                    )
                    plan = node.switch_plan = ExportSwitchPlan(node, slots)
                    self.switches.append(plan)
                    queue.extend(slot for slot in slots if isinstance(slot, ExportNode))
                elif hasattr(node.source, "children"):
                    queue.extend(SceneNode(self, child, scope, node) for child in bl_objects if child.parent == node.source)
            elif any(nif_switch.is_switch(p.source) for p in node.parents):
                self.errors.append(f"'{node.source.name}' is inside a switch-family node but its type ({node.source.type}) cannot be exported")

        return roots

    def finalize_switches(self):
        for plan in self.switches:
            node = plan.owner
            slots = plan.slots
            node.output.replace_children([slot.output if isinstance(slot, ExportNode) else None for slot in slots])
            targets = [slot.source for slot in slots]
            if nif_switch.is_lod(node.source):
                # data.apply_scale converts these to game units later and leaves no-limit untouched
                node.output.active_index = 0
                node.output.lod_center = np.array(node.source.mw.lod_center, dtype="<f")
                ranges = nif_switch.lod_ranges(node.source, targets)
                node.output.lod_levels = np.array(ranges, dtype="<f").reshape(-1, 2)
            else:
                node.output.active_index = nif_switch.active_index(node.source, targets)

    def resolve_armatures(self):
        self.skeletons, self.skeletons_by_source = export_skeletons.resolve_armatures(
            self.armature_candidates,
            self.nodes,
            self.get,
        )

    def resolve_armature(self, skeleton):
        export_skeletons.resolve_armature(skeleton, self.nodes, self.get)

    def apply_axis_corrections(self):
        self.world_matrices = export_skeletons.apply_axis_corrections(self.skeletons, self.nodes)

    def apply_axis_correction(self, skeleton):
        self.world_matrices = export_skeletons.apply_axis_corrections([skeleton], self.nodes)

    def resolve_depsgraph(self):
        temp_modifiers = []
        temp_hide_view = []
        temp_mute_keys = []
        temp_unhidden = []
        try:
            meshes = [node.source for node in self.nodes if node.role is ExportRole.MESH]
            for source in meshes:
                # hidden switch variants still export; disabled objects are not evaluated
                if source.hide_viewport:
                    source.hide_viewport = False
                    temp_unhidden.append(source)

                visible_modifiers = [m for m in source.modifiers if m.show_viewport]
                is_triangulated = False

                # ensure meshes are in rest pose
                for m in visible_modifiers:
                    if m.type == "TRIANGULATE":
                        is_triangulated = True
                    elif m.type == "ARMATURE":
                        m.show_viewport = False
                        temp_hide_view.append(m)

                # ensure meshes are triangulated
                if not is_triangulated:
                    m = source.modifiers.new("", "TRIANGULATE")
                    m.quad_method = "FIXED"
                    m.ngon_method = "CLIP"
                    if bpy.app.version < (4, 2, 0):
                        m.keep_custom_normals = True
                    temp_modifiers.append((source, m))

                # ensure all shape keys are mute
                if source.data.shape_keys:
                    for k in source.data.shape_keys.key_blocks:
                        if not k.mute:
                            k.mute = True
                            temp_mute_keys.append(k)

            # get evaluated dependency graph
            self.depsgraph = self.runtime.context.evaluated_depsgraph_get()
        finally:
            for m in temp_hide_view:
                m.show_viewport = True
            for s, m in temp_modifiers:
                s.modifiers.remove(m)
            for k in temp_mute_keys:
                k.mute = False
            for source in temp_unhidden:
                source.hide_viewport = True

    def export_keyframe_data(self, data):
        nif_path = self.filepath
        xnif_path = nif_path.with_name("x" + nif_path.name)
        xkf_path = xnif_path.with_suffix(".kf")
        data.extract_keyframe_data().save(xkf_path)
        data.save(xnif_path)

    # -------
    # PROCESS
    # -------

    def process(self, node):
        if isinstance(node.source, bpy.types.PoseBone):
            return self.process_bone(node)
        if node.source.type == "EMPTY":
            return self.process_empty(node)
        if node.source.type == "MESH":
            return self.process_mesh(node)
        if node.source.type == "ARMATURE":
            return self.process_armature(node)

        print(f"Warning: Unsupported Type ({node.source.type})")
        return False

    def process_empty(self, node):
        node.role = ExportRole.EMPTY
        return True

    def process_mesh(self, node):
        node.role = ExportRole.MESH
        return True

    def process_armature(self, node):
        node.role = ExportRole.ARMATURE
        return True

    def process_bone(self, node):
        node.role = ExportRole.BONE
        node_type = node.source.mw.node_type
        if node_type != nif_switch.NODE:
            self.errors.append(f"bone '{node.source.name}' is marked as a {node_type}; skeletal {node_type} is not supported yet")
        return True

    def discover_empty(self, node):
        if node.source.instance_type == 'COLLECTION':
            collection = node.source.instance_collection
            self.resolve_nodes(list(collection.objects), parent=node, scope=ExportScope(node.scope))

        if nif_switch.is_switch(node.source):
            return  # an explicit type outranks name conventions

        if nif_switch.is_collision(node.source):
            self.colliders[node.scope, node.source].update(self.blender_descendants(node.source))
        elif not any(scope is node.scope for scope, _source in self.colliders) and node.name.lower().startswith("collision"):
            self.colliders[node.scope, node.source].update(self.blender_descendants(node.source))

    def discover_armature(self, node):
        bones = tuple(node.source.pose.bones)
        self.armature_candidates[node.scope, node.source] = bones
        self.resolve_nodes(bones, parent=node, scope=node.scope)

    @staticmethod
    def blender_descendants(source):
        queue = collections.deque(source.children)
        while queue:
            child = queue.popleft()
            yield child
            queue.extend(child.children)

    # -------
    # UTILITY
    # -------

    def build_creation_schedule(self):
        owned_bones = set()
        for skeleton in self.skeletons:
            duplicate_bones = owned_bones.intersection(skeleton.bones)
            if duplicate_bones:
                raise RuntimeError("Each exported bone must belong to exactly one resolved skeleton")
            owned_bones.update(skeleton.bones)

        jobs = []
        scheduled_bones = set()
        scheduled_targets = set()
        for target in self.nodes:
            if target.role is ExportRole.BONE:
                continue
            if target in scheduled_targets:
                raise RuntimeError("An export target cannot have more than one creation job")
            if target.role is ExportRole.ARMATURE:
                skeleton = self.skeletons_by_source[target.scope, target.source]
                duplicate_bones = scheduled_bones.intersection(skeleton.bones)
                if duplicate_bones:
                    raise RuntimeError("Each exported bone must be owned by exactly one armature creation job")
                jobs.append(ExportCreationJob(target, target.role, skeleton.bones))
                scheduled_bones.update(skeleton.bones)
            elif target.role in (ExportRole.EMPTY, ExportRole.MESH):
                jobs.append(ExportCreationJob(target, target.role))
            else:
                raise RuntimeError(f"Unsupported export creation role: {target.role}")
            scheduled_targets.add(target)

        if scheduled_bones != owned_bones:
            raise RuntimeError("Each exported bone must be owned by exactly one armature creation job")

        jobs_by_target = {job.target: job for job in jobs}
        ordered_jobs = []
        ordered_targets = set()

        def append_with_dependencies(job):
            if job.target in ordered_targets:
                return
            if job.role is ExportRole.MESH:
                find_armature = getattr(job.target.source, "find_armature", None)
                armature_source = find_armature() if find_armature is not None else None
                if armature_source is not None:
                    skeleton = self.get_skeleton(armature_source, job.target.scope, job.target.name)
                    append_with_dependencies(jobs_by_target[skeleton.armature])
            ordered_jobs.append(job)
            ordered_targets.add(job.target)

        for job in jobs:
            append_with_dependencies(job)

        return tuple(ordered_jobs)

    def create_job(self, job):
        target = job.target
        if job.role is ExportRole.EMPTY:
            Empty(target).create()
        elif job.role is ExportRole.MESH:
            Mesh(target).create()
        elif job.role is ExportRole.ARMATURE:
            Armature(target).create(job.owned_bones)
        else:
            raise RuntimeError(f"Unsupported export creation role: {job.role}")
        parent_world = self.world_matrices[target.parent] if target.parent is not None else None
        export_animation.export_object_animation(
            target.source,
            target.name,
            target.output,
            parent_world,
            target.axis_correction,
            self.options,
            self.cycle_type,
            self.warn,
        )

    def get(self, source, scope):
        nodes = self.history[scope, source]
        if len(nodes) != 1:
            raise nif_switch.SwitchError(
                f"Expected one export occurrence of '{source.name}' in its collection scope, found {len(nodes)}"
            )
        return nodes[0]

    def get_skeleton(self, source, scope, user_name):
        current_scope = scope
        while current_scope is not None:
            skeleton = self.skeletons_by_source.get((current_scope, source))
            if skeleton is not None:
                return skeleton
            current_scope = current_scope.parent
        raise nif_switch.SwitchError(
            f"Armature '{source.name}' used by '{user_name}' has no unambiguous included occurrence in its collection scope"
        )

    def warn(self, message):
        print(f"Warning: {message}")
        self.warnings.append(message)

    def get_source_objects(self):
        if self.options.use_active_collection:
            active_layer_collection = self.runtime.view_layer.active_layer_collection
            if active_layer_collection is None:
                raise RuntimeError("NIF export requires an active collection")
            base_objects = list(active_layer_collection.collection.all_objects)
        else:
            base_objects = list(self.runtime.view_layer.objects)

        if self.options.use_selection:
            allowed = set(self.runtime.context.selected_objects)
        else:
            allowed = set(self.runtime.context.visible_objects)

        # Preserve Blender's sequence, while ensuring every included object has a
        # deterministic, top-down path to the scene root.
        objects = []
        seen = set()
        for ob in base_objects:
            if ob not in allowed or ob in seen:
                continue
            chain = []
            current = ob
            while current is not None and current not in seen:
                chain.append(current)
                current = current.parent
            for current in reversed(chain):
                seen.add(current)
                objects.append(current)

        objects = self.expand_switches(objects)

        # show a useful error message if no valid objects were discovered
        if len(objects) == 0:
            if self.options.use_active_collection:
               raise Exception("No valid objects found. Ensure you have an active collection when using the 'Only Active Collection' option.")
            elif self.options.use_selection:
               raise Exception("No valid objects found. Ensure you have at least one object selected when using the 'Only Selected' option.")
            else:
               raise Exception("No valid objects found. Ensure you have at least one object present in your scene.")

        return objects

    def expand_switches(self, objects):
        """Include the whole subtree of every included switch, whatever its selection or visibility."""
        self.scene_objects = list(nif_switch.scene_objects(self.runtime.scene))
        layer_objects = set(self.runtime.view_layer.objects)

        expanded = []
        placed = set()
        for ob in objects:
            if ob in placed:
                continue
            placed.add(ob)
            expanded.append(ob)
            if not nif_switch.is_switch(ob):
                continue
            # descendants in scene order; nested switches are covered by their outermost switch
            for other in self.scene_objects:
                parent = other.parent
                while parent is not None and parent != ob:
                    parent = parent.parent
                if parent is None or other in placed:
                    continue
                placed.add(other)
                expanded.append(other)
                if other not in layer_objects:
                    self.errors.append(
                        f"'{other.name}' in '{ob.name}' is not in the view layer; include its collection"
                    )

        for ob in expanded:
            node_type = ob.mw.node_type
            if node_type != nif_switch.NODE and ob.type != "EMPTY":
                self.errors.append(f"'{ob.name}' is marked as a {node_type}, but only Empties can be")
            elif nif_switch.is_switch_node(ob):
                children = nif_switch.ordered_children(ob, self.scene_objects)
                self.errors.extend(nif_switch.switch_node_problems(ob, children))
            elif nif_switch.is_lod(ob):
                children = nif_switch.ordered_children(ob, self.scene_objects)
                self.errors.extend(nif_switch.lod_problems(ob, children))
                for warning in nif_switch.lod_warnings(ob, children):
                    self.warn(warning)
            if nif_switch.is_null_slot(ob):
                self.errors.extend(nif_switch.null_slot_problems(ob, self.scene_objects))

        return expanded

    def get_root_output(self, roots):
        file_name = self.filepath.name
        root = nif.NiNode(name=file_name, children=[r.output for r in roots])

        # if there was only one root and it had no transforms, use it as the file root
        if len(roots) == 1:
            no_transforms = np.allclose(roots[0].matrix_local, ID44, rtol=0, atol=1e-4)
            if no_transforms or self.options.preserve_root_transforms:
                root = roots[0].output

        # if there are controllers but no text keys, we'll wrap in a NiBSAnimationNode
        needs_animation_node = False

        if self.options.export_animations and not self.options.extract_keyframe_data:
            for node in root.descendants(breadth_first=True):
                if node.extra_datas.find_type(nif.NiTextKeyExtraData):
                    needs_animation_node = False
                    break
                if node.controller:
                    needs_animation_node = True

        if needs_animation_node:
            replaced_root = None
            # if conditions are met we can update in-place rather than using a wrapper
            in_place =  type(root) is nif.NiNode and root.name not in ("Bip01", "Root Bone")
            if in_place:
                # `asdict()` snapshots the child list; construction transfers each child to the replacement.
                replaced_root = root
                kwargs = root.asdict()
            else:
                kwargs = dict(name=file_name, children=[root])
            root = nif.NiBSAnimationNode(**kwargs)
            if in_place:
                assert replaced_root is not None
                replaced_root.replace_children(())  # clear null slots left by engine-style tail detachment
            root.animated = True
            root.not_random = not self.options.randomize_animations

        return root

    def iter_bones(self, root):
        """yields bone nodes in hierarchical order"""
        yield from self.skeletons_by_source[root.scope, root.source].bones

    def setup_collision(self, data):
        # without "RCN" the engine only looks for a RootCollisionNode among the root's direct children
        colliders = list(data.objects_of_type(nif.RootCollisionNode))
        if colliders and not any(c in data.root.children for c in colliders):
            # must precede "MRK": the engine stops scanning the root's extra data at "MRK"
            data.root.extra_datas.append(nif.NiStringExtraData(string_data="RCN"))

    def setup_markers(self, data):
        markers = [
            ob for ob in data.objects_of_type(nif.NiTriShape)
            if ob.name.lower().startswith("tri editormarker")
        ]
        if markers:
            data.root.extra_datas.append(nif.NiStringExtraData(string_data="MRK"))
            # set ambient colors to play nicely with TESCS toggle lighting feature
            for marker in markers:
                material = marker.get_property(nif.NiMaterialProperty)
                if material:
                    material.ambient_color = material.diffuse_color

    @property
    def cycle_type(self):
        return "CLAMP" if self.options.extract_keyframe_data else "CYCLE"

    def widen_controller_ranges(self, data):
        # The engine loops each controller over its own [start_time, stop_time] and holds the end keys
        # outside them, so covering the scene range keeps every loop in step with the Blender timeline.
        # CLAMP controllers already hold their end keys, and keep the range of their own keys.
        for obj in data.objects():
            if isinstance(obj, nif.NiTimeController) and obj.cycle_type != "CLAMP":
                obj.start_time = min(obj.start_time, self.runtime.frame_start)
                obj.stop_time = max(obj.stop_time, self.runtime.frame_end)


class SceneNode(ExportNode):
    def __init__(self, exporter, source, scope, parent=None):
        if isinstance(source, bpy.types.PoseBone):
            matrix_local = np.asarray(source.bone.matrix_local, dtype="<f")
        else:
            matrix_local = np.asarray(source.matrix_local, dtype="<f")
        super().__init__(exporter, source, matrix_local, scope)
        self.parent = parent

    def __repr__(self):
        if not self.parent:
            return f'SceneNode("{self.name}", parent=None)'
        return f'SceneNode("{self.name}", parent="{self.parent.source.name}")'

    def create(self, *args, **kwargs):
        raise NotImplementedError

    @cached_property
    def name(self):
        base_name = self.strip_blender_suffix(self.source.name)

        # Always strip if the export option is active.
        if self.exporter.options.strip_numeric_suffixes:
            return base_name

        # The engine requires exact matches for these.
        if base_name in ("Bip01", "Bounding Box", "PickProxy", "Root Bone"):
            return base_name

        return self.source.name

    @property
    def bone_name(self):
        name = self.name
        if name.startswith("Bip01"):
            if name.endswith(".L"):
                return f"Bip01 L {name[6:-2]}"
            if name.endswith(".R"):
                return f"Bip01 R {name[6:-2]}"
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
        raise NotImplementedError

    @property
    def axis_correction(self):
        return export_axis_correction(self.name)

    @property
    def axis_correction_inverse(self):
        return import_axis_correction(self.name)

    @property
    def is_bounding_box(self):
        return self.name.lower().startswith("bounding box")

    @property
    def is_collider(self):
        return any(
            scope is self.scope and self.source in descendants
            for (scope, _root), descendants in self.exporter.colliders.items()
        )

    def ensure_uniform_scale(self):
        if self.is_bounding_box:
            return

        linear = self.matrix_local[:3, :3]
        axis_lengths = la.norm(linear, axis=0)
        if np.any(axis_lengths < 1e-8):
            raise nif_switch.SwitchError(f"'{self.name}' has a singular transform; zero scale cannot be exported")
        normalized = linear / axis_lengths
        if not np.allclose(normalized.T @ normalized, np.eye(3), rtol=0, atol=1e-4):
            raise nif_switch.SwitchError(f"'{self.name}' has shear, which cannot be represented by a NIF transform")

        t, r, s = decompose(self.matrix_local)

        if not np.allclose(s[:1], s[1:], rtol=0, atol=0.001):
            print(f"[INFO] {self.source.name} has non-uniform scale")

            if isinstance(self.source, bpy.types.PoseBone):
                raise nif_switch.SwitchError(
                    f"Bone '{self.name}' has non-uniform scale, which cannot be represented without changing its bind pose"
                )
            if self.source.type == "MESH":
                self.baked_vertex_scale = s

            self.matrix_local = compose(t, r, 1)
            scale_matrix = ID44.copy()
            scale_matrix[:3, :3] = np.diag(s)
            for child in self.children:
                child.matrix_local = scale_matrix @ child.matrix_local

        for child in self.children:
            cast(SceneNode, child).ensure_uniform_scale()

    @staticmethod
    def strip_blender_suffix(name):
        """Remove Blender's automatic numeric suffixes like .001, .002, etc."""
        base_name, *suffix = name.rsplit('.', 1)
        if suffix and suffix[0].isnumeric():
            return base_name
        return name

class Empty(SceneNode):
    __slots__ = ()

    def __init__(self, node):
        self.__dict__ = node.__dict__

    def create(self, data=None):
        if isinstance(data, nif.NiTriShapeData):
            self.output = nif.NiTriShape(data=data)
        else:
            if nif_switch.is_switch_node(self.source):
                self.output = nif.NiSwitchNode()
            elif nif_switch.is_lod(self.source):
                self.output = nif.NiLODNode()
            elif (self.scope, self.source) in self.exporter.colliders:
                self.output = nif.RootCollisionNode(app_culled=True)
            elif self.name == "PickProxy":
                self.output = nif.NiCollisionSwitch(propagate=True)
            else:
                self.output = nif.NiNode()

        # set fields
        self.output.name = self.name
        self.output.matrix = self.matrix_local

        # set parent
        if self.parent:
            self.parent.output.attach_child(self.output)

        # set flags
        self.output.flags |= self.source.mw.object_flags

        # set hidden
        if self.output.is_shadow:
            self.output.app_culled = True

        # set bounds
        if self.is_bounding_box:
            self.create_bounding_volume()

        return self.output

    def create_bounding_volume(self):
        placement = self.exporter.get(self.source, self.scope)
        t, r, s = decompose(self.exporter.world_matrices[placement])
        s *= self.source.empty_display_size
        self.output.bounding_volume = nif.NiBoxBV(center=t, axes=r, extents=s)
        self.output.matrix = ID44


class Armature(SceneNode):
    __slots__ = ()

    def __init__(self, node):
        self.__dict__ = node.__dict__

    def create(self, bone_nodes):
        # create root
        Empty(self).create()
        # create bones
        for node in bone_nodes:
            Empty(node).create()
            node.output.name = node.bone_name
            parent_world = self.exporter.world_matrices[node.parent] if node.parent is not None else None
            export_animation.export_object_animation(
                node.source,
                node.name,
                node.output,
                parent_world,
                node.axis_correction,
                self.exporter.options,
                self.exporter.cycle_type,
                self.exporter.warn,
            )


class Mesh(SceneNode):
    __slots__ = ()

    def __init__(self, node):
        self.__dict__ = node.__dict__

    def create(self):
        bl_object, bl_data = self.get_mesh_data()

        ni_data = nif.NiTriShapeData()

        self.create_triangles(ni_data, bl_data)
        self.create_vertices(ni_data, bl_data)

        self.create_normals(ni_data, bl_data)
        self.create_vertex_colors(ni_data, bl_data)
        self.create_uv_sets(ni_data, bl_data)

        skin_data = self.get_skin_data(bl_data, ni_data)
        morph_data = self.get_morph_data(bl_data, ni_data)
        material_data = self.get_material_data(bl_object, bl_data)
        bl_object.to_mesh_clear()  # bl_data is freed here

        if len(material_data) == 1:
            outputs = [Empty(self).create(ni_data)]
        else:
            # Create a trishape for each material.
            parent_output = Empty(self).create()
            outputs = [nif.NiTriShape() for _ in material_data]
            parent_output.replace_children(outputs)

        for i, tri in enumerate(outputs):
            tri.name = tri.name or f"Tri {self.name} {i}"
            tri.data = tri.data or nif.NiTriShapeData()
            tri_skin = replace(skin_data) if skin_data is not None else None
            tri_morphs = replace(morph_data) if morph_data is not None else None

            partition = material_data[i]
            export_materials.export_material(
                partition.material,
                tri,
                self.source,
                self.exporter.material_context,
                is_collider=self.is_collider,
            )

            self.extract_vertex_data(
                partition.corner_indices,
                ni_data,
                tri.data,
                skin_data,
                tri_skin,
                morph_data,
                tri_morphs,
            )
            self.optimize_geometry(tri.data, tri_skin, tri_morphs)
            self.create_vertex_groups(tri, tri_skin)
            self.create_vertex_morphs(tri, tri_morphs)

            tri.data.update_center_radius(exact=True)

        return self.output

    def create_vertices(self, ni, bl):
        ni.vertices.resize(len(bl.vertices), 3)
        bl.vertices.foreach_get("co", ni.vertices.ravel())

        # Ensure vertex transforms.
        # TODO refactor this!
        if self.baked_vertex_scale is not None:
            ni.vertices *= self.baked_vertex_scale

        if len(ni.triangles):
            ni.vertices = ni.vertices[ni.triangles].reshape(-1, 3)

    def create_triangles(self, ni, bl):
        ni.triangles.resize(len(bl.polygons), 3)
        bl.polygons.foreach_get("vertices", ni.triangles.ravel())

    def create_normals(self, ni, bl):
        if self.is_collider:
            return
        if bl["ignore_normals"]:
            return

        ni.normals.resize(len(ni.vertices), 3)
        if bpy.app.version >= (4, 1, 0):
            bl.corner_normals.foreach_get("vector", ni.normals.ravel())
        else:
            bl.calc_normals_split()
            bl.loops.foreach_get("normal", ni.normals.ravel())
            bl.free_normals_split()

        if self.baked_vertex_scale is not None:
            ni.normals /= self.baked_vertex_scale
        ni.normals /= la.norm(ni.normals, axis=1, keepdims=True)

    def create_uv_sets(self, ni, bl):
        if self.is_collider:
            return
        if not any(uv.data for uv in bl.uv_layers):
            return

        ni.uv_sets.resize(len(bl.uv_layers), len(ni.vertices), 2)
        for ni_uv, bl_uv in zip(ni.uv_sets, bl.uv_layers):
            bl_uv.data.foreach_get("uv", ni_uv.ravel())
            # convert Blender into OpenGL format
            ni_uv[..., 1] = 1 - ni_uv[..., 1]

    def create_vertex_colors(self, ni, bl):
        if self.is_collider:
            return
        if not any(self.source.material_slots):
            return

        if len(bl.vertex_colors):
            ni.vertex_colors.resize(len(ni.vertices), 4)
            bl.vertex_colors[0].data.foreach_get("color", ni.vertex_colors.ravel())

    def create_vertex_groups(self, trishape, skin_data):
        if skin_data is None:
            return

        missing = [node.name for node in (skin_data.root, *skin_data.bones) if node.output is None]
        if missing:
            raise RuntimeError(f"Skin references skeleton outputs that have not been created: {', '.join(missing)}")

        skin = trishape.skin = nif.NiSkinInstance()
        skin.root, *skin.bones = [node.output for node in (skin_data.root, *skin_data.bones)]

        skin.data = nif.NiSkinData()
        skin.data.bone_data = [nif.NiSkinDataBoneData() for _ in skin_data.bones]

        # root to skin offset
        placement = self.exporter.get(self.source, self.scope)
        offset = la.solve(self.exporter.world_matrices[skin_data.root], self.exporter.world_matrices[placement])
        skin.data.matrix = la.inv(offset)

        # calculate bone data
        for i, bone_data in enumerate(skin.data.bone_data):
            j, = np.nonzero(skin_data.weights[i])

            # skip empty weights
            if len(j) == 0:
                cast(Any, skin.bones)[i] = None
                cast(Any, skin.data.bone_data)[i] = None
                continue

            # set vertex weights
            bone_data.vertex_weights.resize(len(j))
            bone_data.vertex_weights["f0"] = j
            bone_data.vertex_weights["f1"] = skin_data.weights[i, j]

            # skin to bone offset
            matrix_bind = skin_data.bind_relative_to_root[skin_data.bones[i]]
            bone_data.matrix = la.solve(matrix_bind, offset)

            # apply center/radius
            bone_data.update_center_radius(trishape.data.vertices[j], exact=True)

        # filter unused bones
        skin.bones = [*filter(None, skin.bones)]
        skin.data.bone_data = [*filter(None, skin.data.bone_data)]

    def create_vertex_morphs(self, trishape, morph_data):
        if morph_data is None:
            return

        # create controller
        controller = nif.NiGeomMorpherController(
            cycle_type=self.exporter.cycle_type,
            target=trishape,
        )
        controller.stop_time = max(keys[-1, 0] for keys in morph_data.keys if len(keys))

        # create morph data
        controller.data = nif.NiMorphData(relative_targets=1)
        controller.data.targets = [nif.NiMorphDataMorphTarget() for _ in morph_data.targets]

        # assign morph data
        for i, target in enumerate(controller.data.targets):
            target.key_type = morph_data.interpolations[i]
            target.keys = morph_data.keys[i]
            target.vertices = morph_data.targets[i]

        trishape.controllers.appendleft(controller)

    # -- get blender data --

    def get_mesh_data(self):
        # Resolve dependency graph.
        dg = self.exporter.depsgraph

        # Evaluate final mesh data.
        bl_object = self.source.evaluated_get(dg)
        bl_data = bl_object.to_mesh(preserve_all_data_layers=True, depsgraph=dg)

        # Validate mesh + material.
        bl_data.validate(clean_customdata=False)
        bl_data.validate_material_indices()

        # Preserve our custom data.
        bl_data["ignore_normals"] = self.source.data.get("ignore_normals")

        return bl_object, bl_data

    def get_skin_data(self, bl_data, ni_data):
        # ignore collider meshes
        if self.is_collider:
            return None

        # find the armature root
        armature = self.source.find_armature()
        if not armature:
            return None
        skeleton = self.exporter.get_skeleton(armature, self.scope, self.name)
        bone_nodes = {node.source: node for node in skeleton.bones}

        # find valid bone groups
        bones = {}
        for i, vg in enumerate(self.source.vertex_groups):
            try:
                bones[i] = armature.pose.bones[vg.name]
            except KeyError:
                pass

        # allocate weights array
        weights = np.zeros([len(bones), len(bl_data.vertices)])
        if weights.size == 0:
            return None

        # populate weights array
        index_remap = {j: i for i, j in enumerate(bones)}
        for i, vertex in enumerate(bl_data.vertices):
            for vg in vertex.groups:
                try:
                    j = index_remap[vg.group]
                    weights[j, i] = vg.weight
                except KeyError:
                    pass

        # limit bones per vertex
        if len(bones) > 4:
            # indices of all but the 4 highest weights
            indices = np.argpartition(weights, -4, axis=0)[:-4]
            # set the values at target indices to zero
            np.put_along_axis(weights, indices, 0, axis=0)

        # normalize bone weights
        sums = weights.sum(axis=0)
        mask = sums == 0
        # assign unweighted vertices to the first bone
        weights[0, mask] = sums[mask] = 1.0
        weights /= sums

        # view as per-face-loops
        weights = weights[:, ni_data.triangles].reshape(len(bones), -1)

        return ExportSkinBuffers(
            root=skeleton.armature,
            bones=tuple(bone_nodes[bone] for bone in bones.values()),
            weights=weights,
            bind_relative_to_root=skeleton.bind_relative_to_root,
        )

    def get_morph_data(self, bl_data, ni_data):
        # respect export setting
        if not self.exporter.options.export_animations:
            return None

        # ignore collider meshes
        if self.is_collider:
            return None

        # collect shape key data
        try:
            shape_keys = self.source.data.shape_keys
            basis = shape_keys.key_blocks[0]
        except (AttributeError, IndexError):
            return None

        fcurves = get_fcurves(shape_keys.animation_data)

        data_paths = {fc.data_path: fc for fc in fcurves}

        # isolate unmuted layers (a target without keys never contributes in the engine)
        fcurves_dict = {None: basis}
        for sk in shape_keys.key_blocks:
            fc = data_paths.get(sk.path_from_id('value'))
            if fc and len(fc.keyframe_points) and not (fc.mute or sk.mute):
                fcurves_dict[fc] = sk
                assert sk.relative_key == basis, "Shape keys must be relative to basis."

        # bail if all were muted or empty
        if not len(fcurves_dict) > 1:
            return None

        # allocate keyframe data
        keyframes = []
        interpolations = []

        # allocate morph targets
        targets = np.empty((len(fcurves_dict), len(bl_data.vertices), 3))

        for i, (fc, sk) in enumerate(fcurves_dict.items()):
            if i == 0:
                # is basis layer
                key_type = nif.NiFloatData.KeyType.LIN_KEY
                key_data = np.empty(0)
            else:
                key_type = export_animation.get_interpolation_type([fc])
                key_data = export_animation.collect_keyframe_points([fc], key_type)

            # populate keyframe data
            keyframes.append(key_data)
            interpolations.append(key_type)

            # populate morph targets
            sk.data.foreach_get("co", targets[i].ravel())

        # make relative to basis
        targets[1:] -= targets[0]

        if self.baked_vertex_scale is not None:
            targets *= self.baked_vertex_scale

        # view as per-face-loops
        targets = targets[:, ni_data.triangles].reshape(-1, len(ni_data.vertices), 3)

        return ExportMorphBuffers(
            targets=targets,
            keys=tuple(keyframes),
            interpolations=tuple(interpolations),
        )

    def get_material_data(self, bl_object, bl_data):
        # get each polygon's material index
        material_indices = np.empty(len(bl_data.polygons), dtype=int)
        bl_data.polygons.foreach_get("material_index", material_indices)

        # reduce to only the active indices
        active_material_indices = np.unique(material_indices)

        # simple case for a single material
        if len(active_material_indices) <= 1:
            return [MaterialPartition(bl_object.active_material, None)]

        # get each polygon's vertex indices
        loops = np.arange(len(bl_data.loops), dtype=int).reshape(-1, 3)

        material_data = []
        for i in active_material_indices:
            material = bl_object.material_slots[i].material
            indices = loops[material_indices == i].ravel()
            if len(indices):
                material_data.append(MaterialPartition(material, indices))

        return material_data

    # -- optimizer functions --

    def optimize_geometry(self, data, skin_data, morph_data):
        if len(data.vertices) == 0:
            return

        arrays = [
            data.vertices,
            data.normals,
            data.vertex_colors,
            *data.uv_sets,
            # *(skin_data.weights if skin_data is not None else ()),
            # *(morph_data.targets if morph_data is not None else ()),
        ]

        if len(data.uv_sets):
            # Calculate the handedness (winding direction) of the UVs for each triangle.
            # We pass this into `unique_rows` to prevent vertices from being merged if their
            # parent triangles have mirrored UV layouts, which causes tangent/bitangent
            # generation to break in OpenMW.
            uvs = data.uv_sets[0].reshape(-1, 3, 2)
            d_uv_1 = uvs[:, 1] - uvs[:, 0]
            d_uv_2 = uvs[:, 2] - uvs[:, 0]
            cross_z = (d_uv_1[:, 0] * d_uv_2[:, 1]) - (d_uv_1[:, 1] * d_uv_2[:, 0])
            cross_z[np.abs(cross_z) < 1e-6] = 0
            arrays.append(np.sign(cross_z).repeat(3))

        # clear duplicates
        indices, inverse = nif_utils.unique_rows(*arrays, precision=self.exporter.options.vertex_precision)

        # update face data
        try:
            data.triangles = inverse[np.arange(inverse.size, dtype=int).reshape(-1, 3)]
        except ValueError:
            raise ValueError(f"Invalid geometry data (no faces?): {self}")

        # update vert data
        self.extract_vertex_data(indices, data, data, skin_data, skin_data, morph_data, morph_data)
        self.optimize_vertex_cache(data, skin_data, morph_data)

    @staticmethod
    def extract_vertex_data(indices, src_data, dst_data, src_skin, dst_skin, src_morphs, dst_morphs):
        indices = ... if indices is None else indices
        if len(src_data.vertices):
            dst_data.vertices = src_data.vertices[indices]
        if len(src_data.normals):
            dst_data.normals = src_data.normals[indices]
        if len(src_data.vertex_colors):
            dst_data.vertex_colors = src_data.vertex_colors[indices]
        if len(src_data.uv_sets):
            dst_data.uv_sets = src_data.uv_sets[:, indices]
        if src_skin is not None and dst_skin is not None:
            dst_skin.weights = src_skin.weights[:, indices]
        if src_morphs is not None and dst_morphs is not None:
            dst_morphs.targets = src_morphs.targets[:, indices]

    @staticmethod
    def optimize_vertex_cache(data, skin, morphs):
        vertex_remap, triangles = meshoptimizer.optimize(data.vertices, data.triangles.astype(np.uint32))
        if len(data.triangles):
            data.triangles = triangles.astype(np.uint16)
        if len(data.vertices):
            data.vertices[vertex_remap] = data.vertices.copy()
        if len(data.normals):
            data.normals[vertex_remap] = data.normals.copy()
        if len(data.vertex_colors):
            data.vertex_colors[vertex_remap] = data.vertex_colors.copy()
        if len(data.uv_sets):
            data.uv_sets[:, vertex_remap] = data.uv_sets.copy()
        if skin is not None:
            skin.weights[:, vertex_remap] = skin.weights.copy()
        if morphs is not None:
            morphs.targets[:, vertex_remap] = morphs.targets.copy()
