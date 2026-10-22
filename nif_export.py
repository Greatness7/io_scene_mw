import bpy

import timeit
import pathlib
import collections
from typing import cast

import numpy as np
import numpy.linalg as la

from es3 import nif
from es3.utils.geometry import GeometryError
from es3.utils.math import ID44, compose, decompose

from dataclasses import replace
from functools import cached_property

from . import nif_switch
from .conversion import export_animation, export_geometry, export_materials, export_skeletons
from .conversion.bone_axes import export_axis_correction, import_axis_correction
from .conversion.options import ExportOptions, export_options_from_mapping
from .conversion.records import (
    ExportCreationJob,
    ExportNode,
    ExportNullSlot,
    ExportRole,
    ExportScope,
    ExportSwitchPlan,
)
from .conversion.runtime import ExportRuntime, evaluated_mesh, export_evaluation_scope

def save(context, filepath, report=None, **config):
    """save the scene to a nif file"""

    print(f"Export File: {filepath}")
    time = timeit.default_timer()

    options = export_options_from_mapping(config)
    runtime = ExportRuntime.from_context(context)
    exporter = Exporter(filepath, options, runtime)
    try:
        exporter.execute()
    except GeometryError as error:
        # malformed geometry cancels the export like any other failed precondition
        raise nif_switch.SwitchError(str(error)) from error

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
        self.world_matrices = {}
        self.mesh_buffers = {}

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

        # uniform scale fix
        for root in roots:
            root.ensure_uniform_scale()
        self.world_matrices = export_skeletons.evaluate_node_worlds(self.nodes)
        for skeleton in self.skeletons:
            skeleton.world_matrices = self.world_matrices

        # copy every mesh occurrence out of Blender before anything is created
        self.extract_mesh_buffers()

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

    def extract_mesh_buffers(self):
        mesh_nodes = [node for node in self.nodes if node.role is ExportRole.MESH]

        # author state the evaluation scope changes, read while it is untouched
        metadata = {}
        for node in mesh_nodes:
            if node.source not in metadata:
                metadata[node.source] = export_geometry.capture_metadata(node.source, self.options)
        skins = {node: self.get_skin_binding(node) for node in mesh_nodes}

        with export_evaluation_scope(self.runtime, [node.source for node in mesh_nodes]) as depsgraph:
            for node in mesh_nodes:
                with evaluated_mesh(node.source, depsgraph) as (evaluated, mesh):
                    self.mesh_buffers[node] = export_geometry.extract_mesh_buffers(
                        evaluated,
                        mesh,
                        metadata[node.source],
                        skins[node],
                        name=node.source.name,
                        is_collider=node.is_collider,
                        baked_vertex_scale=node.baked_vertex_scale,
                    )

    def get_skin_binding(self, node):
        if node.is_collider:
            return None
        armature = node.source.find_armature()
        if not armature:
            return None
        skeleton = self.get_skeleton(armature, node.scope, node.name)
        return export_geometry.skin_binding(node.source, armature, skeleton)

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
            buffers = self.mesh_buffers.get(target)
            if buffers is None:
                raise RuntimeError(f"Mesh '{target.name}' was scheduled without extracted geometry")
            Mesh(target).create(buffers)
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

    def create(self, buffers):
        if len(buffers.partitions) == 1:
            outputs = [Empty(self).create(nif.NiTriShapeData())]
        else:
            # Create a trishape for each material.
            parent_output = Empty(self).create()
            outputs = [nif.NiTriShape() for _ in buffers.partitions]
            parent_output.replace_children(outputs)

        # root to skin offset
        skin_offset = None
        if buffers.skin is not None:
            placement = self.exporter.get(self.source, self.scope)
            world_matrices = self.exporter.world_matrices
            skin_offset = la.solve(world_matrices[buffers.skin.root], world_matrices[placement])

        for i, (tri, partition) in enumerate(zip(outputs, buffers.partitions)):
            tri.name = tri.name or f"Tri {self.name} {i}"
            tri.data = tri.data or nif.NiTriShapeData()
            part = export_geometry.select_partition(buffers, partition, tri.name)

            # the material decides whether vertex colors are kept, by deleting them from the shape data
            tri.data.vertex_colors = part.vertex_colors
            export_materials.export_material(
                partition.material,
                tri,
                self.source,
                self.exporter.material_context,
                is_collider=self.is_collider,
            )
            if len(tri.data.vertex_colors) == 0:
                part = replace(part, vertex_colors=tri.data.vertex_colors)

            part = export_geometry.weld(part, self.exporter.options.vertex_precision, tri.name)
            export_geometry.write_shape_data(tri.data, part)
            if skin_offset is not None:
                export_geometry.create_skin(tri, part, skin_offset)
            export_geometry.create_morphs(tri, part.morphs, self.exporter.cycle_type)

            tri.data.update_center_radius(exact=True)

        return self.output
