"""Coordinate Blender scene export and write the resulting NIF and optional keyframe files.

Collection instances have separate export records and NIF outputs. The exporter resolves skeletons and
transforms, then copies evaluated meshes into owned buffers before creating NIF objects. It creates
armatures before dependent skins and assigns switch children after all outputs exist.
"""

import collections
from dataclasses import replace
import pathlib
from typing import Any, cast

import bpy
import numpy as np
import numpy.linalg as la

from es3 import nif
from es3.utils.math import ID44, compose, decompose

from .. import nif_switch
from . import export_animation, export_geometry, export_materials, export_skeletons
from .bone_axes import export_axis_correction
from .options import ExportOptions
from .records import (
    ExportCreationJob,
    ExportNode,
    ExportNullSlot,
    ExportRole,
    ExportScope,
    ExportSwitchPlan,
)
from .runtime import ExportRuntime, evaluated_mesh, export_evaluation_scope


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
        # NIF node created for each placement; a multi-material mesh maps to the parent of its shapes
        self.outputs: dict[ExportNode, nif.NiAVObject] = {}

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
            ensure_uniform_scale(root)
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

    # RESOLVE

    def resolve_nodes(self, bl_objects, parent=None, sequence=None, scope=None):
        if scope is None:
            scope = self.root_scope
        roots = [self.new_node(obj, scope, parent) for obj in bl_objects if not obj.parent]
        included = set(bl_objects)

        queue = collections.deque(roots)
        while queue:
            node = queue.popleft()
            if not self.process(node):
                if any(nif_switch.is_switch(p.source) for p in iter_parents(node)):
                    self.errors.append(
                        f"'{node.source.name}' is inside a switch-family node but its type ({node.source.type}) cannot be exported"
                    )
                continue

            self.nodes.append(node)
            self.history[node.scope, node.source].append(node)

            # NIF meshes are leaves; Blender object children beneath them are deliberately ignored.
            if node.role is ExportRole.MESH:
                continue

            if node.role is ExportRole.EMPTY:
                self.discover_empty(node)
            elif node.role is ExportRole.ARMATURE:
                self.discover_armature(node)

            if nif_switch.is_switch(node.source):
                # slots follow the scene sequence; null-slot markers stay out of node creation
                children = nif_switch.ordered_children(node.source, sequence or bl_objects)
                slots = tuple(
                    ExportNullSlot(child) if nif_switch.is_null_slot(child) else self.new_node(child, scope, node)
                    for child in children
                    if child in included
                )
                plan = node.switch_plan = ExportSwitchPlan(node, slots)
                self.switches.append(plan)
                queue.extend(slot for slot in slots if isinstance(slot, ExportNode))
            elif hasattr(node.source, "children"):
                queue.extend(self.new_node(child, scope, node) for child in bl_objects if child.parent == node.source)

        return roots

    def new_node(self, source, scope, parent=None):
        if isinstance(source, bpy.types.PoseBone):
            matrix_local = np.asarray(source.bone.matrix_local, dtype="<f")
        else:
            matrix_local = np.asarray(source.matrix_local, dtype="<f")
        node = ExportNode(source, matrix_local, export_name(source.name, self.options), scope)
        node.reparent(parent)
        return node

    def finalize_switches(self):
        for plan in self.switches:
            node = plan.owner
            output = cast(Any, self.outputs[node])
            slots = plan.slots
            output.replace_children([self.outputs[slot] if isinstance(slot, ExportNode) else None for slot in slots])
            targets = [slot.source for slot in slots]
            if nif_switch.is_lod(node.source):
                # data.apply_scale converts these to game units later and leaves no-limit untouched
                output.active_index = 0
                output.lod_center = np.array(node.source.mw.lod_center, dtype="<f")
                ranges = nif_switch.lod_ranges(node.source, targets)
                output.lod_levels = np.array(ranges, dtype="<f").reshape(-1, 2)
            else:
                output.active_index = nif_switch.active_index(node.source, targets)

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
                        is_collider=self.is_collider(node),
                        baked_vertex_scale=node.baked_vertex_scale,
                    )

    def get_skin_binding(self, node):
        if self.is_collider(node):
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

    # PROCESS

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
            self.errors.append(
                f"bone '{node.source.name}' is marked as a {node_type}; skeletal {node_type} is not supported yet"
            )
        return True

    def discover_empty(self, node):
        if node.source.instance_type == "COLLECTION":
            collection = node.source.instance_collection
            self.resolve_nodes(list(collection.objects), parent=node, scope=ExportScope(node.scope))

        if node.source.mw.node_type not in (nif_switch.NODE, nif_switch.COLLISION):
            return  # an explicit type outranks name conventions

        if nif_switch.is_collision(node.source):
            self.colliders[node.scope, node.source].update(blender_descendants(node.source))
        elif not any(scope is node.scope for scope, _source in self.colliders) and node.name.lower().startswith("collision"):
            self.colliders[node.scope, node.source].update(blender_descendants(node.source))

    def discover_armature(self, node):
        bones = tuple(node.source.pose.bones)
        self.armature_candidates[node.scope, node.source] = bones
        self.resolve_nodes(bones, parent=node, scope=node.scope)

    # UTILITY

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

    def is_collider(self, node):
        return any(
            scope is node.scope and node.source in descendants for (scope, _root), descendants in self.colliders.items()
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

        # viewport visibility is not a filter: a hidden object can still carry a visibility animation
        if self.options.use_selection:
            allowed = set(self.runtime.context.selected_objects)
        else:
            allowed = set(base_objects)

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
        if not objects:
            if self.options.use_active_collection:
                raise Exception(
                    "No valid objects found. Ensure you have an active collection when using the 'Only Active Collection' option."
                )
            if self.options.use_selection:
                raise Exception(
                    "No valid objects found. Ensure you have at least one object selected when using the 'Only Selected' option."
                )
            raise Exception("No valid objects found. Ensure you have at least one object present in your scene.")

        return objects

    def expand_switches(self, objects):
        """Include each switch subtree, stopping at meshes, whatever its selection or visibility."""
        self.scene_objects = [ob for ob in nif_switch.scene_objects(self.runtime.scene) if not is_mesh_descendant(ob)]
        layer_objects = set(self.runtime.view_layer.objects)

        expanded = []
        placed = set()
        for ob in objects:
            if ob in placed or is_mesh_descendant(ob):
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
                    self.errors.append(f"'{other.name}' in '{ob.name}' is not in the view layer; include its collection")

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
        roots = [root for root in roots if root in self.outputs]
        if not roots:
            raise nif_switch.SwitchError("No supported objects found to export. Include a mesh, Empty, or armature.")

        file_name = self.filepath.name
        root = nif.NiNode(name=file_name, children=[self.outputs[r] for r in roots])

        # if there was only one root and it had no transforms, use it as the file root
        if len(roots) == 1:
            no_transforms = np.allclose(roots[0].matrix_local, ID44, rtol=0, atol=1e-4)
            if no_transforms or self.options.preserve_root_transforms:
                root = self.outputs[roots[0]]

        # if there are controllers but no text keys, we'll wrap in a NiBSAnimationNode
        needs_animation_node = False

        if self.options.export_animations and not self.options.extract_keyframe_data:
            for node in root.descendants(breadth_first=True):
                if node.extra_datas.find_type(nif.NiTextKeyExtraData):
                    needs_animation_node = False
                    break
                if node.controller:
                    needs_animation_node = True

        # an authored animation root keeps its own flags
        if needs_animation_node and not isinstance(root, nif.NiBSAnimationNode):
            replaced_root = None
            # if conditions are met we can update in-place rather than using a wrapper
            in_place = type(root) is nif.NiNode and root.name not in ("Bip01", "Root Bone")
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

    def setup_collision(self, data):
        # without "RCN" the engine only looks for a RootCollisionNode among the root's direct children
        colliders = list(data.objects_of_type(nif.RootCollisionNode))
        if colliders and not any(c in data.root.children for c in colliders):
            # must precede "MRK": the engine stops scanning the root's extra data at "MRK"
            data.root.extra_datas.append(nif.NiStringExtraData(string_data="RCN"))

    def setup_markers(self, data):
        markers = [ob for ob in data.objects_of_type(nif.NiTriShape) if ob.name.lower().startswith("tri editormarker")]
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

    # CREATE

    def create_job(self, job):
        target = job.target
        if job.role is ExportRole.EMPTY:
            self.create_node(target)
        elif job.role is ExportRole.MESH:
            buffers = self.mesh_buffers.get(target)
            if buffers is None:
                raise RuntimeError(f"Mesh '{target.name}' was scheduled without extracted geometry")
            self.create_mesh(target, buffers)
        elif job.role is ExportRole.ARMATURE:
            self.create_armature(target, job.owned_bones)
        else:
            raise RuntimeError(f"Unsupported export creation role: {job.role}")
        self.export_animation(target, target.name)

    def export_animation(self, node, name):
        parent_world = self.world_matrices[node.parent] if node.parent is not None else None
        export_animation.export_object_animation(
            node.source,
            name,
            self.outputs[node],
            parent_world,
            export_axis_correction(node.name),
            self.options,
            self.cycle_type,
            self.warn,
        )

    def create_node(self, node, data=None):
        """Create the NIF node of `node` and attach it to its parent's; `data` makes it a shape."""
        mw = node.source.mw
        output: Any
        if isinstance(data, nif.NiTriShapeData):
            output = nif.NiTriShape(data=data)
        elif (node.scope, node.source) in self.colliders:
            output = nif.RootCollisionNode()
        else:
            # node type names are the es3 class names; a type on a bone or non-Empty is an export error
            is_empty = getattr(node.source, "type", None) == "EMPTY"
            output = getattr(nif, mw.node_type if is_empty else nif_switch.NODE)()
        self.outputs[node] = output

        # set payload
        if isinstance(output, nif.NiFltAnimationNode):
            output.period = mw.flt_period
        elif isinstance(output, nif.NiSortAdjustNode):
            output.sorting_mode = nif.NiSortAdjustNode.SortingMode["SORTING_" + mw.sorting_mode]
            if mw.sorting_mode == "SUBSORT":
                # Subsort without an accumulator only turns sorting off
                output.sub_sorter = nif.NiAlphaAccumulator()

        # set fields
        output.name = node.name
        output.matrix = node.matrix_local

        # set parent
        if node.parent:
            self.outputs[node.parent].attach_child(output)

        # set flags: the author's, except that shadows and collision roots are never drawn
        output.flags = mw.object_flags
        if output.is_shadow or isinstance(output, nif.RootCollisionNode):
            output.app_culled = True

        # set bounds
        if is_bounding_box(node):
            placement = self.get(node.source, node.scope)
            t, r, s = decompose(self.world_matrices[placement])
            s *= node.source.empty_display_size
            output.bounding_volume = nif.NiBoxBV(center=t, axes=r, extents=s)
            output.matrix = ID44

        return output

    def create_armature(self, node, bone_nodes):
        # create root
        self.create_node(node)
        # create bones
        for bone in bone_nodes:
            self.create_node(bone).name = bone_name(bone.name)
            self.export_animation(bone, bone.name)

    def create_mesh(self, node, buffers):
        if len(buffers.partitions) == 1:
            outputs = [cast(nif.NiTriShape, self.create_node(node, nif.NiTriShapeData()))]
        else:
            # Create a trishape for each material.
            parent_output = self.create_node(node)
            outputs = [nif.NiTriShape() for _ in buffers.partitions]
            parent_output.replace_children(outputs)

        # root to skin offset
        skin_offset = None
        if buffers.skin is not None:
            placement = self.get(node.source, node.scope)
            skin_offset = la.solve(self.world_matrices[buffers.skin.root], self.world_matrices[placement])

        is_collider = self.is_collider(node)
        for i, (tri, partition) in enumerate(zip(outputs, buffers.partitions)):
            tri.name = tri.name or f"Tri {node.name} {i}"
            tri.data = tri.data or nif.NiTriShapeData()
            part = export_geometry.select_partition(buffers, partition, tri.name)

            # the material decides whether vertex colors are kept, by deleting them from the shape data
            tri.data.vertex_colors = part.vertex_colors
            export_materials.export_material(
                partition.material,
                tri,
                node.source,
                self.material_context,
                is_collider=is_collider,
            )
            if len(tri.data.vertex_colors) == 0:
                part = replace(part, vertex_colors=tri.data.vertex_colors)

            part = export_geometry.weld(part, self.options.vertex_precision, tri.name)
            export_geometry.write_shape_data(tri.data, part)
            if skin_offset is not None:
                export_geometry.create_skin(tri, part, skin_offset, self.outputs)
            export_geometry.create_morphs(tri, part.morphs, self.cycle_type)

            tri.data.update_center_radius(exact=True)

        return self.outputs[node]


def iter_parents(node):
    parent = node.parent
    while parent is not None:
        yield parent
        parent = parent.parent


def is_mesh_descendant(source):
    return any(parent.type == "MESH" for parent in iter_parents(source))


def blender_descendants(source):
    queue = collections.deque(source.children)
    while queue:
        child = queue.popleft()
        yield child
        queue.extend(child.children)


def export_name(name, options):
    """The NIF name of a Blender source, with Blender's numeric suffix removed when requested or required."""
    base_name = strip_blender_suffix(name)

    # Always strip if the export option is active.
    if options.strip_numeric_suffixes:
        return base_name

    # The engine requires exact matches for these.
    if base_name in ("Bip01", "Bounding Box", "PickProxy", "Root Bone"):
        return base_name

    return name


def strip_blender_suffix(name):
    """Remove Blender's automatic numeric suffixes like .001, .002, etc."""
    base_name, *suffix = name.rsplit(".", 1)
    if suffix and suffix[0].isnumeric():
        return base_name
    return name


def bone_name(name):
    """The NIF name of a Blender bone: .L/.R suffixes become Bip01 sides."""
    if name.startswith("Bip01"):
        if name.endswith(".L"):
            return f"Bip01 L {name[6:-2]}"
        if name.endswith(".R"):
            return f"Bip01 R {name[6:-2]}"
    return name


def is_bounding_box(node):
    return node.name.lower().startswith("bounding box")


def ensure_uniform_scale(node):
    """Move nonuniform scale out of `node`'s transform: into its baked vertices and its children's transforms."""
    if is_bounding_box(node):
        return

    linear = node.matrix_local[:3, :3]
    axis_lengths = la.norm(linear, axis=0)
    if np.any(axis_lengths < 1e-8):
        raise nif_switch.SwitchError(f"'{node.name}' has a singular transform; zero scale cannot be exported")
    normalized = linear / axis_lengths
    if not np.allclose(normalized.T @ normalized, np.eye(3), rtol=0, atol=1e-4):
        raise nif_switch.SwitchError(f"'{node.name}' has shear, which cannot be represented by a NIF transform")

    t, r, s = decompose(node.matrix_local)

    if not np.allclose(s[:1], s[1:], rtol=0, atol=0.001):
        print(f"[INFO] {node.source.name} has non-uniform scale")

        if isinstance(node.source, bpy.types.PoseBone):
            raise nif_switch.SwitchError(
                f"Bone '{node.name}' has non-uniform scale, which cannot be represented without changing its bind pose"
            )
        if node.source.type == "MESH":
            node.baked_vertex_scale = s

        node.matrix_local = compose(t, r, 1)
        scale_matrix = ID44.copy()
        scale_matrix[:3, :3] = np.diag(s)
        for child in node.children:
            child.matrix_local = scale_matrix @ child.matrix_local

    for child in node.children:
        ensure_uniform_scale(child)
