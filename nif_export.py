import bpy

import timeit
import pathlib
import itertools
import collections

import numpy as np
import numpy.linalg as la

from es3 import nif
from es3.utils import meshoptimizer
from es3.utils.math import ID44, compose, decompose

from functools import cached_property

from . import nif_utils
from . import nif_shader

from bpy_extras.io_utils import axis_conversion  # type: ignore

biped_axis_correction = np.array(axis_conversion('Y', 'Z', '-X', 'Z').to_4x4(), dtype="<f")
biped_axis_correction_inverse = la.inv(biped_axis_correction)

other_axis_correction = np.array(axis_conversion('-Z', '-Y', 'Y', 'Z').to_4x4(), dtype="<f")
other_axis_correction_inverse = la.inv(other_axis_correction)


def save(context, filepath, **config):
    """save the scene to a nif file"""

    print(f"Export File: {filepath}")
    time = timeit.default_timer()

    exporter = Exporter(filepath, config)
    exporter.execute()

    time = timeit.default_timer() - time
    print(f"Export Done: {time:.4f} seconds")

    return {"FINISHED"}


class Exporter:
    vertex_precision = 0.001
    extract_keyframe_data = False
    export_animations = True
    preserve_root_tranforms = False
    preserve_material_names = True

    def __init__(self, filepath, config):
        vars(self).update(config)
        self.filepath = pathlib.Path(filepath)
        self.nodes = {}
        self.materials = {}
        self.pixel_datas = {}
        self.history = collections.defaultdict(set)
        self.armatures = collections.defaultdict(set)
        self.colliders = collections.defaultdict(set)
        self.depsgraph = None

    def execute(self):
        bl_objects = self.get_source_objects()

        # resolve heirarchy
        roots = self.resolve_nodes(bl_objects)

        # resolve armatures
        if any(self.armatures):
            self.resolve_armatures()
            self.apply_axis_corrections()

        # resolve depsgraph
        self.resolve_depsgraph()

        # uniform scale fix
        for root in roots:
            root.ensure_uniform_scale()

        # create ni objects
        for node, cls in self.nodes.items():
            if node.output is None:
                cls(node).create()
                node.animation.create()

        data = nif.NiStream()
        data.root = self.get_root_output(roots)
        data.apply_scale(self.scale_correction)
        data.merge_properties(ignore={"name", "shine", "specular_color"})
        data.sort()
        self.setup_markers(data)
        data.save(self.filepath)

        # extract x/kf file
        if self.extract_keyframe_data:
            self.export_keyframe_data(data)

    # -------
    # RESOLVE
    # -------

    def resolve_nodes(self, bl_objects, parent=None):
        roots = [SceneNode(self, obj, parent) for obj in bl_objects if not obj.parent]

        queue = collections.deque(roots)
        while queue:
            node = queue.popleft()
            if self.process(node):
                self.history[node.source].add(node)
                if hasattr(node.source, "children"):
                    queue.extend(
                        SceneNode(self, child, node)
                        for child in node.source.children
                        if child in bl_objects
                    )

        return roots

    def resolve_armatures(self):
        """ TODO
            support multiple armatures
        """
        root, = self.armatures
        root_node = self.get(root)
        root_matrix = root_node.matrix_world

        # convert from armature space to local space
        for node in self.iter_bones(root_node):
            node.matrix_world = root_matrix @ node.matrix_local

        # update parent of those using bone parenting
        for node in list(root_node.children):
            if type(node.source) is bpy.types.Object:
                if node.source.parent_type == "BONE":
                    pose_bone = root.pose.bones[node.source.parent_bone]
                    node.parent = self.get(pose_bone)
                    node.matrix_world = np.array(node.source.matrix_world)

    def apply_axis_corrections(self):
        """ TODO
            support multiple armatures
            only apply on 'Bip01' nodes
        """
        root_node = self.get(*self.armatures)
        root_matrix = root_node.matrix_world
        root_inverse = la.inv(root_matrix)

        for node in self.iter_bones(root_node):
            prev_matrix_local = node.matrix_local.copy()

            node.matrix_bind = root_inverse @ node.matrix_world @ node.axis_correction
            node.matrix_world = root_matrix @ node.source.matrix @ node.axis_correction

            inverse_transform = la.solve(node.matrix_local, prev_matrix_local)
            for child in node.children:
                child.matrix_local = inverse_transform @ child.matrix_local

    def resolve_depsgraph(self):
        temp_modifiers = []
        temp_hide_view = []
        temp_mute_keys = []
        try:
            meshes = [node.source for node, cls in self.nodes.items() if cls is Mesh]
            for source in meshes:
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
                        if k.mute != True:
                            k.mute = True
                            temp_mute_keys.append(k)

            # get evaluated dependency graph
            self.depsgraph = bpy.context.evaluated_depsgraph_get()
        finally:
            for m in temp_hide_view:
                m.show_viewport = True
            for s, m in temp_modifiers:
                s.modifiers.remove(m)
            for k in temp_mute_keys:
                k.mute = False

    def export_keyframe_data(self, data):
        nif_path = self.filepath
        xnif_path = nif_path.with_name("x" + nif_path.name)
        xkf_path = xnif_path.with_suffix(".kf")
        data.extract_keyframe_data().save(xkf_path)
        data.save(xnif_path)

    # -------
    # PROCESS
    # -------

    @nif_utils.dispatcher
    def process(self, node):
        print(f"Warning: Unsupported Type ({node.source.type})")
        return False

    @process.register("EMPTY")
    def process_empty(self, node):
        self.nodes[node] = Empty

        if node.source.instance_type == 'COLLECTION':
            collection = node.source.instance_collection
            self.resolve_nodes(set(collection.objects), parent=node)

        if (not self.colliders) and node.name.lower().startswith("collision"):
            self.colliders[node.source].update(nif.NiAVObject.descendants(node.source))

        return True

    @process.register("MESH")
    def process_mesh(self, node):
        self.nodes[node] = Mesh
        return True

    @process.register("ARMATURE")
    def process_armature(self, node):
        self.nodes[node] = Armature
        self.armatures[node.source].update(node.source.pose.bones)
        self.resolve_nodes(self.armatures[node.source], parent=node)
        return True

    @process.register("NONE")
    def process_bone(self, node):
        self.nodes[node] = Empty
        return True

    # -------
    # UTILITY
    # -------

    def get(self, source):
        return next(iter(self.history[source]))

    def get_source_objects(self):
        if self.use_active_collection:
            objects = set(bpy.context.view_layer.active_layer_collection.collection.all_objects)
        else:
            objects = set(bpy.context.view_layer.objects)

        if self.use_selection:
            objects &= set(bpy.context.selected_objects)
        else:
            objects &= set(bpy.context.visible_objects)

        # ensure all objects have a valid path to the scene root
        for ob in objects.copy():
            while (ob.parent is not None) and (ob.parent not in objects):
                objects.add(ob.parent)
                ob = ob.parent

        # show a useful error message if no valid objects were discovered
        if len(objects) == 0:
            if self.use_active_collection:
               raise Exception("No valid objects found. Ensure you have an active collection when using the 'Only Active Collection' option.")
            elif self.use_selection:
               raise Exception("No valid objects found. Ensure you have at least one object selected when using the 'Only Selected' option.")
            else:
               raise Exception("No valid objects found. Ensure you have at least one object present in your scene.")

        return objects

    def get_root_output(self, roots):
        root = nif.NiNode(name=self.filepath.name, children=[r.output for r in roots])

        if self.export_animations and not self.extract_keyframe_data:
            # convert to NiBSAnimationNode if controllers are present without text keys
            for obj in root.descendants():
                if obj.extra_datas.find_type(nif.NiTextKeyExtraData):
                    break
                if obj.controller:
                    root = nif.NiBSAnimationNode(name=root.name, children=root.children, animated=True)
                    break

        if type(root) is nif.NiNode and len(roots) == 1:
            # if there's only one root and it has no transforms, use it as the file root
            no_transforms = np.allclose(roots[0].matrix_local, ID44, rtol=0, atol=1e-4)
            if no_transforms or self.preserve_root_tranforms:
                root = roots[0].output

        return root

    def iter_bones(self, root):
        """yields bone nodes in hierarchical order"""
        edit_bones = root.source.data.bones
        pose_bones = root.source.pose.bones
        for bone in edit_bones:
            yield self.get(pose_bones[bone.name])

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
    def scale_correction(self):
        addon = bpy.context.preferences.addons[__package__]
        return 1 / addon.preferences.scale_correction

    @property
    def cycle_type(self):
        return "CLAMP" if self.extract_keyframe_data else "CYCLE"


class SceneNode:
    def __init__(self, exporter, source, parent=None):
        self.exporter = exporter
        #
        self.source = source
        self.output = None
        #
        self.parent = parent
        self.children = list()
        #
        if isinstance(source, bpy.types.PoseBone):
            self.matrix_local = np.asarray(source.bone.matrix_local, dtype="<f")
        else:
            self.matrix_local = np.asarray(source.matrix_local, dtype="<f")

    def __repr__(self):
        if not self.parent:
            return f'SceneNode("{self.name}", parent=None)'
        return f'SceneNode("{self.name}", parent="{self.parent.name}")'

    def create(self, *args, **kwargs):
        raise NotImplementedError

    @cached_property
    def name(self):
        # discard numeric suffixes generated by blender
        name, *suffix = self.source.name.rsplit('.', 1)
        if suffix and suffix[0].isnumeric():
            if name in ("Bip01", "Root Bone"):
                return name
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
        try:  # remove from old children list
            self._parent.children.remove(self)
        except (AttributeError, ValueError):
            pass
        self._parent = node
        try:  # append onto new children list
            self._parent.children.append(self)
        except (AttributeError, ValueError):
            pass

    @property
    def parents(self):
        node = self.parent
        while node:
            yield node
            node = node.parent

    @property
    def properties(self):
        raise NotImplementedError

    @property
    def matrix_world(self):
        if self.parent:
            return self.parent.matrix_world @ self.matrix_local
        return self.matrix_local

    @matrix_world.setter
    def matrix_world(self, matrix):
        if self.parent:
            matrix = la.solve(self.parent.matrix_world, matrix)
        self.matrix_local = matrix

    @property
    def axis_correction(self):
        if "Bip01" in self.name:
            return biped_axis_correction
        return other_axis_correction

    @property
    def axis_correction_inverse(self):
        if "Bip01" in self.name:
            return biped_axis_correction_inverse
        return other_axis_correction_inverse

    @property
    def is_bounding_box(self):
        return self.name.lower().startswith("bounding box")

    @property
    def is_collider(self):
        return any(self.source in s for s in self.exporter.colliders.values())

    def ensure_uniform_scale(self):
        if self.is_bounding_box:
            return

        l, r, s = decompose(self.matrix_local)

        if not np.allclose(s[:1], s[1:], rtol=0, atol=0.001):
            print(f"[INFO] {self.source.name} has non-uniform scale")

            try:
                if self.source.type == "MESH":
                    self.vertex_transform = s
            except AttributeError:
                pass  # pose bone

            self.matrix_local = compose(l, r, 1)
            for child in self.children:
                child.matrix_local[:3, :3] *= s

        for child in self.children:
            child.ensure_uniform_scale()

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

    def create(self, data=None):
        if isinstance(data, nif.NiTriShapeData):
            self.output = nif.NiTriShape(data=data)
        else:
            if self.source in self.exporter.colliders:
                self.output = nif.RootCollisionNode(app_culled=True)
            else:
                self.output = nif.NiNode()

        # set fields
        self.output.name = self.name
        self.output.matrix = self.matrix_local

        # set parent
        if self.parent:
            self.parent.output.children.append(self.output)

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
        l, r, s = decompose(self.matrix_world)
        s *= self.source.empty_display_size
        self.output.bounding_volume = nif.NiBoxBV(center=l, axes=r, extents=s)
        self.output.matrix = ID44


class Armature(SceneNode):
    __slots__ = ()

    def __init__(self, node):
        self.__dict__ = node.__dict__

    def create(self):
        # create root
        Empty(self).create()
        # create bones
        for node in self.exporter.iter_bones(self):
            Empty(node).create()
            node.output.name = node.bone_name
            node.animation.create()


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

        if len(material_data) == 1:
            outputs = [Empty(self).create(ni_data)]
        else:
            # Create a trishape for each material.
            outputs = Empty(self).create().children = [nif.NiTriShape() for _ in material_data]

        for i, tri in enumerate(outputs):
            tri.name = tri.name or f"Tri {self.name} {i}"
            tri.data = tri.data or nif.NiTriShapeData()
            tri_skin = skin_data.copy()
            tri_morphs = morph_data.copy()

            material, indices = material_data[i]
            self.material.create(material, tri)

            self.extract_vertex_data(indices, ni_data, tri.data, skin_data, tri_skin, morph_data, tri_morphs)
            self.optimize_geometry(tri.data, tri_skin, tri_morphs)
            self.create_vertex_groups(tri, tri_skin)
            self.create_vertex_morphs(tri, tri_morphs)

            tri.data.update_center_radius()

        return self.output

    def create_vertices(self, ni, bl):
        ni.vertices.resize(len(bl.vertices), 3)
        bl.vertices.foreach_get("co", ni.vertices.ravel())

        # Ensure vertex transforms.
        # TODO refactor this!
        if hasattr(self, "vertex_transform"):
            ni.vertices *= self.vertex_transform

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
        if len(skin_data.weights) == 0:
            return

        skin = trishape.skin = nif.NiSkinInstance()
        skin.root, *skin.bones = [node.output for node in (skin_data.root, *skin_data.bones)]

        skin.data = nif.NiSkinData()
        skin.data.bone_data = [nif.NiSkinDataBoneData() for _ in skin_data.bones]

        # root to skin offset
        offset = la.solve(skin_data.root.matrix_world, self.matrix_world)
        skin.data.matrix = la.inv(offset)

        # calculate bone data
        for i, bone_data in enumerate(skin.data.bone_data):
            j, = np.nonzero(skin_data.weights[i])

            # skip empty weights
            if len(j) == 0:
                skin.bones[i] = skin.data.bone_data[i] = None
                continue

            # set vertex weights
            bone_data.vertex_weights.resize(len(j))
            bone_data.vertex_weights["f0"] = j
            bone_data.vertex_weights["f1"] = skin_data.weights[i, j]

            # skin to bone offset
            bone_data.matrix = la.solve(skin_data.bones[i].matrix_bind, offset)

            # apply center/radius
            bone_data.update_center_radius(trishape.data.vertices[j])

        # filter unused bones
        skin.bones = [*filter(None, skin.bones)]
        skin.data.bone_data = [*filter(None, skin.data.bone_data)]

    def create_vertex_morphs(self, trishape, morph_data):
        if len(morph_data.targets) == 0:
            return

        # create controller
        controller = nif.NiGeomMorpherController(
            cycle_type=self.exporter.cycle_type,
            target=trishape,
        )
        controller.stop_time = max(kf[-1, 0] for kf in filter(len, morph_data.keys))

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
        # Can't evaluate if hidden.
        self.source.hide_viewport = False

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
        skin_data = nif_utils.Namespace(root=None, bones=(), weights=())

        # ignore collider meshes
        if self.is_collider:
            return skin_data

        # find the armature root
        armature = self.source.find_armature()
        if not armature:
            return skin_data

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
            return skin_data

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

        # get all armature nodes
        skin_data.root = self.exporter.get(armature)
        skin_data.bones = [self.exporter.get(bone) for bone in bones.values()]
        skin_data.weights = weights

        return skin_data

    def get_morph_data(self, bl_data, ni_data):
        morph_data = nif_utils.Namespace(targets=(), keys=(), interpolations=())

        # respect export setting
        if not self.exporter.export_animations:
            return morph_data

        # ignore collider meshes
        if self.is_collider:
            return morph_data

        # collect shape key data
        try:
            shape_keys = self.source.data.shape_keys
            action = shape_keys.animation_data.action
            basis = shape_keys.key_blocks[0]
        except (AttributeError, IndexError):
            return morph_data

        animation = self.animation
        data_paths = {fc.data_path: fc for fc in action.fcurves}

        # isolate unmuted layers
        fcurves_dict = {None: basis}
        for sk in shape_keys.key_blocks:
            fc = data_paths.get(sk.path_from_id('value'))
            if fc and not (fc.mute or sk.mute):
                fcurves_dict[fc] = sk
                assert sk.relative_key == basis, "Shape keys must be relative to basis."

        # bail if all were muted
        if not len(fcurves_dict) > 1:
            return morph_data

        # allocate keyframe data
        morph_data.keys = [None] * len(fcurves_dict)
        morph_data.interpolations = [None] * len(fcurves_dict)

        # allocate morph targets
        morph_data.targets = np.empty((len(fcurves_dict), len(bl_data.vertices), 3))

        for i, (fc, sk) in enumerate(fcurves_dict.items()):
            if i == 0:
                # is basis layer
                key_type = nif.NiFloatData.KeyType.LIN_KEY
                keys = np.empty(0)
            else:
                key_type = animation.get_interpolation_type([fc])
                keys = animation.collect_keyframe_points([fc], key_type)

            # populate keyframe data
            morph_data.keys[i] = keys
            morph_data.interpolations[i] = key_type

            # populate morph targets
            sk.data.foreach_get("co", morph_data.targets[i].ravel())

        # make relative to basis
        morph_data.targets[1:] -= morph_data.targets[0]

        # view as per-face-loops
        morph_data.targets = morph_data.targets[:, ni_data.triangles].reshape(-1, len(ni_data.vertices), 3)

        return morph_data

    def get_material_data(self, bl_object, bl_data):
        # get each polygon's material index
        material_indices = np.empty(len(bl_data.polygons), dtype=int)
        bl_data.polygons.foreach_get("material_index", material_indices)

        # reduce to only the active indices
        active_material_indices = np.unique(material_indices)

        # simple case for a single material
        if len(active_material_indices) <= 1:
            return [(bl_object.active_material, ...)]

        # get each polygon's vertex indices
        loops = np.arange(len(bl_data.loops), dtype=int).reshape(-1, 3)

        material_data = []
        for i in active_material_indices:
            material = bl_object.material_slots[i].material
            indices = loops[material_indices == i].ravel()
            if len(indices):
                material_data.append((material, indices))

        return material_data

    # -- optimizer functions --

    def optimize_geometry(self, data, skin_data, morph_data):
        if len(data.vertices) == 0:
            return

        # clear duplicates
        indices, inverse = nif_utils.unique_rows(
            data.vertices,
            data.normals,
            data.vertex_colors,
            *data.uv_sets,
            # *skin_data.weights,
            # *morph_data.targets,
            precision=self.exporter.vertex_precision
        )

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
        if len(src_data.vertices):
            dst_data.vertices = src_data.vertices[indices]
        if len(src_data.normals):
            dst_data.normals = src_data.normals[indices]
        if len(src_data.vertex_colors):
            dst_data.vertex_colors = src_data.vertex_colors[indices]
        if len(src_data.uv_sets):
            dst_data.uv_sets = src_data.uv_sets[:, indices]
        if len(src_skin.weights):
            dst_skin.weights = src_skin.weights[:, indices]
        if len(src_morphs.targets):
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
        if len(skin.weights):
            skin.weights[:, vertex_remap] = skin.weights.copy()
        if len(morphs.targets):
            morphs.targets[:, vertex_remap] = morphs.targets.copy()


class Material(SceneNode):
    __slots__ = ()

    def __init__(self, node):
        self.__dict__ = node.__dict__

    def create(self, material, ni_object):
        if self.is_collider:
            return  # no properties on colliders
        if not (material and material.use_nodes):
            return  # not an applicable material

        try:
            bl_prop = material.mw.validate()
        except TypeError:
            bl_prop = None

        try:
            if not bl_prop.use_vertex_colors:
                del ni_object.data.vertex_colors
        except AttributeError:
            pass

        # Re-use duplicate materials
        if material in self.exporter.materials:
            ni_object.properties = self.exporter.materials[material]
            # Blender keeps UV animations on the material, but NIF keeps them on the object.
            # Thus copying a material to the new object doesn't copy over any UV animations.
            # Go through and manually copy them instead.
            if bl_prop is not None:
                for slot in bl_prop.texture_slots:
                    self.animation.create_uv_controller(bl_prop, slot)
            return

        if bl_prop is None:
            # Not a MW Material
            self.create_fallback_property(ni_object, material)
        else:
            # Material Property
            self.create_material_property(ni_object, bl_prop)
            # Alpha Property
            self.create_alpha_property(ni_object, bl_prop)
            # Texture Property
            self.create_texturing_property(ni_object, bl_prop)
            # Wireframe Property
            self.create_wireframe_property(ni_object, bl_prop)
            # Stencil Property
            self.create_stencil_property(ni_object, bl_prop)

        # Update Material Cache
        self.exporter.materials[material] = ni_object.properties

    def create_material_property(self, ni_object, bl_prop):
        ni_prop = nif.NiMaterialProperty()
        # Material Name
        ni_prop.name = bl_prop.material.name
        # Material Color
        ni_prop.ambient_color[:] = bl_prop.ambient_color[:3]
        ni_prop.diffuse_color[:] = bl_prop.diffuse_color[:3]
        ni_prop.specular_color[:] = bl_prop.specular_color[:3]
        ni_prop.emissive_color[:] = bl_prop.emissive_color[:3]
        # Material Shine
        ni_prop.shine = bl_prop.shine
        # Material Alpha
        ni_prop.alpha = bl_prop.alpha
        # Material Anims
        self.animation.create_material_controllers(ni_prop, bl_prop)
        # Apply Property
        ni_object.properties.append(ni_prop)

    def create_alpha_property(self, ni_object, bl_prop):
        if not (bl_prop.use_alpha_blend or bl_prop.use_alpha_clip):
            return
        ni_prop = nif.NiAlphaProperty()
        # Alpha Blending
        ni_prop.alpha_blending = bl_prop.use_alpha_blend
        # Src Blend Mode
        ni_prop.src_blend_mode = 'SRC_ALPHA'
        # Dst Blend Mode
        ni_prop.dst_blend_mode = 'INV_SRC_ALPHA'
        # Alpha Testing
        ni_prop.alpha_testing = bl_prop.use_alpha_clip
        # Test Threshold
        ni_prop.test_ref = int(bl_prop.material.alpha_threshold * 255)
        # Test Method
        ni_prop.test_mode = 'GREATER'
        # Apply Property
        ni_object.properties.append(ni_prop)

    def create_texturing_property(self, ni_object, bl_prop):
        if not any(slot.image for slot in bl_prop.texture_slots):
            return
        ni_prop = nif.NiTexturingProperty()
        # Texture Slots
        for name in nif.NiTexturingProperty.texture_keys:
            self.create_texturing_property_map(bl_prop, ni_prop, name)
        # Apply Property
        ni_object.properties.append(ni_prop)

    def create_texturing_property_map(self, bl_prop, ni_prop, name):
        try:
            bl_slot = getattr(bl_prop, name)
            filepath = bl_slot.image.filepath
        except AttributeError:
            return

        ni_slot = nif.NiTexturingPropertyMap()

        # material name
        if not self.exporter.preserve_material_names:
            if (name == "base_texture") and filepath:
                prop = self.output.get_property(nif.NiMaterialProperty)
                if prop is not None:
                    prop.name = pathlib.Path(filepath).stem

        # texture index
        for i, uv in enumerate(self.source.data.uv_layers):
            if uv.name == bl_slot.layer:
                ni_slot.uv_set = i
                break

        # use repeat
        if not bl_slot.use_repeat:
            ni_slot.clamp_mode = 0  # CLAMP_S_CLAMP_T

        # use mipmaps
        if not bl_slot.use_mipmaps:
            bl_slot.use_mipmaps = 0  # NO

        # source image
        ni_slot.source = nif.NiSourceTexture()

        if (bl_slot.image.source == "GENERATED") or bl_slot.image.packed_file:
            include_alpha = bl_prop.use_alpha_blend or bl_prop.use_alpha_clip
            ni_slot.source.pixel_data = self.create_pixel_data(bl_slot.image, include_alpha)
        else:
            ni_slot.source.filename = bpy.path.abspath(filepath)

        setattr(ni_prop, name, ni_slot)

        # uv animations
        self.animation.create_uv_controller(bl_prop, bl_slot)

    def create_wireframe_property(self, ni_object, bl_prop):
        if self.source.display_type == "WIRE":
            ni_prop = nif.NiWireframeProperty(wireframe=True)
            ni_object.properties.append(ni_prop)

    def create_stencil_property(self, ni_object, bl_prop):
        if not bl_prop.material.use_backface_culling:
            m = nif.NiStencilProperty.DrawMode.DRAW_BOTH
            ni_prop = nif.NiStencilProperty(draw_mode=m)
            ni_object.properties.append(ni_prop)

    def create_pixel_data(self, image, include_alpha) -> nif.NiPixelData:
        if pixel_data := self.exporter.pixel_datas.get(image):
            assert pixel_data.pixel_format.has_alpha == include_alpha
        else:
            assert image.channels == 4

            width, height = image.size
            for i in (width, height):
                if (i == 0) or (i & (i - 1) != 0):
                    raise ValueError("Image dimensions must be a power of 2")

            # TODO: dxt compression (does not work in OpenMW yet)
            if include_alpha:
                pixel_format = nif.NiPixelFormat.RGBA
                pixel_stride = 4
            else:
                pixel_format = nif.NiPixelFormat.RGB
                pixel_stride = 3

            pixel_data = nif.NiPixelData(pixel_format=pixel_format, pixel_stride=pixel_stride)

            # TODO: generated mipmaps (image.resize() might work)
            pixel_data.mipmap_levels.resize((1, 3))
            pixel_data.mipmap_levels[:] = [width, height, 0]

            # generated images seem to always have alpha channels
            pixel_floats = np.empty((width, height, 4), np.float32)
            image.pixels.foreach_get(pixel_floats.ravel())
            if pixel_stride == 3:
                pixel_floats = pixel_floats[:, :, :3]

            # convert pixels from opengl layout to directx layout
            pixel_floats = np.flipud(pixel_floats)

            # convert pixels from floats into flat array of bytes
            pixel_data.pixel_data = (pixel_floats.ravel() * 255.0).round().astype(np.ubyte)

            self.exporter.pixel_datas[image] = pixel_data

        return pixel_data

    @staticmethod
    def create_fallback_property(ni_object, material):
        bl_image = nif_shader.get_base_texture_image(material)
        if bl_image:
            # Material Property
            ni_prop = nif.NiMaterialProperty()
            ni_prop.ambient_color[:] = 1.0
            ni_prop.diffuse_color[:] = 1.0
            ni_prop.alpha = 1.0
            ni_object.properties.append(ni_prop)
            # Texturing Property
            ni_prop = nif.NiTexturingProperty()
            ni_object.properties.append(ni_prop)
            # Base Texture Map
            m = ni_prop.base_texture = nif.NiTexturingPropertyMap()
            # Base Texture Source
            m.source = nif.NiSourceTexture()
            m.source.filename = bpy.path.abspath(bl_image.filepath)


class Animation(SceneNode):
    __slots__ = ()

    def __init__(self, node):
        self.__dict__ = node.__dict__

    def create(self):
        if not self.exporter.export_animations:
            return

        # create text keys even if no animations are assigned
        # this is necessary as text keys are specified on the
        # file root and will influence all descending objects
        self.create_text_keys()

        fcurves_dict = self.get_fcurves_dict(self.source.id_data)
        if len(fcurves_dict) == 0:
            return

        self.create_vis_controller(fcurves_dict)
        self.create_translations(fcurves_dict)
        self.create_rotations(fcurves_dict)
        self.create_scales(fcurves_dict)

    def create_text_keys(self):
        try:
            markers = self.source.animation_data.action.pose_markers
        except AttributeError:
            return False

        if len(markers) == 0:
            return False

        text_data = nif.NiTextKeyExtraData()
        text_data.keys.resize(len(markers))

        for i, marker in enumerate(markers):
            time = marker.frame / bpy.context.scene.render.fps
            name = marker.name.replace("; ", "\r\n")
            text_data.keys[i] = time, name

        text_data.collapse_groups()
        text_data.keys.sort()

        self.output.extra_data = text_data

        return True

    def create_vis_controller(self, fcurves_dict):
        fcurves = self.collect_fcurves(fcurves_dict, "hide_viewport", num_axes=1)
        if len(fcurves) == 0:
            return False

        key_type = nif.NiFloatData.KeyType.LIN_KEY
        keys = self.collect_keyframe_points(fcurves, key_type)
        if len(keys) == 0:
            return False

        controller = nif.NiVisController(
            cycle_type=self.exporter.cycle_type,
            target=self.output,
            data=nif.NiVisData(),
        )
        controller.data.keys.resize(len(keys))
        controller.data.times = keys[:, 0]
        controller.data.values = 1 - keys[:, 1]

        # update controller times
        controller.update_start_stop_times()

        self.output.controllers.appendleft(controller)

        return True

    def create_translations(self, fcurves_dict):
        fcurves = self.collect_fcurves(fcurves_dict, "location", num_axes=3)
        if len(fcurves) == 0:
            return False

        key_type = self.get_interpolation_type(fcurves)
        if key_type is None:
            return False

        # collect keyframe points
        keys = self.collect_keyframe_points(fcurves, key_type)
        if len(keys) == 0:
            return False

        # get keyframe controller
        controller = self.create_keyframe_controller()
        controller.data.translations.key_type = key_type

        # set the controller keys
        controller.data.translations.keys = keys

        # update start/stop times
        controller.start_time = min(keys[0, 0], controller.start_time)
        controller.stop_time = max(keys[-1, 0], controller.stop_time)

        # convert to local space
        offset = self.to_local_space()
        if not np.allclose(offset, ID44, rtol=0, atol=1e-4):
            t = controller.data.translations
            t.values[:] = t.values @ offset[:3, :3].T + offset[:3, 3]
            if key_type.name == "BEZ_KEY":
                t.in_tans[:] = t.in_tans @ offset[:3, :3].T
                t.out_tans[:] = t.out_tans @ offset[:3, :3].T

        return True

    def create_rotations(self, fcurves_dict):
        has_euler = self.create_euler_rotations(fcurves_dict)
        has_quats = self.create_quaternion_rotations(fcurves_dict)
        if has_euler and has_quats:
            raise ValueError(f"'({self.name})' mixing euler and quaternion rotations in the same action is not supported")
        return has_euler or has_quats

    def create_euler_rotations(self, fcurves_dict):
        fcurves = self.collect_fcurves(fcurves_dict, "rotation_euler", num_axes=3)
        if len(fcurves) == 0:
            return False

        key_type = self.get_interpolation_type(fcurves)
        if key_type is None:
            return False

        if isinstance(self.source, bpy.types.PoseBone):
            raise ValueError(f'({self.name}) bones euler animations are not currently supported.')

        # get keyframe controller
        controller = self.create_keyframe_controller()

        # prepare euler rotations
        rotations = controller.data.rotations
        rotations.key_type = rotations.KeyType.EULER_KEY
        rotations.euler_data = tuple(nif.NiFloatData(key_type=key_type) for _ in range(3))

        for i, euler_data in enumerate(rotations.euler_data):
            axis_fcurves = [fc for fc in fcurves if fc.array_index == i]
            if len(axis_fcurves) == 0:
                continue

            # collect keyframe points
            keys = self.collect_keyframe_points(axis_fcurves, key_type)
            if len(keys) == 0:
                continue

            # set the controller keys
            euler_data.keys = keys

            # update start/stop times
            controller.start_time = min(keys[0, 0], controller.start_time)
            controller.stop_time = max(keys[-1, 0], controller.stop_time)

        # convert to local space
        offset = self.to_local_space()
        if not np.allclose(offset, ID44, rtol=0, atol=1e-4):
            rotations.convert_to_quaternions()
            values = controller.data.rotations.values
            offset = nif_utils.quaternion_from_matrix(offset)
            nif_utils.quaternion_mul(offset, values, out=values)
            values /= la.norm(values, axis=1)[:, None]

        return True

    def create_quaternion_rotations(self, fcurves_dict):
        fcurves = self.collect_fcurves(fcurves_dict, "rotation_quaternion", num_axes=4)
        if len(fcurves) == 0:
            return False

        # TODO: TCB interpolation
        key_type = nif.NiRotData.KeyType.LIN_KEY

        # collect keyframe points
        keys = self.collect_keyframe_points(fcurves, key_type)
        if len(keys) == 0:
            return False

        # get keyframe controller
        controller = self.create_keyframe_controller()

        # split times from values
        values = keys[:, 1:]

        # apply axis corrections
        if isinstance(self.source, bpy.types.PoseBone):
            offset = nif_utils.quaternion_from_matrix(self.axis_correction)
            nif_utils.quaternion_mul(values, offset, out=values)

        # convert to local space
        offset = self.to_local_space()
        if not np.allclose(offset, ID44, rtol=0, atol=1e-4):
            offset = nif_utils.quaternion_from_matrix(offset)
            nif_utils.quaternion_mul(offset, values, out=values)

        # normalize rotation keys
        values /= la.norm(values, axis=1)[:, None]

        # set the controller keys
        controller.data.rotations.keys = keys

        # update start/stop times
        controller.start_time = min(keys[0, 0], controller.start_time)
        controller.stop_time = max(keys[-1, 0], controller.stop_time)

        return True

    def create_scales(self, fcurves_dict):
        fcurves = self.collect_fcurves(fcurves_dict, "scale", num_axes=3)
        if len(fcurves) == 0:
            return False

        key_type = self.get_interpolation_type(fcurves)
        if key_type is None:
            return False

        # collect keyframe points
        keys = self.collect_keyframe_points(fcurves, key_type)
        if len(keys) == 0:
            return False

        # require uniform scaling
        scales_min = keys[:, 1:4].min(axis=1)
        scales_max = keys[:, 1:4].max(axis=1)
        if not np.allclose(scales_min, scales_max, rtol=0, atol=1e-4):
            print(f"({self.name}) non-uniform scale animations are not supported")
            return False

        # use the first axis only
        if key_type.name == 'LIN_KEY':
            keys = keys[:, :2]  # time, x.value
        else:
            keys = keys[:, (0, 1, 4, 7)]  # time, x.value, x.in_tan, x.out_tan

        # get keyframe controller
        controller = self.create_keyframe_controller()

        # set the controller keys
        controller.data.scales.key_type = key_type
        controller.data.scales.keys = keys

        # update start/stop times
        controller.start_time = min(keys[0, 0], controller.start_time)
        controller.stop_time = max(keys[-1, 0], controller.stop_time)

        # convert to local space
        offset = self.to_local_space()
        if not np.allclose(offset, ID44, rtol=0, atol=1e-4):
            offset_scale = decompose(offset)[-1][-1]
            s = controller.data.scales
            s.values[:] *= offset_scale
            if key_type.name == "BEZ_KEY":
                s.in_tans[:] *= offset_scale
                s.out_tans[:] *= offset_scale

        return True

    def create_uv_controller(self, bl_prop, bl_slot):
        if not self.exporter.export_animations:
            return False
        if not (bl_slot.image and bl_slot.layer):
            return False

        anims = self.get_fcurves_dict(bl_prop.texture_group.node_tree)
        if len(anims) == 0:
            return False

        uv_data = nif.NiUVData()
        bl_node = bl_slot.mapping_node

        channels = {
            (uv_data.u_offset_data, uv_data.v_offset_data):
                bl_node.inputs["Location"].path_from_id("default_value"),
            (uv_data.u_tiling_data, uv_data.v_tiling_data):
                bl_node.inputs["Scale"].path_from_id("default_value"),
        }

        # bail if no keyframes present
        if not any(map(anims.get, channels.values())):
            return False

        for outputs, data_path in channels.items():
            for i, output in enumerate(outputs):
                fcurves = [fc for fc in anims[data_path] if fc.array_index == i]

                key_type = self.get_interpolation_type(fcurves)
                if key_type is None:
                    continue

                output.key_type = key_type
                output.keys = self.collect_keyframe_points(fcurves, key_type)

        try:
            # TODO: do these in shader instead
            uv_data.u_offset_data.keys[:, 1] *= -1
            uv_data.v_offset_data.keys[:, 1] *= -1
        except AttributeError:
            pass

        # create controller
        controller = nif.NiUVController(
            cycle_type=self.exporter.cycle_type,
            target=self.output,
            data=uv_data
        )
        controller.update_start_stop_times()

        # find uv set index
        for i, uv in enumerate(self.source.data.uv_layers):
            if uv.name == bl_slot.layer:
                controller.texture_set = i
                break

        # attach controller
        self.output.controllers.appendleft(controller)

        return True

    def create_material_controllers(self, ni_prop, bl_prop):
        if not self.exporter.export_animations:
            return

        anims = self.get_fcurves_dict(bl_prop.material.node_tree)
        if len(anims) == 0:
            return

        self.create_color_controller(anims, ni_prop, bl_prop)
        self.create_alpha_controller(anims, ni_prop, bl_prop)

    def create_color_controller(self, fcurves_dict, ni_prop, bl_prop):
        channels = {
            'DIFFUSE': bl_prop.diffuse_input,
            'EMISSIVE': bl_prop.emissive_input,
        }

        for color_field, source in channels.items():
            data_path = source.path_from_id("default_value")
            fcurves = fcurves_dict[data_path]

            key_type = self.get_interpolation_type(fcurves)
            if key_type is None:
                continue

            keys = self.collect_keyframe_points(fcurves[:3], key_type)
            if len(keys) == 0:
                continue

            # create output controller
            controller = nif.NiMaterialColorController(
                cycle_type=self.exporter.cycle_type,
                color_field=color_field,
                target=ni_prop,
                data=nif.NiPosData(
                    key_type=key_type,
                    keys=keys
                ),
            )

            # update controller times
            controller.update_start_stop_times()

            # attach the controller
            ni_prop.controllers.appendleft(controller)

        return True

    def create_alpha_controller(self, fcurves_dict, ni_prop, bl_prop):
        data_path = bl_prop.opacity_input.path_from_id("default_value")
        fcurves = fcurves_dict[data_path]

        key_type = self.get_interpolation_type(fcurves)
        if key_type is None:
            return False

        keys = self.collect_keyframe_points(fcurves, key_type)
        if len(keys) == 0:
            return False

        # create output controller
        controller = nif.NiAlphaController(
            cycle_type=self.exporter.cycle_type,
            target=ni_prop,
            data=nif.NiFloatData(key_type=key_type, keys=keys),
        )

        # update controller times
        controller.update_start_stop_times()

        # attach the controller
        ni_prop.controllers.appendleft(controller)

        return True

    def to_local_space(self):
        if isinstance(self.source, bpy.types.PoseBone):
            matrix = self.source.id_data.convert_space(
                pose_bone=self.source, matrix=ID44.T, from_space="LOCAL", to_space="WORLD"
            )
            return la.solve(self.parent.matrix_world, matrix)
        else: # non-bone animations
            if self.parent is not None:
                return la.inv(self.parent.matrix_world)
            return ID44

    @staticmethod
    def get_fcurves_dict(bl_object):
        fcurves_dict = collections.defaultdict(list)

        try:
            fcurves = bl_object.animation_data.action.fcurves
        except AttributeError:
            pass
        else:
            for fc in fcurves:
                fcurves_dict[fc.data_path].append(fc)

        return fcurves_dict

    def collect_fcurves(self, fcurves_dict, key, num_axes):
        assert num_axes != 0

        # correct data path for pose bones
        if isinstance(self.source, bpy.types.PoseBone):
            data_path = f'pose.bones["{self.name}"].{key}'
        else:
            data_path = self.source.path_from_id(key)

        # skip invalid data paths
        if data_path not in fcurves_dict:
            return ()

        # collect defined fcurves
        fcurves = fcurves_dict[data_path]

        # fill in missing fcurves
        if len(fcurves) != num_axes:
            group_name = fcurves[0].group.name
            action = self.source.id_data.animation_data.action
            for i in set(range(num_axes)).difference(fc.array_index for fc in fcurves):
                fc = action.fcurves.new(data_path, index=i, action_group=group_name)

        # fill in missing keyframes
        scene = bpy.context.scene
        frames_per_axis = [{kp.co[0] for kp in fc.keyframe_points} for fc in fcurves]
        required_frames = {scene.frame_start, scene.frame_end}.union(*frames_per_axis)
        for fc, frames in zip(fcurves, frames_per_axis):
            for frame in (required_frames - frames):
                fc.keyframe_points.insert(frame, fc.evaluate(frame), options={'FAST'})

        # ensure fcurves are sorted
        fcurves.sort(key=lambda fc: fc.array_index)

        # ensure fcurves are updated
        for fc in fcurves:
            fc.update()

        return fcurves

    def create_keyframe_controller(self):
        result = self.output.controllers.find_type_with_owner(nif.NiKeyframeController)
        if result is not None:
            owner, controller = result
        else:
            owner = self.output
            self.output.controllers.appendleft(
                nif.NiKeyframeController(
                    cycle_type=self.exporter.cycle_type,
                    target=owner,
                    data=nif.NiKeyframeData(),
                )
            )
        return owner.controller

    @staticmethod
    def collect_keyframe_points(fcurves, key_type, dtype=np.float32):
        # collect/convert keyframe keys
        num_axes = len(fcurves)
        num_frames = len(fcurves[0].keyframe_points)
        num_values = 1 + num_axes * (1 if key_type.name == 'LIN_KEY' else 3)
        # e.g. times column + a values column with in/out tans for each axis

        keys = np.empty((num_frames, num_values), dtype)
        temp = np.empty((num_frames, 2), dtype)

        for i, fc in enumerate(fcurves, start=1):
            # collect times/values
            fc.keyframe_points.foreach_get("co", temp.ravel())
            keys[:, (0, i)] = temp

            if key_type.name == 'BEZ_KEY':
                # collect incoming tangents
                fc.keyframe_points.foreach_get("handle_left", temp.ravel())
                keys[:, i + num_axes * 1] = (keys[:, i] - temp[:, 1]) * 3.0
                # collect outgoing tangents
                fc.keyframe_points.foreach_get("handle_right", temp.ravel())
                keys[:, i + num_axes * 2] = (temp[:, 1] - keys[:, i]) * 3.0

        # convert from frames to times
        keys[:, 0] /= bpy.context.scene.render.fps

        # sort the keys by their times
        keys = keys[keys[:, 0].argsort()]

        return keys

    @staticmethod
    def get_interpolation_type(fcurves):
        # Blender lets each keyframe point define its own interpolation mode.
        # Morrowind supports only a single interpolation mode per-controller.
        # For now we will just use the interpolation mode from the first key.
        try:
            interpolation = fcurves[0].keyframe_points[0].interpolation
        except IndexError:
            return None
        if interpolation == 'LINEAR':
            return nif.NiFloatData.KeyType.LIN_KEY
        else:
            return nif.NiFloatData.KeyType.BEZ_KEY
