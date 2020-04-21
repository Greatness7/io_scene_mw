import bpy

import math
import timeit
import pathlib
import itertools
import collections

import numpy as np
import numpy.linalg as la

from es3 import nif
from es3.utils import meshoptimizer
from es3.utils.math import ID44, compose, decompose

from . import nif_utils
from . import nif_shader

from bpy_extras.io_utils import axis_conversion

biped_axis_correction = np.array(axis_conversion('Y', 'Z', '-X', 'Z').to_4x4(), dtype="<f")
biped_axis_correction_inverse = la.inv(biped_axis_correction)

other_axis_correction = np.array(axis_conversion('-Z', '-Y', 'Y', 'Z').to_4x4(), dtype="<f")
other_axis_correction_inverse = la.inv(other_axis_correction)


def save(context, filepath, **config):
    """save the scene to a nif file"""

    print(f"Export File: {filepath}")
    time = timeit.default_timer()

    exporter = Exporter(config)
    exporter.save(filepath)

    time = timeit.default_timer() - time
    print(f"Export Done: {time:.4f} seconds")

    return {"FINISHED"}


class Exporter:
    vertex_precision = 0.001
    extract_keyframe_data = False

    def __init__(self, config):
        vars(self).update(config)
        self.nodes = {}  # type: Dict[SceneNode, Type]
        self.materials = {}  # type: Dict[FrozenSet[NiProperty], NiMaterialProps]
        self.history = collections.defaultdict(set)  # type: Dict[NiAVObject, Set[SceneNode]]
        self.armatures = collections.defaultdict(set)  # type: Dict[SceneNode, Set[NiNode]]
        self.colliders = collections.defaultdict(set)  # type: Dict[NiNode, Set[NiNode]]
        self.depsgraph = None

    def save(self, filepath):
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
        data.save(filepath)

        # extract x/kf file
        if self.extract_keyframe_data:
            self.export_keyframe_data(data, filepath)

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
        (root, bones), = self.armatures.items()

        root_node = self.get(root)  # __history__
        root_matrix = root_node.matrix_world

        # convert from armature space to local space
        for node in map(self.get, bones):  # __history__
            node.matrix_world = root_matrix @ node.matrix_local

        # update parent of those using bone parenting
        for node in list(root_node.children):
            if type(node.source) is bpy.types.Object:
                if node.source.parent_type == "BONE":
                    node.parent = self.get(bones[node.source.parent_bone])  # __history__
                    node.matrix_world = np.array(node.source.matrix_world)

    def apply_axis_corrections(self):
        """ TODO
            support multiple armatures
            only apply on 'Bip01' nodes
        """
        (root, bones), = self.armatures.items()

        root_matrix = self.get(root).matrix_world  # __history__
        root_inverse = la.inv(root_matrix)

        for node in map(self.get, root.pose.bones):
            child_world_matrices = {c: c.matrix_world for c in node.children}

            node.matrix_bind = root_inverse @ node.matrix_world @ node.axis_correction
            node.matrix_world = root_matrix @ node.source.matrix @ node.axis_correction

            # ensure child transforms were not modified
            for child, matrix in child_world_matrices.items():
                child.matrix_world = matrix

    def resolve_depsgraph(self):
        temp_modifiers = []
        temp_hide_view = []
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

                # ensure meshes are triangualted
                if not is_triangulated:
                    m = source.modifiers.new("", "TRIANGULATE")
                    m.keep_custom_normals = True
                    m.quad_method = "FIXED"
                    m.ngon_method = "CLIP"
                    temp_modifiers.append((source, m))

            # get evaluated dependency graph
            self.depsgraph = bpy.context.evaluated_depsgraph_get()
        finally:
            for m in temp_hide_view:
                m.show_viewport = True
            for s, m in temp_modifiers:
                s.modifiers.remove(m)

    @staticmethod
    def export_keyframe_data(data, filepath):
        path = pathlib.Path(filepath)
        xnif_path = path.with_name("x" + path.name)
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
        bones = node.source.pose.bones
        self.armatures[node.source] = bones
        self.resolve_nodes(set(bones), parent=node)
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
        if len(roots) == 1:
            return roots[0].output
        # return nif.NiBSAnimationNode(name="Scene Root", flags=32, children=[r.output for r in roots])
        return nif.NiNode(name="Scene Root", flags=32, children=[r.output for r in roots])

    @property
    def scale_correction(self):
        addon = bpy.context.preferences.addons["io_scene_mw"]
        return 1 / addon.preferences.scale_correction


class SceneNode:
    """ TODO
        support for multiple armatures
    """

    def __init__(self, exporter, source, parent=None):
        self.exporter = exporter
        #
        self.source = source
        self.output = None
        #
        self.parent = parent
        self.children = list()
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

    @property
    def name(self):
        return self.source.name

    @property
    def bone_name(self):
        name = self.source.name
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
        if "Bip01" in self.source.name:
            return biped_axis_correction
        return other_axis_correction

    @property
    def axis_correction_inverse(self):
        if "Bip01" in self.source.name:
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
            print(f"[INFO] {self.name} has non-uniform scale")

            if self.source.type == "MESH":
                self.vertex_transform = s

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
        c, r, e = decompose(self.matrix_world)
        e *= self.source.empty_display_size
        self.output.bounding_volume = nif.NiBoxBV(center=c, rotation=r, extents=e)
        self.output.matrix = ID44


class Armature(SceneNode):
    __slots__ = ()

    def __init__(self, node):
        self.__dict__ = node.__dict__

    def create(self):
        # create root
        Empty(self).create()
        # create bones
        for pose in self.source.pose.bones:
            node = self.exporter.get(pose)  # __history__
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

        ni.vertices = ni.vertices[ni.triangles].reshape(-1, 3)

    def create_triangles(self, ni, bl):
        ni.triangles.resize(len(bl.polygons), 3)
        bl.polygons.foreach_get("vertices", ni.triangles.ravel())

    def create_normals(self, ni, bl):
        if self.is_collider:
            return

        # TODO custom UI for ignore normals
        if not bl["ignore_normals"]:
            ni.normals.resize(len(ni.vertices), 3)
            bl.calc_normals_split()
            bl.loops.foreach_get("normal", ni.normals.ravel())
            bl.free_normals_split()
            # RuntimeWarning: invalid value encountered in true_divide
            ni.normals /= la.norm(ni.normals, axis=1)[:, None]

    def create_uv_sets(self, ni, bl):
        if self.is_collider:
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
        controller = nif.NiGeomMorpherController(flags=12)
        controller.stop_time = max(kf[-1, 0] for kf in filter(len, morph_data.keys))
        controller.target, trishape.controller = trishape, controller

        # create morph data
        controller.data = nif.NiMorphData(relative_targets=1)
        controller.data.targets = [nif.NiMorphDataMorphTarget() for _ in morph_data.targets]

        # assign morph data
        for i, target in enumerate(controller.data.targets):
            target.interpolation = 1  # LINEAR_KEY
            target.keys = morph_data.keys[i]
            target.vertices = morph_data.targets[i]

    # -- get blender data --

    def get_mesh_data(self):
        """ TODO
            Use existing TRIANGULATE modifier if present.
            Fix the depsgraph implementation performance.
        """
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
        skin_data = nif_utils.Namespace(weights=())

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
        for i, vertex in enumerate(bl_data.vertices):
            for vg in vertex.groups:
                if vg.group in bones:
                    weights[vg.group, i] = vg.weight

        # limit bones per vertex
        if len(bones) > 4:
            # indices of all but the 4 highest weights
            indices = np.argpartition(weights, -4, axis=0)[:-4]
            # set the values at target indices to zero
            np.put_along_axis(weights, indices, 0, axis=0)

        # normalize bone weights
        sums = weights.sum(axis=0, keepdims=True)
        sums[sums == 0] = 1  # divide by zero fix
        weights /= sums

        # view as per-face-loops
        weights = weights[:, ni_data.triangles].reshape(len(bones), -1)

        # get all armature nodes
        skin_data.root = self.exporter.get(armature)  # __history__
        skin_data.bones = [self.exporter.get(bone) for bone in bones.values()]  # __history__
        skin_data.weights = weights

        return skin_data

    def get_morph_data(self, bl_data, ni_data):
        morph_data = nif_utils.Namespace(targets=())

        # ignore collider meshes
        if self.is_collider:
            return morph_data

        try:
            shape_keys = self.source.data.shape_keys
            action = shape_keys.animation_data.action
        except AttributeError:
            # source object does not have morph data
            return morph_data

        if not (shape_keys.key_blocks and action.fcurves):
            return morph_data

        basis = shape_keys.key_blocks[0]
        fcurves = {fc.data_path: fc for fc in action.fcurves}

        # isolate unmuted
        fcurves_to_shape_keys = {None: basis}
        for sk in shape_keys.key_blocks:
            fc = fcurves.get(sk.path_from_id('value'))
            if fc and not sk.mute:
                fcurves_to_shape_keys[fc] = sk
                assert sk.relative_key == basis

        # allocate morph targets
        morph_targets = np.empty((len(fcurves_to_shape_keys), len(bl_data.vertices), 3))

        # allocate morph sk list
        morph_data.keys = [None] * len(fcurves_to_shape_keys)

        # collect animation data
        for i, (fc, sk) in enumerate(fcurves_to_shape_keys.items()):
            if i == 0:
                # this is basis shape key
                kf_array = np.empty(0)
            else:
                # allocate keyframe array
                kf_array = np.empty((len(fc.keyframe_points), 2))
                # populate keyframe array
                fc.keyframe_points.foreach_get("co", kf_array.ravel())
                # convert frames to times
                kf_array[:, 0] /= bpy.context.scene.render.fps

            # collect keyframe array
            morph_data.keys[i] = kf_array

            # populate morph targets
            sk.data.foreach_get("co", morph_targets[i].ravel())

        # make relative to basis
        morph_targets[1:] -= morph_targets[0]

        # view as per-face-loops
        morph_data.targets = morph_targets[:, ni_data.triangles].reshape(-1, len(ni_data.vertices), 3)

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
            material_data.append((material, indices))

        return material_data

    # -- optimizer functions --

    def optimize_geometry(self, data, skin_data, morph_data):
        if not len(data.vertices):
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
        data.triangles = np.arange(inverse.size, dtype=int).reshape(-1, 3)
        data.triangles = inverse[data.triangles]

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
        vertex_remap, triangles = meshoptimizer.optimize(data.vertices, data.triangles)
        if len(data.triangles):
            data.triangles = triangles
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

        if material in self.exporter.materials:
            ni_object.properties = self.exporter.materials[material]
            return  # material already processed

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

        # Update Material Cache
        self.exporter.materials[material] = ni_object.properties

    def create_material_property(self, ni_object, bl_prop):
        ni_prop = nif.NiMaterialProperty()
        # Material Name
        ni_prop.name = bl_prop.material.name
        # Material Flags
        ni_prop.flags = bl_prop.material_flags
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
        # Texture Flags
        ni_prop.flags = bl_prop.texture_flags
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
        ni_slot.source.filename = bpy.path.abspath(filepath)
        setattr(ni_prop, name, ni_slot)

        # uv animations
        self.animation.create_uv_controller(bl_prop, bl_slot)

    def create_wireframe_property(self, ni_object, bl_prop):
        if self.source.display_type == "WIRE":
            ni_prop = nif.NiWireframeProperty(wireframe=True)
            ni_object.properties.append(ni_prop)

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
        # create text keys even if no animations as assigned
        self.create_text_keys()

        anims = self.collect_animations(self.source.id_data)
        if len(anims) == 0:
            return

        # visibility data
        self.create_vis_controller(anims)

        # translation data
        self.create_translations(anims)

        # rotation data
        self.create_rotations(anims)

        # scale data
        self.create_scales(anims)

    def create_text_keys(self):
        try:
            markers = self.source.animation_data.action.pose_markers
        except AttributeError:
            return

        if len(markers) == 0:
            return

        text_data = self.output.extra_data = nif.NiTextKeyExtraData()
        text_data.keys.resize(len(markers))

        for i, marker in enumerate(markers):
            time = marker.frame / bpy.context.scene.render.fps
            name = marker.name.replace("; ", "\r\n")
            text_data.keys[i] = time, name

        text_data.keys.sort()

    def create_vis_controller(self, anims):
        path = self.get_anim_path("hide_viewport")
        fcurves = anims[path]
        if len(fcurves) == 0:
            return False

        keys = self.collect_keyframe_points(fcurves, 1)
        if len(keys) == 0:
            return False

        controller = nif.NiVisController(
            frequency = 1.0,
            target = self.output,
            data = nif.NiVisData(),
        )
        controller.data.keys.resize(len(keys))
        controller.data.keys["f0"] = keys[:, 0]
        controller.data.keys["f1"] = 1 - keys[:, 1]

        # update controller times
        controller.update_start_stop_times()

        self.output.controllers.appendleft(controller)

        return True

    # -- transform controllers --

    def create_translations(self, anims):
        path = self.get_anim_path("location")
        fcurves = anims[path]
        if len(fcurves) == 0:
            return False

        # get keyframe controller
        controller = self.create_keyframe_controller()

        # collect keyframe points
        keys = self.collect_keyframe_points(fcurves, 3)
        if len(keys) == 0:
            return False

        # split times from values
        values = keys[:, 1:5]

        # convert to output space
        if isinstance(self.source, bpy.types.PoseBone):
            offset = self.get_posed_offset()
            values[:] = values @ offset[:3, :3].T + offset[:3, 3]

        # set the controller keys
        controller.data.translations.keys = keys

        # update start/stop times
        controller.start_time = min(keys[0, 0], controller.start_time)
        controller.stop_time = max(keys[-1, 0], controller.stop_time)

        return True

    def create_rotations(self, anims):
        has_euler = self.create_euler_rotations(anims)
        has_quats = self.create_quaternion_rotations(anims)
        if has_euler and has_quats:
            raise ValueError(f"'({self.name})' mixing euler and quaternion rotations in the same action is not supported")
        return has_euler or has_quats

    def create_scales(self, anims):
        path = self.get_anim_path("scale")
        fcurves = anims[path]
        if len(fcurves) == 0:
            return False

        # collect keyframe points
        keys = self.collect_keyframe_points(fcurves, 3)
        if len(keys) == 0:
            return

        # get keyframe controller
        controller = self.create_keyframe_controller()

        # require uniform scaling
        if not np.allclose(keys[:, 1:].min(1), keys[:, 1:].max(1), rtol=0, atol=1e-04):
            print(f"({self.name}) non-uniform scale animations are not supported")
            return False

        # set the controller keys
        controller.data.scales.keys = keys[:, :2]

        # update start/stop times
        controller.start_time = min(keys[0, 0], controller.start_time)
        controller.stop_time = max(keys[-1, 0], controller.stop_time)

        return True

    def create_euler_rotations(self, anims):
        path = self.get_anim_path("rotation_euler")
        fcurves = anims[path]
        if len(fcurves) == 0:
            return False

        # collect keyframe points
        keys = self.collect_keyframe_points(fcurves, 3)
        if len(keys) == 0:
            return

        # split times from values
        values = keys[:, 1:]

        # get keyframe controller
        controller = self.create_keyframe_controller()

        # pose / axis corrections
        if isinstance(self.source, bpy.types.PoseBone):
            from mathutils import Euler, Matrix
            mode = self.source.rotation_mode
            axis_fix = Matrix(self.axis_correction)
            to_posed = Matrix(self.get_posed_offset())
            for i, v in enumerate(values):
                v = Euler(v, mode).to_matrix().to_4x4()
                v = (to_posed @ v @ axis_fix).to_euler()
                values[i] = v

        # set the controller keys
        rotations = controller.data.rotations
        rotations.interpolation = nif.NiRotData.KeyType.EULER_KEY
        for i in range(3):
            rotations.euler_data += nif.NiFloatData(),
            rotations.euler_data[i].interpolation = nif.NiFloatData.KeyType.LIN_KEY
            rotations.euler_data[i].keys = keys[:, (0, i+1)]

        # update start/stop times
        controller.start_time = min(keys[0, 0], controller.start_time)
        controller.stop_time = max(keys[-1, 0], controller.stop_time)

        return True

    def create_quaternion_rotations(self, anims):
        path = self.get_anim_path("rotation_quaternion")
        fcurves = anims[path]
        if len(fcurves) == 0:
            return False

        # collect keyframe points
        keys = self.collect_keyframe_points(fcurves, 4)
        if len(keys) == 0:
            return False

        # get keyframe controller
        controller = self.create_keyframe_controller()

        # split times from values
        values = keys[:, 1:]

        # pose / axis corrections
        if isinstance(self.source, bpy.types.PoseBone):
            offset = nif_utils.quaternion_from_matrix(self.axis_correction)
            nif_utils.quaternion_mul(values, offset, out=values)
            offset = nif_utils.quaternion_from_matrix(self.get_posed_offset())
            nif_utils.quaternion_mul(offset, values, out=values)

        # normalize rotation keys
        values /= la.norm(values, axis=1)[:, None]

        # set the controller keys
        controller.data.rotations.keys = keys

        # update start/stop times
        controller.start_time = min(keys[0, 0], controller.start_time)
        controller.stop_time = max(keys[-1, 0], controller.stop_time)

        return True

    # -- shader controllers --

    def create_uv_controller(self, bl_prop, bl_slot):
        anims = self.collect_animations(bl_prop.texture_group.node_tree)
        if len(anims) == 0:
            return

        bl_node = bl_slot.mapping_node

        # offset keys
        data_path = bl_node.inputs["Location"].path_from_id('default_value')
        fcurves = anims[data_path]
        offset_keys = self.collect_keyframe_points(fcurves, 3)

        # tiling keys
        data_path = bl_node.inputs["Scale"].path_from_id('default_value')
        fcurves = anims[data_path]
        tiling_keys = self.collect_keyframe_points(fcurves, 3)

        # is animated
        if len(offset_keys) == len(tiling_keys) == 0:
            return

        controller = nif.NiUVController(
            frequency = 1.0,
            target = self.output,
            data = nif.NiUVData(),
        )

        # uv offset
        if len(offset_keys):
            offset_keys[:, 1] = 0 - offset_keys[:, 1]
            offset_keys[:, 2] = 1 - offset_keys[:, 2]
            controller.data.offset_u.keys = offset_keys[:, (0, 1)]
            controller.data.offset_v.keys = offset_keys[:, (0, 2)]

        # uv tiling
        if len(tiling_keys):
            controller.data.tiling_u.keys = tiling_keys[:, (0, 1)]
            controller.data.tiling_v.keys = tiling_keys[:, (0, 2)]

        # target uv set
        for i, uv in enumerate(self.source.data.uv_layers):
            if uv.name == bl_slot.layer:
                controller.texture_set = i
                break

        # update times
        controller.update_start_stop_times()

        # attach the controller
        self.output.controllers.appendleft(controller)

    def create_material_controllers(self, ni_prop, bl_prop):
        anims = self.collect_animations(bl_prop.material.node_tree)
        if len(anims) == 0:
            return

        self.create_color_controller(anims, ni_prop, bl_prop)
        self.create_alpha_controller(anims, ni_prop, bl_prop)

    def create_color_controller(self, anims, ni_prop, bl_prop):
        channels = [
            (bl_prop.diffuse_input, 'DIFFUSE'),
            (bl_prop.emissive_input, 'EMISSIVE'),
        ]

        for source, color_field in channels:
            data_path = source.path_from_id("default_value")
            fcurves = anims[data_path]
            keys = self.collect_keyframe_points(fcurves, 4)
            if len(keys) == 0:
                continue

            # create output controller
            controller = nif.NiMaterialColorController(
                target=ni_prop,
                color_field=color_field,
                data=nif.NiPosData(keys=keys[:, :4]),
            )

            # update controller times
            controller.update_start_stop_times()

            # attach the controller
            ni_prop.controllers.appendleft(controller)

    def create_alpha_controller(self, anims, ni_prop, bl_prop):
        data_path = bl_prop.opacity_input.path_from_id("default_value")
        fcurves = anims[data_path]
        keys = self.collect_keyframe_points(fcurves, 1)
        if len(keys) == 0:
            return

        # create output controller
        controller = nif.NiAlphaController(
            target=ni_prop,
            data=nif.NiFloatData(keys=keys),
        )

        # update controller times
        controller.update_start_stop_times()

        # attach the controller
        ni_prop.controllers.appendleft(controller)

    # -- get blender data --

    @staticmethod
    def collect_animations(bl_object):
        try:
            fcurves = bl_object.animation_data.action.fcurves
        except AttributeError:
            return {}

        anims = collections.defaultdict(list)
        for fc in fcurves:
            anims[fc.data_path].append(fc)

        return anims

    def get_anim_path(self, key):
        if isinstance(self.source, bpy.types.PoseBone):
            return f'pose.bones["{self.name}"].{key}'
        return self.source.path_from_id(key)

    def get_posed_offset(self):
        offset = self.source.id_data.convert_space(
            pose_bone=self.source,
            matrix=ID44.T,
            from_space="LOCAL",
            to_space="WORLD",
        )
        return la.solve(self.parent.matrix_world, offset)

    # -- get nif controllers --

    def create_keyframe_controller(self):
        result = self.output.controllers.find_type_with_owner(nif.NiKeyframeController)
        if result is not None:
            owner, controller = result
        else:
            owner = self.output
            self.output.controllers.appendleft(
                nif.NiKeyframeController(
                    flags=12,
                    target=owner,
                    data=nif.NiKeyframeData(),
                )
            )
        return owner.controller

    # --

    @staticmethod
    def collect_keyframe_points(fcurves, size, dtype=np.float32):
        template = [np.nan] * (size + 1)

        # parse keyframe points
        d = collections.defaultdict(template.copy)
        for fc in fcurves:
            i = fc.array_index + 1
            for p in fc.keyframe_points:
                if p.type in ('KEYFRAME', 'BREAKDOWN'):
                    frame, value = p.co
                    d[frame][i] = value

        # prep array parameters
        count = len(d) * len(template)
        values = itertools.chain.from_iterable(d.values())
        if count == 0:
            return ()

        # create the keys array
        keys = np.fromiter(values, dtype, count).reshape(len(d), -1)
        keys[:, 0] = np.fromiter(d.keys(), dtype, len(d))  # frames

        # sort by frame numbers
        keys = keys[keys[:, 0].argsort()]

        # convert frame to time
        keys[:, 0] /= bpy.context.scene.render.fps

        # evaluate nan elements
        i = np.arange(keys.size).reshape(keys.shape)
        np.putmask(i, np.isnan(keys), 0)
        keys = keys.take(np.maximum.accumulate(i))

        return keys
