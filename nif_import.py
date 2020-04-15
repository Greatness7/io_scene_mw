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

biped_axis_correction = np.array(axis_conversion('-X', 'Z', 'Y', 'Z').to_4x4(), dtype="<f")
biped_axis_correction_inverse = la.inv(biped_axis_correction)

other_axis_correction = np.array(axis_conversion('Y', 'Z', '-Z', '-Y').to_4x4(), dtype="<f")
other_axis_correction_inverse = la.inv(other_axis_correction)


def load(context, filepath, **config):
    """load a scene from a nif file"""

    print(f"Import File: {filepath}")
    time = timeit.default_timer()

    importer = Importer(config)
    importer.load(filepath)

    time = timeit.default_timer() - time
    print(f"Import Done: {time:.4f} seconds")

    return {"FINISHED"}


class Importer:
    vertex_precision = 0.001
    attach_keyframe_data = False
    discard_root_transforms = True

    def __init__(self, config):
        vars(self).update(config)
        self.nodes = {}  # type: Dict[SceneNode, Type]
        self.materials = {}  # type: Dict[FrozenSet[NiProperty], NiMaterialProps]
        self.history = collections.defaultdict(set)  # type: Dict[NiAVObject, Set[SceneNode]]
        self.armatures = collections.defaultdict(set)  # type: Dict[NiNode, Set[NiNode]]
        self.colliders = collections.defaultdict(set)  # type: Dict[NiNode, Set[NiNode]]

    def load(self, filepath):
        data = nif.NiStream()
        data.load(filepath)

        # fix transforms
        if self.discard_root_transforms:
            data.root.matrix = ID44

        # attach kf file
        if self.attach_keyframe_data:
            self.import_keyframe_data(data, filepath)

        # apply settings
        data.apply_scale(self.scale_correction)

        # resolve heirarchy
        roots = self.resolve_nodes(data.roots)

        # resolve armatures
        if any(self.armatures):
            self.resolve_armatures()
            self.apply_axis_corrections()
            self.correct_rest_positions()

        # create bl objects
        for node, cls in self.nodes.items():
            if node.output is None:
                cls(node).create()
                node.animation.create()

        # set active object
        bpy.context.view_layer.objects.active = self.get_root_output(roots)

    # -------
    # RESOLVE
    # -------

    def resolve_nodes(self, ni_roots, parent=None):
        root_nodes = [SceneNode(self, root, parent) for root in ni_roots]

        queue = collections.deque(root_nodes)
        while queue:
            node = queue.popleft()
            if self.process(node):
                self.history[node.source].add(node)
                if hasattr(node.source, "children"):
                    queue.extend(SceneNode(self, child, node) for child in node.source.children if child)

        return root_nodes

    def resolve_armatures(self):
        """ TODO
            support for multiple skeleton roots
        """
        orphan_bones = self.armatures.pop(None, {})

        if len(self.armatures) == 1:
            (root, bones), = self.armatures.items()
        else:
            root = next(n.source for n in self.nodes if self.armatures.get(n.source))
            bones = self.armatures[root]

        # collect all orphan bones
        bones.update(orphan_bones)

        # collect all others bones
        for other_root in self.armatures.keys() - {root}:
            other_bones = self.armatures.pop(other_root)
            bones.add(other_root)
            bones.update(other_bones)

        # only descendants of root
        bones.discard(root)
        bones -= {p.source for p in self.get(root).parents}  # __history__

        # bail if no bones present
        if len(bones) == 0:
            self.armatures.clear()
            return

        # validate all bone chains
        for node in list(map(self.get, bones)):  # __history__
            for parent in node.parents:
                source = parent.source
                if (source is root) or (source in bones):
                    break
                bones.add(source)

        # order bones by heirarchy
        self.armatures[root] = dict.fromkeys(n.source for n in self.nodes if n.source in bones).keys()

        # specify node as Armature
        self.nodes[self.get(root)] = Armature  # __history__

    def apply_axis_corrections(self):
        """ TODO
            Support multiple armatures.
            Could this be moved into nif library?
            Update Animations here, rather than later?
        """
        if not self.armatures:
            return

        root = self.get(*self.armatures)  # __history__
        bones = list(self.iter_bones(root))  # __history__

        # TODO handle undefined bones
        # These don't get sent to bind position properly, see: armor.1st files
        errors = self.armatures[root.source] - {b.source for b in bones}
        if errors:
            print(f"Warning: Undefined Bone Bind Poses!\n\t{errors}")

        # preserve bone pose matrices
        for node in bones:
            node.matrix_posed = node.matrix_world @ node.axis_correction

        # send all bones to rest pose
        root.source.apply_bone_bind_poses()
        root.source.apply_skins(keep_skins=True)

        # apply bone axis corrections
        for node in reversed(bones):
            node.matrix_local = node.source.matrix @ node.axis_correction
            for child in node.children:
                child.matrix_local = node.axis_correction_inverse @ child.matrix_local

    def correct_rest_positions(self):
        if not self.armatures:
            return

        root = self.get(*self.armatures)  # __history__
        root_bone = next(self.iter_bones(root))  # __history__

        # calculate corrected transformation matrix
        l, r, s = decompose(root_bone.matrix_posed)
        r = nif_utils.snap_rotation(r)
        corrected_matrix = compose(l, r, s)

        # correct the rest matrix of skinned meshes
        bone_inverse = la.inv(root_bone.matrix_world)
        for node in self.nodes:
            skin = getattr(node.source, "skin", None)
            if skin and (skin.root is root.source) and (root_bone not in node.parents):
                node.matrix_world = corrected_matrix @ bone_inverse @ node.matrix_world

        # correct the rest matrix of the root bone
        root_bone.matrix_world = corrected_matrix

    # -------
    # PROCESS
    # -------

    @nif_utils.dispatcher
    def process(self, node):
        print(f"Warning: Unhandled Type: {node.source.type}")
        return False

    @process.register("NiNode")
    @process.register("NiLODNode")
    @process.register("NiSwitchNode")
    @process.register("NiBSAnimationNode")
    def process_empty(self, node):
        self.nodes[node] = Empty

        # detect bones via name conventions
        name = node.name.lower()
        if name == "bip01":
            self.armatures[node.source].update()
        elif ("bip01" in name) or ("bone" in name):
            self.armatures[None].add(node.source)

        return True

    @process.register("NiTriShape")
    def process_mesh(self, node):
        self.nodes[node] = Mesh

        # track skinned meshes
        skin = node.source.skin
        if skin and skin.root and skin.bones:
            self.armatures[skin.root].update(skin.bones)

        return True

    @process.register("RootCollisionNode")
    def process_collision(self, node):
        self.nodes[node] = Empty
        self.colliders[node.source].update(node.source.descendants())
        return True

    # -------
    # UTILITY
    # -------

    def get(self, source):
        return next(iter(self.history[source]))  # __history__

    def iter_bones(self, root):
        yield from map(self.get, self.armatures[root.source])  # __history__

    def get_root_output(self, roots):
        return roots[0].output.id_data if roots else None

    @staticmethod
    def import_keyframe_data(data, filepath):
        kf_path = pathlib.Path(filepath).with_suffix(".kf")
        if not kf_path.exists():
            print(f'import_keyframe_data: "{kf_path}" does not exist')
        else:
            kf_data = nif.NiStream()
            kf_data.load(kf_path)
            data.attach_keyframe_data(kf_data)

    @property
    def scale_correction(self):
        addon = bpy.context.preferences.addons["io_scene_mw"]
        return addon.preferences.scale_correction


class SceneNode:
    """ TODO
        support for multiple armatures
    """

    def __init__(self, importer, source, parent=None):
        self.importer = importer
        #
        self.source = source
        self.output = None
        #
        self.parent = parent
        self.children = list()
        self.matrix_local = np.asarray(source.matrix, dtype="<f")

    def __repr__(self):
        if not self.parent:
            return f'SceneNode("{self.name}", parent=None)'
        return f'SceneNode("{self.name}", parent={self.parent.name})'

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
        props = {type(p): p for p in self.source.properties}
        if self.parent:
            return {**self.parent.properties, **props}
        return props

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
        self.output.empty_display_size *= self.importer.scale_correction
        self.output.mw.object_flags = self.source.flags

        bl_parent = getattr(self.parent, "output", None)
        try:
            self.output.parent = bl_parent
        except TypeError:
            # parent is an armature bone
            self.output.parent = bl_parent.id_data
            self.output.parent_type = "BONE"
            self.output.parent_bone = bl_parent.name
            self.output.matrix_world = (self.parent.matrix_posed @ self.matrix_local).T
        else:
            # parent is an empty or None
            self.output.matrix_local = self.matrix_local.T

        if self.source in self.importer.colliders:
            self.output.name = "Collision"
            self.output.display_type = "WIRE"

        if self.source.is_bounding_box:
            self.convert_to_bounding_box()

        return self.output

    def create_object(self, bl_data=None):
        bl_object = bpy.data.objects.new(self.name, bl_data)
        bpy.context.scene.collection.objects.link(bl_object)
        bl_object.select_set(True)
        return bl_object

    def convert_to_bounding_box(self):
        self.output.empty_display_size = 1.0
        self.output.empty_display_type = 'CUBE'
        self.output.matrix_world = self.source.bounding_volume.matrix.T


class Armature(SceneNode):
    __slots__ = ()

    def __init__(self, node):
        self.__dict__ = node.__dict__

    def create(self):
        # create armature object
        bl_data = bpy.data.armatures.new(self.name)
        bl_object = Empty(self).create(bl_data)

        # apply default settings
        bl_data.display_type = "STICK"
        bl_object.show_in_front = True

        # swap to edit mode to allow creation of bones
        bpy.context.view_layer.objects.active = bl_object
        bpy.ops.object.mode_set(mode="EDIT")

        # used for calculating armature space matrices
        root_inverse = la.inv(self.matrix_world)

        # bone mappings cache
        bones = {}

        # position bone heads
        for node in self.importer.iter_bones(self):  # __history__
            # create bone and assign its parent
            bone = bones[node] = bl_data.edit_bones.new(node.bone_name)
            bone.parent = bones.get(node.parent)
            bone.select = True

            # compute the armature-space matrix
            matrix = root_inverse @ node.matrix_world

            # calculate axis/roll and head/tail
            bone.matrix = matrix.T
            bone.tail = matrix[:3, 1] + matrix[:3, 3]  # axis + head

        # position bone tails
        for node, bone in bones.items():
            # edit_bones will not persist outside of edit mode
            bones[node] = bone.name

            if bone.children:
                # calculate length from children mean location
                locations = [c.matrix_posed[:3, 3] for c in node.children if c in bones]
                bone.length = la.norm(node.matrix_posed[:3, 3] - np.mean(locations, axis=0))
            elif bone.parent:
                # set length to half of the parent bone length
                bone.length = bone.parent.length / 2

            if bone.length <= 0:
                # TODO figure out a proper fix for zero length bones
                bone.tail = bone.tail + type(bone.tail)([0, 0, -1])
                print(f"Zero length bones are not supported! ({bone})")

        # back to object mode now that all bones exist
        bpy.ops.object.mode_set(mode="OBJECT")

        # assign node.output and apply pose transforms
        for node, name in bones.items():
            pose_bone = node.output = bl_object.pose.bones[name]
            # compute the armature-space matrix
            pose_bone.matrix = (root_inverse @ node.matrix_posed).T
            # TODO try not to call scene update
            bpy.context.view_layer.depsgraph.update()
            # create animations
            node.animation.create()

        return bl_object


class Mesh(SceneNode):
    __slots__ = ()

    def __init__(self, node):
        self.__dict__ = node.__dict__

    def create(self):
        bl_data = bpy.data.meshes.new(self.name)
        bl_object = Empty(self).create(bl_data)

        ni_data = self.get_mesh_data()

        self.create_vertices(bl_object, ni_data.vertices)
        self.create_triangles(bl_object, ni_data.triangles)

        self.create_normals(bl_object, ni_data.normals)
        self.create_vertex_colors(bl_object, ni_data.vertex_colors)
        self.create_uv_sets(bl_object, ni_data.uv_sets)

        self.create_vertex_weights(bl_object, ni_data.vertex_weights)
        self.create_vertex_morphs(bl_object, ni_data.vertex_morphs)

        bl_data.validate(verbose=False, clean_customdata=False)

        # if self.source.is_shadow:
        #     bl_object.display_type = "WIRE"

        # if self.importer.show_mesh_radius:
        #     self.create_mesh_radius()

        try:
            self.output.display_type = self.parent.output.display_type
        except AttributeError:
            pass

        self.material.create()

        return bl_object

    @staticmethod
    def create_vertices(ob, vertices):
        ob.data.vertices.add(len(vertices))
        ob.data.vertices.foreach_set("co", vertices.ravel())

    @staticmethod
    def create_triangles(ob, triangles):
        n = len(triangles)
        ob.data.loops.add(3 * n)
        ob.data.loops.foreach_set("vertex_index", triangles.ravel())

        ob.data.polygons.add(n)
        ob.data.polygons.foreach_set("loop_total", [3] * n)
        ob.data.polygons.foreach_set("loop_start", range(0, 3 * n, 3))
        ob.data.polygons.foreach_set("use_smooth", [True] * n)

        ob.data.update()

    @staticmethod
    def create_normals(ob, normals):
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
            ob.data.use_auto_smooth = True
            ob.data.normals_split_custom_set(normals)
            # ob.data.edges.foreach_set("use_edge_sharp", [False] * len(ob.data.edges))

    @staticmethod
    def create_uv_sets(ob, uv_sets):
        for i, uv in enumerate(uv_sets):
            ob.data.uv_layers.new()
            ob.data.uv_layers[i].data.foreach_set("uv", uv.ravel())

    @staticmethod
    def create_vertex_colors(ob, vertex_colors):
        if len(vertex_colors):
            vc = ob.data.vertex_colors.new()
            vc.data.foreach_set("color", vertex_colors.ravel())

    def create_vertex_weights(self, ob, vertex_weights):
        if not len(vertex_weights):
            return

        root = self.importer.get(self.source.skin.root)  # __history__
        bones = map(self.importer.get, self.source.skin.bones)  # __history__

        # Make Armature
        armature = ob.modifiers.new("", "ARMATURE")
        armature.object = root.output.id_data

        # Vertex Weights
        for i, node in enumerate(bones):
            vg = ob.vertex_groups.new(name=node.output.name)

            weights = vertex_weights[i]
            for j in np.flatnonzero(weights).tolist():
                vg.add([j], weights[j], "ADD")

    def create_vertex_morphs(self, ob, vertex_morphs):
        if not len(vertex_morphs):
            return

        # add basis key
        ob.shape_key_add(name="Basis")

        # add anim data
        action = self.animation.get_action(ob.data.shape_keys)

        # add morph keys
        for i, target in enumerate(self.source.morph_targets):

            # from times to frames
            target.keys[:, 0] *= bpy.context.scene.render.fps

            # create morph targets
            shape_key = ob.shape_key_add(name="")
            shape_key.data.foreach_set("co", vertex_morphs[i].ravel())

            # create morph fcurves
            data_path = shape_key.path_from_id("value")
            fcurve = action.fcurves.new(data_path)

            # add fcurve keyframes
            fcurve.keyframe_points.add(len(target.keys))
            fcurve.keyframe_points.foreach_set("co", target.keys[:, :2].ravel())
            fcurve.update()

        # update frame range
        self.animation.update_frame_range(self.source.controller)

    def create_mesh_radius(self):
        ob = Empty(self).create_object()
        ob.parent = self.output
        ob.name = "Radius"
        ob.empty_display_size = 1.0
        ob.empty_display_type = 'SPHERE'
        ob.location = self.source.data.center
        ob.scale = [self.source.data.radius] * 3
        return ob

    def get_mesh_data(self):
        vertices = self.source.data.vertices
        normals = self.source.data.normals
        uv_sets = self.source.data.uv_sets
        vertex_colors = self.source.data.vertex_colors
        vertex_weights = self.source.vertex_weights()
        vertex_morphs = self.source.vertex_morphs()
        triangles = self.source.data.triangles

        if len(normals):
            # reconstruct as per-triangle layout
            normals = normals[triangles].reshape(-1, 3)

        if len(uv_sets):
            # convert OpenGL into Blender format
            uv_sets[..., 1] = 1 - uv_sets[..., 1]
            # reconstruct as per-triangle layout
            uv_sets = uv_sets[:, triangles].reshape(-1, triangles.size, 2)

        if len(vertex_colors):
            # reconstruct as per-triangle layout
            vertex_colors = vertex_colors[triangles].reshape(-1, 3)

        # remove doubles
        scale = decompose(self.matrix_world)[-1]
        indices, inverse = nif_utils.unique_rows(
            vertices * scale,
            *vertex_weights,
            *vertex_morphs,
            precision=self.importer.vertex_precision,
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
            triangles[target_faces] = new_vertex_indices.reshape(-1, 3)
        # '''

        return nif_utils.Namespace(
            triangles=triangles,
            vertices=vertices,
            normals=normals,
            uv_sets=uv_sets,
            vertex_colors=vertex_colors,
            vertex_weights=vertex_weights,
            vertex_morphs=vertex_morphs,
        )


class Material(SceneNode):
    __slots__ = ()

    def __init__(self, node):
        self.__dict__ = node.__dict__

    def create(self):
        properties = self.properties

        # Merge Duplicates
        props_hash = len(self.output.data.vertex_colors), *properties.values()
        try:
            bl_prop = self.importer.materials[props_hash]
        except KeyError:
            bl_prop = self.importer.materials[props_hash] = nif_shader.execute(self.output)
        else:
            # material already exists, reuse it
            self.output.data.materials.append(bl_prop.material)
            return

        # Alpha Property
        ni_alpha = properties.get(nif.NiAlphaProperty)
        if ni_alpha:
            self.create_alpha_property(bl_prop, ni_alpha)

        # Material Property
        ni_material = properties.get(nif.NiMaterialProperty)
        if ni_material:
            self.create_material_property(bl_prop, ni_material)

        # Stencil Property
        ni_stencil = properties.get(nif.NiStencilProperty)
        if ni_stencil:
            self.create_stencil_property(bl_prop, ni_stencil)

        # Texture Property
        ni_texture = properties.get(nif.NiTexturingProperty)
        if ni_texture:
            self.create_texturing_property(bl_prop, ni_texture)

        # Wireframe Property
        ni_wireframe = properties.get(nif.NiWireframeProperty)
        if ni_wireframe:
            self.create_wireframe_property(bl_prop, ni_wireframe)

    def create_alpha_property(self, bl_prop, ni_prop):
        """ TODO:
            src_blend_mode :
            dst_blend_mode :
            test_mode      :
            no_sorter      :
        """
        # Alpha Flags
        bl_prop.alpha_flags = ni_prop.flags
        # Alpha Threshold
        bl_prop.material.alpha_threshold = float(ni_prop.test_ref / 255)
        # Blending Method
        if ni_prop.alpha_blending:
            bl_prop.use_alpha_blend = True
        if ni_prop.alpha_testing:
            bl_prop.use_alpha_clip = True

    def create_material_property(self, bl_prop, ni_prop):
        # Material Name
        bl_prop.material.name = ni_prop.name
        # Material Flags
        bl_prop.material_flags = ni_prop.flags
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
        # Texture Flags
        bl_prop.texture_flags = ni_prop.flags
        # Texture Slots
        for name in nif.NiTexturingProperty.texture_keys:
            self.create_texturing_property_map(bl_prop, ni_prop, name)
        # Vertex Colors
        if self.output.data.vertex_colors:
            bl_prop.vertex_color.layer_name = self.output.data.vertex_colors[0].name
            bl_prop.create_link(bl_prop.vertex_color, bl_prop.shader, "Color", "Diffuse Color")
            bl_prop.create_link(bl_prop.vertex_color, bl_prop.shader, "Alpha", "Diffuse Alpha")

    def create_wireframe_property(self, bl_prop, ni_prop):
        if ni_prop.wireframe:
            self.output.display_type = "WIRE"

    def create_stencil_property(self, bl_prop, ni_prop):
        bl_prop.material.use_backface_culling = False
        bl_prop.material.show_transparent_back = True

    def create_texturing_property_map(self, bl_prop, ni_prop, name):
        try:
            bl_slot = getattr(bl_prop, name)
            ni_slot = getattr(ni_prop, name)
            # only supports slots with texture image attached
            image = self.create_image(ni_slot.source.filename)
        except AttributeError:
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

    def create_image(self, filename):
        abspath = self.resolve_texture_path(filename)

        if abspath.exists():
            image = bpy.data.images.load(str(abspath), check_existing=True)
        else:  # placeholder
            image = bpy.data.images.new(name=abspath.name, width=1, height=1)
            image.filepath = str(abspath)
            image.source = "FILE"

        return image

    @staticmethod
    def resolve_texture_path(relpath):
        # get the initial filepath
        path = pathlib.Path(relpath.lower())

        # discard "data files" prefix
        if path.parts[0] == "data files":
            path = path.relative_to("data files")

        # discard "textures" prefix
        if path.parts[0] == "textures":
            path = path.relative_to("textures")

        # evaluate final image path
        prefs = bpy.context.preferences.addons["io_scene_mw"].preferences
        for item in prefs.texture_paths:
            abspath = item.name / path
            if abspath.parent.exists():
                for suffix in {abspath.suffix, ".dds", ".tga", ".bmp"}:
                    abspath = abspath.with_suffix(suffix)
                    if abspath.exists():
                        return abspath

        return ("textures" / path)


class Animation(SceneNode):
    __slots__ = ()

    def __init__(self, node):
        self.__dict__ = node.__dict__

    def create(self):
        if not self.source.controller:
            return

        # get blender object
        bl_object = self.output.id_data

        # NiTextKeyExtraData
        self.create_text_keys(bl_object)
        # NiKeyframeController
        self.create_kf_controller(bl_object)
        # NiUVController
        self.create_uv_controller(bl_object)
        # NiVisController
        self.create_vis_controller(bl_object)

    # -- NiTextKeyExtraData --

    def create_text_keys(self, bl_object):
        text_data = self.source.extra_datas.find_type(nif.NiTextKeyExtraData)
        if text_data is None:
            return

        action = self.get_action(bl_object)

        # convert time to frame
        text_data.keys["f0"] *= bpy.context.scene.render.fps

        for time, text in text_data.keys:
            text = text.replace("\r\n", "; ")
            m = action.pose_markers.new(text)
            m.frame = math.ceil(time)  # TODO warn if not on an integer frame

    # -- NiKeyframeController --

    def create_kf_controller(self, bl_object):
        controller = self.source.controllers.find_type(nif.NiKeyframeController)
        if controller is None:
            return

        # get animation action
        action = self.get_action(bl_object)
        # get offset from pose
        posed_offset = self.get_posed_offset(bl_object)

        # translation keys
        self.create_kf_translations(controller, action, posed_offset)
        # rotation keys
        self.create_kf_rotations(controller, action, posed_offset)
        # scale keys
        self.create_kf_scales(controller, action, posed_offset)

        self.update_frame_range(controller)

    def create_kf_translations(self, controller, action, posed_offset):
        data = controller.data.translations
        if len(data.keys) == 0:
            return

        # get keys times/values
        times, values = data.times, data.values

        # convert time to frame
        times *= bpy.context.scene.render.fps

        # convert to posed space
        if hasattr(self, "matrix_posed"):
            values[:] = values @ posed_offset[:3, :3].T + posed_offset[:3, 3]

        # bezier tangent handles
        handles = data.get_tangent_handles()

        # get blender data path
        data_path = self.output.path_from_id("location")

        # build blender fcurves
        for i in range(3):
            fc = action.fcurves.new(data_path, index=i, action_group=self.bone_name)
            fc.keyframe_points.add(len(data.keys))
            fc.keyframe_points.foreach_set("co", data.keys[:, (0, i+1)].ravel())
            if data.interpolation.name != 'LIN_KEY':
                self.set_handle_types(fc.keyframe_points, 'FREE')
                fc.keyframe_points.foreach_set("handle_left", handles[0, i].ravel())
                fc.keyframe_points.foreach_set("handle_right", handles[1, i].ravel())
            fc.update()

    def create_kf_rotations(self, controller, action, posed_offset):
        if controller.data.rotations.euler_data:
            if isinstance(self.output, bpy.types.PoseBone):
                print("[INFO] Euler animations on bones are not currently supported.")
                controller.data.rotations.convert_to_quaternions()
            else:
                self.output.rotation_mode = controller.data.rotations.euler_axis_order.name
                self.create_kf_euler_rotations(controller, action, posed_offset)
                return
        self.create_kf_quaternion_rotations(controller, action, posed_offset)

    def create_kf_euler_rotations(self, controller, action, posed_offset):
        for i, data in enumerate(controller.data.rotations.euler_data):
            if len(data.keys) == 0:
                continue

            # convert time to frame
            data.keys[:, 0] *= bpy.context.scene.render.fps

            # get blender data path
            data_path = self.output.path_from_id("rotation_euler")

            # build blender fcurves
            fc = action.fcurves.new(data_path, index=i, action_group=self.output.name)
            fc.keyframe_points.add(len(data.keys))
            fc.keyframe_points.foreach_set("co", data.keys.ravel())
            fc.update()

    def create_kf_quaternion_rotations(self, controller, action, posed_offset):
        keys = controller.data.rotations.keys
        if len(keys) == 0:
            return

        # get keys times/values
        times, values = keys[:, 0], keys[:, 1:5]

        # convert time to frame
        times *= bpy.context.scene.render.fps

        if hasattr(self, "matrix_posed"):
            # apply axis correction
            axis_fix = nif_utils.quaternion_from_matrix(self.axis_correction)
            nif_utils.quaternion_mul(values, axis_fix, out=values)
            # convert to pose space
            to_posed = nif_utils.quaternion_from_matrix(posed_offset)
            nif_utils.quaternion_mul(to_posed, values, out=values)

        # get blender data path
        data_path = self.output.path_from_id("rotation_quaternion")

        # build blender fcurves
        for i in range(4):
            fc = action.fcurves.new(data_path, index=i, action_group=self.output.name)
            fc.keyframe_points.add(len(keys))
            fc.keyframe_points.foreach_set("co", keys[:, (0, i+1)].ravel())
            fc.update()

    def create_kf_scales(self, controller, action, posed_offset):
        keys = controller.data.scales.keys
        if len(keys) == 0:
            return

        # convert time to frame
        keys[:, 0] *= bpy.context.scene.render.fps

        # get blender data path
        data_path = self.output.path_from_id("scale")

        # build blender fcurves
        for i in range(3):
            fc = action.fcurves.new(data_path, index=i, action_group=self.output.name)
            fc.keyframe_points.add(len(keys))
            fc.keyframe_points.foreach_set("co", keys.ravel())
            fc.update()

    # -- NiVisController --

    def create_vis_controller(self, bl_object):
        controller = self.source.controllers.find_type(nif.NiVisController)
        if controller is None:
            return

        data = controller.data
        if (data is None) or len(data.keys) == 0:
            return

        keys = np.empty((len(data.keys), 2), dtype=np.float32)

        # convert time to frame
        keys[:, 0] = data.times * bpy.context.scene.render.fps

        # invert appculled flag
        keys[:, 1] = 1 - data.values

        # get animations action
        action = self.get_action(bl_object)

        # get blender data path
        data_path = self.output.path_from_id("hide_viewport")

        # build blender fcurves
        fc = action.fcurves.new(data_path, index=0, action_group=self.output.name)
        fc.keyframe_points.add(len(keys))
        fc.keyframe_points.foreach_set("co", keys.ravel())
        fc.update()


    # -- NiUVController --

    def create_uv_controller(self, bl_object):
        controller = self.source.controllers.find_type(nif.NiUVController)
        if controller is None:
            return

        data = controller.data
        if data is None:
            return

        # get blender property
        bl_prop = self.output.active_material.mw

        # get animation action
        action = self.get_action(bl_prop.texture_group.node_tree)
        if len(action.fcurves):
            return  # duplicate

        # get the texture slot
        uv_name = self.output.data.uv_layers[controller.texture_set].name
        bl_slot = next(s for s in bl_prop.texture_slots if s.layer == uv_name)

        # create offset fcurves
        data_path = bl_slot.mapping_node.inputs["Location"].path_from_id('default_value')
        for i, keys in enumerate((data.offset_u.keys, data.offset_v.keys)):
            if len(keys):
                # convert from times to frames
                keys[:, 0] *= bpy.context.scene.render.fps
                # convert to blender uv layout
                keys[:, 1] = i - keys[:, 1]
                # build blender fcurves
                fc = action.fcurves.new(data_path, index=i, action_group=uv_name)
                fc.keyframe_points.add(len(keys))
                fc.keyframe_points.foreach_set("co", keys[:, :2].ravel())
                fc.update()

        # create tiling fcurves
        data_path = bl_slot.mapping_node.inputs["Scale"].path_from_id('default_value')
        for i, keys in enumerate((data.tiling_u.keys, data.tiling_v.keys)):
            if len(keys):
                # convert from times to frames
                keys[:, 0] *= bpy.context.scene.render.fps
                # build blender fcurves
                fc = action.fcurves.new(data_path, index=i, action_group=uv_name)
                fc.keyframe_points.add(len(keys))
                fc.keyframe_points.foreach_set("co", keys[:, :2].ravel())
                fc.update()

        self.update_frame_range(controller)

    # -- NiMaterialColorController --

    def create_color_controller(self, bl_prop, ni_prop):
        controller = ni_prop.controllers.find_type(nif.NiMaterialColorController)
        if controller is None:
            return

        data = controller.data
        if data is None:
            return

        # keyframe points array
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

        # convert time to frame
        keys[:, 0] *= bpy.context.scene.render.fps

        # create blender action
        action = self.get_action(bl_prop.material.node_tree)

        # build blender fcurves
        for i in range(3):
            fc = action.fcurves.new(data_path, index=i, action_group=bl_prop.material.name)
            fc.keyframe_points.add(len(keys))
            fc.keyframe_points.foreach_set("co", keys[:, (0, i+1)].ravel())
            fc.update()

        self.update_frame_range(controller)

    # -- NiAlphaController --

    def create_alpha_controller(self, bl_prop, ni_prop):
        controller = ni_prop.controllers.find_type(nif.NiAlphaController)
        if controller is None:
            return

        data = controller.data
        if data is None:
            return

        # keyframe points array
        keys = controller.data.keys
        if len(keys) == 0:
            return

        # convert time to frame
        keys[:, 0] *= bpy.context.scene.render.fps

        # create blender action
        action = self.get_action(bl_prop.material.node_tree)

        # get blender data path
        data_path = bl_prop.opacity_input.path_from_id("default_value")

        # build blender fcurves
        fc = action.fcurves.new(data_path, index=0, action_group=bl_prop.material.name)
        fc.keyframe_points.add(len(keys))
        fc.keyframe_points.foreach_set("co", keys[:, :2].ravel())
        fc.update()

        self.update_frame_range(controller)

    # -- utility functions --

    def get_posed_offset(self, bl_object):
        try:
            world_offset = self.parent.matrix_posed @ self.axis_correction_inverse
        except AttributeError:
            world_offset = self.parent.matrix_world if self.parent else ID44
        try:
            posed_offset = bl_object.convert_space(pose_bone=self.output, matrix=ID44.T, from_space="WORLD", to_space="LOCAL")
        except TypeError:
            posed_offset = bl_object.convert_space(pose_bone=None, matrix=ID44.T, from_space="WORLD", to_space="LOCAL")
        return np.matmul(posed_offset, world_offset)

    @staticmethod
    def get_action(bl_object):
        try:
            action = bl_object.animation_data.action
        except AttributeError:
            action = bl_object.animation_data_create().action = bpy.data.actions.new(f"{bl_object.name}Action")
        return action

    @staticmethod
    def set_handle_types(keyframe_points, handle_type):
        for kp in keyframe_points:
            kp.handle_left_type = kp.handle_right_type = handle_type

    @staticmethod
    def update_frame_range(controller):
        scene = bpy.context.scene
        frame_end = math.ceil(controller.stop_time * scene.render.fps)
        scene.frame_end = scene.frame_preview_end = max(scene.frame_end, frame_end)