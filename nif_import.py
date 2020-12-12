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

    importer = Importer(filepath, config)
    importer.execute()

    time = timeit.default_timer() - time
    print(f"Import Done: {time:.4f} seconds")

    return {"FINISHED"}


class Importer:
    vertex_precision = 0.001
    attach_keyframe_data = False
    discard_root_transforms = True
    use_existing_materials = False
    ignore_collision_nodes = False
    ignore_custom_normals = False
    ignore_animations = False

    def __init__(self, filepath, config):
        vars(self).update(config)
        self.nodes = {}
        self.materials = {}
        self.history = collections.defaultdict(set)
        self.armatures = collections.defaultdict(set)
        self.colliders = collections.defaultdict(set)
        self.active_collection = bpy.context.view_layer.active_layer_collection.collection
        self.filepath = pathlib.Path(filepath)

    def execute(self):
        data = nif.NiStream()
        data.load(self.filepath)
        data.merge_properties()

        # fix transforms
        if self.discard_root_transforms:
            data.root.matrix = ID44

        # attach kf file
        if self.attach_keyframe_data:
            self.import_keyframe_data(data)

        # apply settings
        data.apply_scale(self.scale_correction)

        # give root name
        if data.root.name == "":
            data.root.name = self.filepath.name

        # resolve heirarchy
        roots = self.resolve_nodes(data.roots)

        # resolve armatures
        if any(self.armatures):
            self.resolve_armatures()
            self.correct_rest_positions()
            self.apply_axis_corrections()
            self.correct_bone_parenting()

        # discard frame pos
        frame_current = bpy.context.scene.frame_current
        bpy.context.scene.frame_set(0)

        # create bl objects
        for node, cls in self.nodes.items():
            if node.output is None:
                cls(node).create()

        # unmute animations
        for node in map(self.get, self.armatures):
            node.animation.set_mute(False)

        # restore frame pos
        bpy.context.scene.frame_current = frame_current

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

        # sort roots via heirarchy
        roots = list(map(self.get, self.armatures))
        roots.sort(key=lambda r: len([*r.parents]))

        # select the top-most root
        root = roots[0].source
        bones = self.armatures[root]

        # collect all orphan bones
        bones.update(orphan_bones)

        # collect all others bones
        for other_root in self.armatures.keys() - {root}:
            other_bones = self.armatures.pop(other_root)
            bones.add(other_root)
            bones.update(other_bones)

        # only descendants of root
        root_node = self.get(root)
        bones -= {node.source for node in (root_node, *root_node.parents)}

        # bail if no bones present
        if len(bones) == 0:
            self.armatures.clear()
            return

        # consider any descendants which are animated to be bones
        # this is usually desired, and to not do so would mean we
        # have to fix the animations of any node who's transforms
        # are modified by a parent bone receiving axis correction
        for root_bone in filter(bones.__contains__, root.children):
            for child in root_bone.descendants():
                if isinstance(child, nif.NiNode):
                    if child.controllers.find_type(nif.NiKeyframeController):
                        bones.add(child)

        # validate all bone chains
        for node in list(map(self.get, bones)):
            for parent in node.parents:
                source = parent.source
                if (source is root) or (source in bones):
                    break
                bones.add(source)

        # order bones by heirarchy
        self.armatures[root] = dict.fromkeys(node.source for node in self.nodes if node.source in bones).keys()

        # preserve bone pose matrices
        for node in self.iter_bones(root_node):
            node.matrix_posed = node.matrix_world

        # send all bones to rest pose
        root.apply_bone_bind_poses()
        root.apply_skins(keep_skins=True)

        # apply updated rest matrices
        for node in self.iter_bones(root_node):
            node.matrix_local = node.source.matrix

        # specify node as Armature
        self.nodes[root_node] = Armature

    def correct_rest_positions(self):
        """Correct the rest pose of the root bone.

        The rest pose of vanilla assets often feature inconvenient transforms.
        This is not an issue in-game since you would only ever see actors with
        their animations applied. When working in Blender however, a sane rest
        pose will make the lives of artists much easier.

        This function replaces the root bone's rest matrix with an edited copy
        of its posed matrix. This edited copy is identical to pose matrix with
        regards to location and scale, but has had its rotation about all axes
        aligned to the nearest 90 degree angle.
        """
        if not self.armatures:
            return

        root = self.get_armature_node()
        root_bone = next(self.iter_bones(root))

        # calculate corrected transformation matrix
        l, r, s = decompose(root_bone.matrix_posed)
        r = nif_utils.snap_rotation(r)
        corrected_matrix = compose(l, r, s)

        # only do corrections if they are necessary
        if np.allclose(root_bone.matrix_world, corrected_matrix, rtol=0, atol=1e-6):
            return

        # correct the rest matrix of skinned meshes
        inverse = la.inv(root_bone.matrix_world)
        for node in self.get_skinned_meshes():
            if root_bone not in node.parents:
                node.matrix_world = corrected_matrix @ (inverse @ node.matrix_world)

        # correct the rest matrix of the root bone
        root_bone.matrix_world = corrected_matrix

    def apply_axis_corrections(self):
        if not self.armatures:
            return

        root = self.get_armature_node()
        bones = list(self.iter_bones(root))

        # apply bone axis corrections
        for node in reversed(bones):
            node.matrix_posed = node.matrix_posed @ node.axis_correction
            node.matrix_local = node.matrix_local @ node.axis_correction
            for child in node.children:
                child.matrix_local = node.axis_correction_inverse @ child.matrix_local

        # apply anim axis corrections
        root_inverse = la.inv(root.matrix_world)
        for node in bones:
            kf_controller = node.source.controllers.find_type(nif.NiKeyframeController)
            if not (kf_controller and kf_controller.data):
                continue

            try:
                parent_matrix = node.parent.matrix_posed
                parent_matrix_uncorrected = parent_matrix @ node.parent.axis_correction_inverse
            except AttributeError:  # parent is not bone
                parent_matrix = node.parent.matrix_world if node.parent else ID44
                parent_matrix_uncorrected = parent_matrix

            matrix = parent_matrix @ node.matrix_local
            matrix_relative_to_root = root_inverse @ matrix

            posed_offset = la.solve(matrix_relative_to_root, root_inverse)
            posed_offset = posed_offset @ parent_matrix_uncorrected

            values = kf_controller.data.translations.values
            if len(values):
                # convert to pose space
                values[:] = values @ posed_offset[:3, :3].T + posed_offset[:3, 3]

            values = kf_controller.data.rotations.values
            if len(values):
                # apply axis correction
                axis_fix = nif_utils.quaternion_from_matrix(node.axis_correction)
                values[:] = nif_utils.quaternion_mul(values, axis_fix)
                # convert to pose space
                to_posed = nif_utils.quaternion_from_matrix(posed_offset)
                values[:] = nif_utils.quaternion_mul(to_posed, values)

    def correct_bone_parenting(self):
        """Set the parent of skinned meshes to the armature responsible for deforming them.

        This must be done as using both skinning and bone-parenting at the same time does not
        behave correctly in Blender.

        Usually this occurs when a file contains nested armatures.
        See `Tri Hand01` in the vanilla `r/skeleton.nif` model for example.
        """
        if not self.armatures:
            return

        armature = self.get_armature_node()
        for node in self.get_skinned_meshes():
            if node.parent != armature:
                matrix_world = node.matrix_world
                node.parent = armature
                node.matrix_world = matrix_world

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
        if (name == "bip01") or (name == "root bone"):
            self.armatures[node.source].update()
        elif ("bip01" in name) or name.endswith(" bone"):
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
        if self.ignore_collision_nodes:
            return False

        self.nodes[node] = Empty
        self.colliders[node.source].update(node.source.descendants())
        return True

    # -------
    # UTILITY
    # -------

    def get(self, source):
        return next(iter(self.history[source]))

    def iter_bones(self, root):
        yield from map(self.get, self.armatures[root.source])

    def get_root_output(self, roots):
        return roots[0].output.id_data if roots else None

    def get_armature_node(self):
        return self.get(*self.armatures)

    def get_skinned_meshes(self):
        for node in self.nodes:
            if getattr(node.source, "skin", None):
                yield node

    def import_keyframe_data(self, data):
        kf_path = self.filepath.with_suffix(".kf")
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
        if "Bip01" in self.name:
            return biped_axis_correction
        return other_axis_correction

    @property
    def axis_correction_inverse(self):
        if "Bip01" in self.name:
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

        self.animation.create()

        return self.output

    def create_object(self, bl_data=None):
        bl_object = bpy.data.objects.new(self.name, bl_data)
        self.importer.active_collection.objects.link(bl_object)
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
        for node in self.importer.iter_bones(self):
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
            pose_bone.matrix = (root_inverse @ node.matrix_posed).T
            # TODO try not to call scene update
            bpy.context.view_layer.depsgraph.update()
            # create animations, preserve poses
            node.animation.create()
            node.animation.set_mute(True)

        return bl_object


class Mesh(SceneNode):
    __slots__ = ()

    def __init__(self, node):
        self.__dict__ = node.__dict__

    def create(self):
        bl_data = bpy.data.meshes.new(self.name)
        bl_object = Empty(self).create(bl_data)
        if len(self.source.data.vertices) == 0:
            return bl_object

        ni_data = self.get_mesh_data()

        self.create_vertices(bl_object, ni_data.vertices)
        self.create_triangles(bl_object, ni_data.triangles)

        self.create_normals(bl_object, ni_data.normals)
        self.create_vertex_colors(bl_object, ni_data.vertex_colors)
        self.create_uv_sets(bl_object, ni_data.uv_sets)

        self.create_vertex_weights(bl_object, ni_data.vertex_weights)
        self.create_vertex_morphs(bl_object, ni_data.vertex_morphs)

        bl_data.validate(verbose=False, clean_customdata=False)

        try:
            self.output.display_type = self.parent.output.display_type
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
            if not self.importer.ignore_custom_normals:
                ob.data.use_auto_smooth = True
                ob.data.normals_split_custom_set(normals)
            # ob.data.edges.foreach_set("use_edge_sharp", [False] * len(ob.data.edges))

    def create_uv_sets(self, ob, uv_sets):
        for i, uv in enumerate(uv_sets[:8]):  # max 8 uv sets (blender limitation)
            ob.data.uv_layers.new()
            ob.data.uv_layers[i].data.foreach_set("uv", uv.ravel())

    def create_vertex_colors(self, ob, vertex_colors):
        if len(vertex_colors):
            vc = ob.data.vertex_colors.new()
            vc.data.foreach_set("color", vertex_colors.ravel())

    def create_vertex_weights(self, ob, vertex_weights):
        if not len(vertex_weights):
            return

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
                vg.add([j], weights[j], "ADD")

    def create_vertex_morphs(self, ob, vertex_morphs):
        if not len(vertex_morphs):
            return

        animation = self.animation

        # add basis key
        ob.shape_key_add(name="Basis")

        # add anim data
        action = animation.get_action(ob.data.shape_keys)

        # add morph keys
        for i, target in enumerate(self.source.morph_targets):

            # from times to frames
            target.keys[:, 0] *= bpy.context.scene.render.fps

            # create morph targets
            shape_key = ob.shape_key_add(name="")
            shape_key.data.foreach_set("co", vertex_morphs[i].ravel())

            # create morph fcurves
            data_path = shape_key.path_from_id("value")
            fc = action.fcurves.new(data_path)

            # add fcurve keyframes
            fc.keyframe_points.add(len(target.keys))
            fc.keyframe_points.foreach_set("co", target.keys[:, :2].ravel())
            animation.create_interpolation_data(target, fc)
            fc.update()

        # update frame range
        animation.update_frame_range(self.source.controller)

    def get_mesh_data(self):
        vertices = self.source.data.vertices
        normals = self.source.data.normals
        uv_sets = self.source.data.uv_sets.copy()
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
        if len(properties) == 0:
            return

        ni_alpha = properties.get(nif.NiAlphaProperty)
        ni_material = properties.get(nif.NiMaterialProperty)
        ni_stencil = properties.get(nif.NiStencilProperty)
        ni_texture = properties.get(nif.NiTexturingProperty)
        ni_wireframe = properties.get(nif.NiWireframeProperty)

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
        try:
            bl_prop = self.importer.materials[props_hash]
        except KeyError:
            bl_prop = self.importer.materials[props_hash] = nif_shader.execute(self.output)
        else:
            # material already exists, reuse it
            self.output.data.materials.append(bl_prop.material)
            return
        finally:
            if self.importer.use_existing_materials:
                bl_prop.material.name = name

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
        bl_prop.material.alpha_threshold = float(ni_prop.test_ref / 255)
        # Blending Method
        if ni_prop.alpha_blending:
            bl_prop.use_alpha_blend = True
        if ni_prop.alpha_testing:
            bl_prop.use_alpha_clip = True

    def create_material_property(self, bl_prop, ni_prop):
        # # Material Name
        # bl_prop.material.name = ni_prop.name
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
        if not self.importer.use_existing_materials:
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
        if not self.importer.use_existing_materials:
            return

        use_vertex_colors = bool(len(self.source.data.vertex_colors))
        use_alpha_blend = getattr(ni_alpha, "alpha_blending", False)
        use_alpha_clip = getattr(ni_alpha, "alpha_testing", False)

        base_name, index = name, 0
        while True:
            try:
                bl_prop = bpy.data.materials[name].mw.validate()
            except (LookupError, TypeError):
                break
            if (
                bl_prop.use_vertex_colors == use_vertex_colors
                and bl_prop.use_alpha_blend == use_alpha_blend
                and bl_prop.use_alpha_clip == use_alpha_clip
            ):
                self.output.data.materials.append(bl_prop.material)
                return True
            index += 1
            name = f"{base_name}.{index:03}"

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

        # potential file extensions
        suffixes = {path.suffix, ".dds", ".tga", ".bmp"}

        # evaluate final image path
        prefs = bpy.context.preferences.addons["io_scene_mw"].preferences
        for item in prefs.texture_paths:
            abspath = item.name / path
            if abspath.parent.exists():
                for suffix in suffixes:
                    abspath = abspath.with_suffix(suffix)
                    if abspath.exists():
                        return abspath

        return ("textures" / path)


class Animation(SceneNode):
    __slots__ = ()

    def __init__(self, node):
        self.__dict__ = node.__dict__

    def create(self):
        if self.importer.ignore_animations:
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

        # convert time to frame
        text_data.times = np.ceil(text_data.times * bpy.context.scene.render.fps)

        for frame, text in text_data.keys.tolist():
            for name in filter(None, text.splitlines()):
                assert len(name) < 64, f"Marker exceeds character limit ({name})"
                action.pose_markers.new(name).frame = frame

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

        # get keys times/values
        times, values = data.times, data.values

        # convert time to frame
        times *= bpy.context.scene.render.fps

        # get blender data path
        data_path = self.output.path_from_id("location")

        # build blender fcurves
        for i in range(3):
            fc = action.fcurves.new(data_path, index=i, action_group=self.bone_name)
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

            # convert time to frame
            data.keys[:, 0] *= bpy.context.scene.render.fps

            # get blender data path
            data_path = self.output.path_from_id("rotation_euler")

            # build blender fcurves
            fc = action.fcurves.new(data_path, index=i, action_group=self.output.name)
            fc.keyframe_points.add(len(data.keys))
            fc.keyframe_points.foreach_set("co", data.keys[:, :2].ravel())
            self.create_interpolation_data(data, fc)
            fc.update()

    def create_quaternion_rotations(self, controller, action):
        data = controller.data.rotations
        if len(data.keys) == 0:
            return

        # get keys times/values
        times, values = data.times, data.values

        # convert time to frame
        times *= bpy.context.scene.render.fps

        # get blender data path
        data_path = self.output.path_from_id("rotation_quaternion")

        # build blender fcurves
        for i in range(4):
            fc = action.fcurves.new(data_path, index=i, action_group=self.output.name)
            fc.keyframe_points.add(len(data.keys))
            fc.keyframe_points.foreach_set("co", data.keys[:, (0, i+1)].ravel())
            fc.update()

    def create_scales(self, controller, action):
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
            fc.keyframe_points.foreach_set("co", keys[:, :2].ravel())
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

        # convert time to frame
        keys[:, 0] = data.times * bpy.context.scene.render.fps

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
        fc = action.fcurves.new(data_path, index=0, action_group=self.output.name)
        fc.keyframe_points.add(len(keys))
        fc.keyframe_points.foreach_set("co", keys.ravel())
        fc.update()

    def create_uv_controller(self, controller):
        if self.importer.ignore_animations:
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
            print(f"Warning: skipping NiUVController due to invalid texture set")
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
                # convert from times to frames
                uv_data.keys[:, 0] *= bpy.context.scene.render.fps
                # build blender fcurves
                fc = action.fcurves.new(data_path, index=i, action_group=uv_name)
                fc.keyframe_points.add(len(uv_data.keys))
                fc.keyframe_points.foreach_set("co", uv_data.keys[:, :2].ravel())
                self.create_interpolation_data(uv_data, fc)
                fc.update()

        self.update_frame_range(controller)

    def create_color_controller(self, bl_prop, ni_prop):
        if self.importer.ignore_animations:
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

        # convert time to frame
        keys[:, 0] *= bpy.context.scene.render.fps

        # create blender action
        action = self.get_action(bl_prop.material.node_tree)

        # build blender fcurves
        for i in range(3):
            fc = action.fcurves.new(data_path, index=i, action_group=bl_prop.material.name)
            fc.keyframe_points.add(len(keys))
            fc.keyframe_points.foreach_set("co", keys[:, (0, i+1)].ravel())
            self.create_interpolation_data(data, fc)
            fc.update()

        self.update_frame_range(controller)

    def create_alpha_controller(self, bl_prop, ni_prop):
        if self.importer.ignore_animations:
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
        self.create_interpolation_data(data, fc)
        fc.update()

        self.update_frame_range(controller)

    @staticmethod
    def get_action(bl_object):
        try:
            action = bl_object.animation_data.action
        except AttributeError:
            action = bl_object.animation_data_create().action = bpy.data.actions.new(f"{bl_object.name}Action")
        return action

    @staticmethod
    def create_interpolation_data(ni_data, fcurves, axis=...):
        if ni_data.interpolation.name  == 'LIN_KEY':
            for kp in fcurves.keyframe_points:
                kp.interpolation = 'LINEAR'
        else:
            handles = ni_data.get_tangent_handles()
            fcurves.keyframe_points.foreach_set("handle_left", handles[0, axis].ravel())
            fcurves.keyframe_points.foreach_set("handle_right", handles[1, axis].ravel())
            for kp in fcurves.keyframe_points:
                kp.handle_left_type = kp.handle_right_type = 'FREE'

    @staticmethod
    def update_frame_range(controller):
        scene = bpy.context.scene
        frame_end = np.ceil(controller.stop_time * scene.render.fps)
        scene.frame_end = scene.frame_preview_end = max(scene.frame_end, frame_end)

    def set_mute(self, state, fcurves=None):
        if self.importer.ignore_animations:
            return

        if fcurves is None:
            try:
                fcurves = self.output.id_data.animation_data.action.fcurves
            except AttributeError:
                return
        fcurves.foreach_set("mute", [state] * len(fcurves))
