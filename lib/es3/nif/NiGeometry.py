from __future__ import annotations

from es3.utils.flags import bool_property
from es3.utils.math import decompose_uniform, zeros
from .NiAVObject import NiAVObject


class NiGeometry(NiAVObject):
    data: NiGeometryData | None = None
    skin: NiSkinInstance | None = None

    # flags access
    compress_vertices = bool_property(mask=0x0008)
    compress_normals = bool_property(mask=0x0010)
    compress_uv_sets = bool_property(mask=0x0020)
    shadow = bool_property(mask=0x0040)

    _refs = (*NiAVObject._refs, "data", "skin")

    def load(self, stream):
        super().load(stream)
        self.data = stream.read_link()
        self.skin = stream.read_link()

    def save(self, stream):
        super().save(stream)
        stream.write_link(self.data)
        stream.write_link(self.skin)

    @property
    def bone_influences(self):
        try:
            skin = self.skin.root and self.skin
            assert len(skin.bones) == len(skin.data.bone_data)
        except (AttributeError, AssertionError):
            return ()
        return tuple(zip(skin.bones, skin.data.bone_data))

    @property
    def morph_targets(self):
        try:
            basis, *targets = self.controller.data.targets
            assert len(basis.vertices) == len(self.data.vertices)
        except (AttributeError, AssertionError, ValueError):
            return ()
        return tuple(targets)

    def vertex_weights(self):
        bone_influences = self.bone_influences
        vertex_weights = zeros(len(bone_influences), len(self.data.vertices))

        for i, (_, bone_data) in enumerate(bone_influences):
            indices = bone_data.vertex_weights["f0"]
            weights = bone_data.vertex_weights["f1"]
            vertex_weights[i, indices] = weights

        return vertex_weights

    def vertex_morphs(self):
        morph_targets = self.morph_targets
        vertex_morphs = zeros(len(morph_targets), len(self.data.vertices), 3)

        for i, target in enumerate(morph_targets):
            vertex_morphs[i] = self.data.vertices + target.vertices  # TODO not always relative!

        return vertex_morphs

    def apply_skin(self, keep_skin=False):
        data = self.data
        skin = self.skin
        if not keep_skin:
            self.skin = None

        deformed_verts = zeros(*data.vertices.shape)
        deformed_norms = zeros(*data.normals.shape)

        root_to_skin = skin.data.matrix
        for bone, bone_data in zip(skin.bones, skin.data.bone_data):
            skin_to_bone = bone_data.matrix

            bone_matrix = bone.matrix_relative_to(skin.root)
            bind_matrix = root_to_skin @ bone_matrix @ skin_to_bone

            location, rotation, scale = decompose_uniform(bind_matrix)

            # indices and weights
            i = bone_data.vertex_weights["f0"]
            w = bone_data.vertex_weights["f1"][:, None]

            if len(deformed_verts):
                deformed_verts[i] += w * (data.vertices[i] @ rotation.T * scale + location.T)
            if len(deformed_norms):
                deformed_norms[i] += w * (data.normals[i] @ rotation.T)

        data.vertices = deformed_verts
        data.normals = deformed_norms  # TODO: normalize?


if __name__ == "__main__":
    from es3.nif import NiGeometryData, NiSkinInstance
    from es3.utils.typing import *
