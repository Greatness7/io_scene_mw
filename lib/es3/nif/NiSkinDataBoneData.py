from __future__ import annotations

from ..utils.math import compose, decompose_uniform, ID33, la, np, ZERO3, zeros, get_exact_center_radius
from ..utils.typing import *

from .NiBinaryStream import NiBinaryStream
from .NiObject import NiObject


_dtype = np.dtype("<H, <f")


class NiSkinDataBoneData(NiObject):  # TODO Not NiObject
    rotation: NiMatrix3 = ID33
    translation: NiPoint3 = ZERO3
    scale: float32 = 1.0
    center: NiPoint3 = ZERO3
    radius: float32 = 0.0
    vertex_weights: ndarray = zeros(0, dtype=_dtype)

    def load(self, stream: NiBinaryStream) -> None:
        self.rotation = stream.read_floats(3, 3)
        self.translation = stream.read_floats(3)
        self.scale = stream.read_float()
        self.center = stream.read_floats(3)
        self.radius = stream.read_float()
        num_weights = stream.read_ushort()
        if num_weights:
            self.vertex_weights = stream.read_array(num_weights, _dtype)

    def save(self, stream: NiBinaryStream) -> None:
        stream.write_floats(self.rotation)
        stream.write_floats(self.translation)
        stream.write_float(self.scale)
        stream.write_floats(self.center)
        stream.write_float(self.radius)
        stream.write_ushort(len(self.vertex_weights))
        stream.write_array(self.vertex_weights, _dtype)

    def apply_scale(self, scale: float) -> None:
        self.translation *= scale
        self.center *= scale
        self.radius *= scale

    def update_center_radius(self, vertices: ndarray, exact: bool = False) -> None:
        if len(vertices) == 0:
            self.center[:] = self.radius = 0
        else:
            if not exact:
                center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
                radius = float(la.norm(center - vertices, axis=1).max())  # pyright: ignore[reportUnknownMemberType]
            else:
                center, radius = get_exact_center_radius(
                    vertices.astype(dtype=np.float64, order="C", copy=False)  # pyright: ignore[reportUnknownMemberType]
                )

            self.center = self.scale * (self.rotation @ center) + self.translation
            self.radius = self.scale * radius

    @property
    def matrix(self) -> ndarray:
        return compose(self.translation, self.rotation, self.scale)

    @matrix.setter
    def matrix(self, value: ndarray):
        self.translation, self.rotation, self.scale = decompose_uniform(value)
