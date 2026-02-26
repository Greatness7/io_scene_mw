from __future__ import annotations

from es3.utils.math import la, ZERO3, zeros, get_exact_center_radius
from .NiObject import NiObject


class NiGeometryData(NiObject):
    vertices: ndarray = zeros(0, 3)
    normals: ndarray = zeros(0, 3)
    center: NiPoint3 = ZERO3
    radius: float32 = 0.0
    vertex_colors: ndarray = zeros(0, 4)
    uv_sets: ndarray = zeros(0, 0, 2)

    def load(self, stream):
        num_vertices = stream.read_ushort()
        has_vertices = stream.read_bool()
        if has_vertices:
            self.vertices = stream.read_floats(num_vertices, 3)
        has_normals = stream.read_bool()
        if has_normals:
            self.normals = stream.read_floats(num_vertices, 3)
        self.center = stream.read_floats(3)
        self.radius = stream.read_float()
        has_vertex_colors = stream.read_bool()
        if has_vertex_colors:
            self.vertex_colors = stream.read_floats(num_vertices, 4)
        num_uv_sets = stream.read_ushort()
        has_uv_sets = stream.read_bool()
        if has_uv_sets:
            self.uv_sets = stream.read_floats(num_uv_sets, num_vertices, 2)

    def save(self, stream):
        num_vertices = len(self.vertices)
        stream.write_ushort(num_vertices)
        stream.write_bool(num_vertices)
        if num_vertices:
            stream.write_floats(self.vertices)
        num_normals = len(self.normals)
        stream.write_bool(num_normals)
        if num_normals:
            stream.write_floats(self.normals)
        stream.write_floats(self.center)
        stream.write_float(self.radius)
        num_vertex_colors = len(self.vertex_colors)
        stream.write_bool(num_vertex_colors)
        if num_vertex_colors:
            stream.write_floats(self.vertex_colors)
        num_uv_sets = len(self.uv_sets)
        stream.write_ushort(num_uv_sets)
        stream.write_bool(num_uv_sets)
        if num_uv_sets:
            stream.write_floats(self.uv_sets)

    def apply_scale(self, scale):
        self.vertices *= scale
        self.center *= scale
        self.radius *= scale

    def update_center_radius(self, exact=False):
        import numpy as np

        if len(self.vertices) == 0:
            self.center[:] = self.radius = 0
        else:
            vertices = self.vertices

            if not exact:
                center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
                radius = float(la.norm(center - vertices, axis=1).max())
            else:
                center, radius = get_exact_center_radius(
                    vertices.astype(dtype=np.float64, order="C", copy=False)
                )

            self.center = center
            self.radius = radius


if __name__ == "__main__":
    from es3.utils.typing import *
