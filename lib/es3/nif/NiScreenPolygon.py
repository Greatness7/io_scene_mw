from __future__ import annotations

from es3.utils.math import zeros
from .NiObject import NiObject


class NiScreenPolygon(NiObject):
    vertices: ndarray = zeros(0, 3)
    uv_coords: ndarray = zeros(0, 2)
    vertex_colors: ndarray = zeros(0, 4)
    property_states: ndarray = zeros(0, dtype="<i")  # TODO NiPropertyState

    def load(self, stream):
        num_vertices = stream.read_ushort()
        self.vertices = stream.read_floats(num_vertices, 3)
        has_uv_coords = stream.read_bool()
        if has_uv_coords:
            self.uv_coords = stream.read_floats(num_vertices, 2)
        has_vertex_colors = stream.read_bool()
        if has_vertex_colors:
            self.vertex_colors = stream.read_floats(num_vertices, 4)
        num_property_states = stream.read_uint()
        if num_property_states:
            self.property_states = stream.read_ints(num_property_states)

    def save(self, stream):
        stream.write_ushort(len(self.vertices))
        stream.write_floats(self.vertices)
        has_uv_coords = len(self.uv_coords)
        stream.write_bool(has_uv_coords)
        if has_uv_coords:
            stream.write_floats(self.uv_coords)
        has_vertex_colors = len(self.vertex_colors)
        stream.write_bool(has_vertex_colors)
        if has_vertex_colors:
            stream.write_floats(self.vertex_colors)
        num_property_states = len(self.property_states)
        stream.write_uint(num_property_states)
        if num_property_states:
            stream.write_ints(self.property_states)


if __name__ == "__main__":
    from es3.utils.typing import *
