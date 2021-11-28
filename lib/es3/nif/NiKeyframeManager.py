from __future__ import annotations

from es3.utils.math import np, zeros

from .NiObject import NiObject
from .NiTimeController import NiTimeController


_dtype = np.dtype("O, <i")


class NiSequence(NiObject):
    sequence_name: str = ""
    keyframe_file: str = ""
    unknown_int: int32 = 0
    unknown_object: NiObject | None = None
    name_controller_pairs: ndarray = zeros(0, dtype=_dtype)

    def load(self, stream):
        self.sequence_name = stream.read_str()

        has_external_kf = stream.read_ubyte()
        if has_external_kf:
            self.keyframe_file = stream.read_str()
        else:
            self.unknown_int = stream.read_int()
            self.unknown_object = stream.read_link()

        num_name_controller_pairs = stream.read_uint()
        if num_name_controller_pairs:
            self.name_controller_pairs = zeros(num_name_controller_pairs, dtype=_dtype)
            for i in range(num_name_controller_pairs):
                self.name_controller_pairs[i] = stream.read_str(), stream.read_int()

    def save(self, stream):
        stream.write_str(self.sequence_name)

        stream.write_ubyte(bool(self.keyframe_file))
        if self.keyframe_file:
            stream.write_str(self.keyframe_file)
        else:
            stream.write_int(self.unknown_int)
            stream.write_link(self.unknown_object)

        stream.write_uint(len(self.name_controller_pairs))
        for name, controller in self.name_controller_pairs.tolist():
            stream.write_str(name)
            stream.write_int(controller)


class NiKeyframeManager(NiTimeController):
    sequences: list[NiSequence] = []

    def load(self, stream):
        super().load(stream)
        num_sequences = stream.read_uint()
        self.sequences = [stream.read_type(NiSequence) for _ in range(num_sequences)]

    def save(self, stream):
        super().save(stream)
        stream.write_int(len(self.sequences))
        for item in self.sequences:
            item.save(stream)


if __name__ == "__main__":
    from es3.utils.typing import *
