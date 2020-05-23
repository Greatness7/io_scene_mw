from __future__ import annotations

from .NiObject import NiObject
from .NiTimeController import NiTimeController


class NiSequence(NiObject):
    sequence_name: str = ""
    keyframe_file: str = ""
    unknown_int: int32 = 0
    unknown_object: Optional[NiObject] = None
    target_names: List[str] = []
    target_controllers: List[Optional[NiKeyframeController]] = []

    def load(self, stream):
        self.sequence_name = stream.read_str()

        external_kf = stream.read_ubyte()
        if external_kf:
            self.keyframe_file = stream.read_str()
        else:
            self.unknown_int = stream.read_int()
            self.unknown_object = stream.read_link()

        num_targets = stream.read_uint()
        for _ in range(num_targets):
            target_name = stream.read_str()
            target_controller = stream.read_link()  # note: engine will create invalid links here
            self.target_names.append(target_name)
            self.target_controllers.append(target_controller)

    def save(self, stream):
        stream.write_str(self.sequence_name)

        stream.write_ubyte(bool(self.keyframe_file))
        if self.keyframe_file:
            stream.write_str(self.keyframe_file)
        else:
            stream.write_int(self.unknown_int)
            stream.write_link(self.unknown_object)

        stream.write_uint(len(self.target_names))
        for target_name in self.target_names:
            stream.write_str(target_name)
            stream.write_int(-1)  # target_controller is ignored -- see load function comment


class NiKeyframeManager(NiTimeController):
    sequences: List[NiSequence] = []

    def load(self, stream):
        super().load(stream)
        num_sequences = stream.read_int()
        self.sequences = [stream.read_type(NiSequence) for _ in range(num_sequences)]

    def save(self, stream):
        super().save(stream)
        stream.write_int(len(self.sequences))
        for item in self.sequences:
            item.save(stream)


if __name__ == "__main__":
    from es3.nif import NiKeyframeController
    from es3.utils.typing import *
