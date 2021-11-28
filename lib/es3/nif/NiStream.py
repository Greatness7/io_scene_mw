from __future__ import annotations

from itertools import chain
from math import isclose

from es3 import nif
from .NiBinaryStream import NiBinaryStream


class NiStream:
    __slots__ = "roots",

    HEADER = b"NetImmerse File Format, Version 4.0.0.2\n"
    VERSION = 0x4000002
    TYPES = vars(nif)

    def __init__(self):
        self.roots: list[NiObject] = []

    def load(self, filepath: PathLike):
        with NiBinaryStream.reader(filepath) as stream:
            assert stream.readline() == self.HEADER
            assert stream.read_uint() == self.VERSION
            self.roots += stream.read_objects(self.TYPES)

    def save(self, filepath: PathLike):
        with NiBinaryStream.writer(filepath) as stream:
            stream.write(self.HEADER)
            stream.write_uint(self.VERSION)
            stream.write_objects(self.objects(), self.roots)

    def sort(self):
        for obj in self.objects():
            obj.sort()

    def apply_scale(self, scale: float):
        if not isclose(scale, 1.0, rel_tol=0, abs_tol=1e-6):
            for obj in self.objects():
                obj.apply_scale(scale)

    @property
    def root(self) -> NiObject | None:
        return self.roots[0] if self.roots else None

    @root.setter
    def root(self, node: NiObject | None):
        self.roots = [node]

    def objects(self, iterator=chain.from_iterable) -> Iterator[NiObject]:
        yield from iterator(root._traverse({None}) for root in self.roots)

    def objects_of_type(self, cls: type[T]) -> Iterator[T]:
        return (obj for obj in self.objects() if isinstance(obj, cls))

    def find_object_by_name(self, name, object_type=None, fn=str.lower):
        name = fn(name)
        for obj in self.objects_of_type(object_type or nif.NiObjectNET):
            if fn(obj.name) == name:
                return obj

    def merge_properties(self, digits=4, ignore=(), sanitize_filenames=True):
        """..."""
        cache = {}

        def ensure_unique(obj: NiObject):
            try:
                return cache[obj]
            except KeyError:
                key = obj._astuple(digits, ignore)
                obj = cache[obj] = cache.setdefault(key, obj)
                return obj

        # flags on these properties do nothing and can interfere with the merging process
        for prop in self.objects_of_type((nif.NiMaterialProperty, nif.NiTexturingProperty)):
            prop.flags = 0

        for obj in self.objects_of_type(nif.NiAVObject):
            for i, prop in enumerate(obj.properties):
                if prop is None:
                    continue

                # We must first handle any objects referenced within the properties. For
                # now this only matters with NiTexturingProperty, which holds references
                # to multiple NiSourceTexture(s). Replace any duplicated source textures
                # with those already encountered earlier in the routine.
                if isinstance(prop, nif.NiTexturingProperty):
                    for name, slot in zip(prop.texture_keys, prop.texture_maps):
                        if not (slot and slot.source):
                            continue
                        if sanitize_filenames:
                            slot.source.sanitize_filename()
                        # merge duplicate source textures
                        slot.source = ensure_unique(slot.source)
                        # merge duplicate texturing slots
                        setattr(prop, name, ensure_unique(slot))

                # merge duplicate properties
                obj.properties[i] = ensure_unique(prop)

    def extract_keyframe_data(self) -> NiStream:
        """Extract animation data. Useful for generating 'x.nif' and 'x.kf' files."""

        # extract text data
        for obj in self.objects_of_type(nif.NiObjectNET):
            text_data = obj.extra_datas.discard_type(nif.NiTextKeyExtraData)
            if text_data:
                break
        else:
            raise ValueError("extract_keyframe_data: no NiTextKeyExtraData object was found.")

        # extract controllers
        kf_controllers = {}

        def extract_kf_controller(owner):
            if isinstance(owner, nif.NiObjectNET):
                kf_controller = owner.controllers.discard_type(nif.NiKeyframeController)
                if kf_controller:
                    kf_controller.target = None
                    kf_controllers[owner] = kf_controller

        for root in self.roots:
            extract_kf_controller(root)
            for node in root.descendants():
                extract_kf_controller(node)

        if not kf_controllers:
            raise ValueError("extract_keyframe_data: no NiKeyframeController objects were found.")

        # create x.kf output
        output = nif.NiStream()

        # assign root object
        output.root = nif.NiSequenceStreamHelper(extra_data=text_data)

        # assign controllers
        extra_datas = []
        controllers = []

        for target, controller in kf_controllers.items():
            extra_data = nif.NiStringExtraData(string_data=target.name)
            extra_datas.append(extra_data)
            controllers.append(controller)

        output.root.extra_datas.extend(extra_datas)
        output.root.controllers.extend(controllers)

        return output

    def attach_keyframe_data(self, kf_data: NiStream):
        # TODO can a single node have multiple keyframe controllers? if so find_type() is not enough

        kf_root = kf_data.root
        if not isinstance(kf_root, nif.NiSequenceStreamHelper):
            raise ValueError("attach_keyframe_data: kf_data root must be a NiSequenceStreamHelper")

        kf_text_data, *kf_string_datas = kf_root.extra_datas
        assert len(kf_string_datas) >= 1

        # find the skeleton root node
        skeleton_root_name = kf_string_datas[0].string_data
        skeleton_root = self.find_object_by_name(skeleton_root_name)
        if skeleton_root is None:
            raise ValueError(f"attach_keyframe_data: unable to find skeleton root {skeleton_root_name}")

        # set skeleton root text data
        skeleton_root.extra_datas.appendleft(kf_text_data)

        # collect controllers/targets
        controllers_to_attach: dict[str, nif.NiKeyframeController] = {
            s.string_data: c for s, c in zip(kf_string_datas, kf_root.controllers)
        }

        # merge controllers into target objects of self
        for obj in self.objects_of_type(nif.NiObjectNET):
            new_controller = controllers_to_attach.get(obj.name)
            if new_controller is not None:
                obj.controllers.discard_type(nif.NiKeyframeController)
                obj.controllers.appendleft(new_controller)
                new_controller.target = obj


if __name__ == "__main__":
    from .NiObject import NiObject
    from es3.utils.typing import *
