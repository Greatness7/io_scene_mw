from __future__ import annotations

from es3 import nif
from collections.abc import Callable, Collection, Iterable
from itertools import chain
from math import isclose
from typing import TypeVar, cast

from ..utils.typing import *

from .NiBinaryStream import NiBinaryStream
from .NiObject import NiObject


NiObjectT = TypeVar("NiObjectT", bound=NiObject)


class NiStream:
    __slots__ = ("roots",)

    HEADER = b"NetImmerse File Format, Version 4.0.0.2\n"
    VERSION = 0x4000002
    TYPES = vars(nif)

    def __init__(self) -> None:
        self.roots: list[NiObject | None] = []

    def load(self, filepath: PathLike) -> None:
        with NiBinaryStream.reader(filepath) as stream:
            assert stream.readline() == self.HEADER
            assert stream.read_uint() == self.VERSION
            self.roots += stream.read_objects(self.TYPES)

    def save(self, filepath: PathLike) -> None:
        with NiBinaryStream.writer(filepath) as stream:
            stream.write(self.HEADER)
            stream.write_uint(self.VERSION)
            stream.write_objects(self.objects(), self.roots)

    def sort(self) -> None:
        for obj in self.objects():
            obj.sort()

    def apply_scale(self, scale: float):
        if not isclose(scale, 1.0, rel_tol=0, abs_tol=1e-6):
            for obj in self.objects():
                obj.apply_scale(scale)

    def apply_time_scale(self, scale: float):
        if not isclose(scale, 1.0, rel_tol=0, abs_tol=1e-6):
            for obj in self.objects():
                obj.apply_time_scale(scale)

    @property
    def root(self) -> NiObject | None:
        return self.roots[0] if self.roots else None

    @root.setter
    def root(self, node: NiObject | None) -> None:
        self.roots = [node]

    def roots_of_type(self, cls: type[NiObjectT]) -> Iterator[NiObjectT]:
        yield from (root for root in self.roots if isinstance(root, cls))

    def objects(
        self, iterator: Callable[[Iterable[Iterable[NiObject]]], Iterable[NiObject]] = chain.from_iterable
    ) -> Iterator[NiObject]:
        traversals = (root.traverse(set()) for root in self.roots if root is not None)
        yield from iterator(traversals)

    def objects_of_type(self, cls: type[NiObjectT] | tuple[type[NiObjectT], ...]) -> Iterator[NiObjectT]:
        return (obj for obj in self.objects() if isinstance(obj, cls))

    def find_object_by_name(
        self, name: str, object_type: type[nif.NiObjectNET] | None = None, fn: Callable[[str], str] = str.lower
    ) -> nif.NiObjectNET | None:
        name = fn(name)
        for obj in self.objects_of_type(object_type or nif.NiObjectNET):
            if fn(obj.name) == name:
                return obj

    def merge_properties(self, digits: int = 4, ignore: Collection[str] = (), sanitize_filenames: bool = True) -> None:
        """..."""
        cache: dict[object, NiObject] = {}

        def ensure_unique(obj: NiObjectT) -> NiObjectT:
            try:
                return cast(NiObjectT, cache[obj])
            except KeyError:
                key = obj.astuple(digits, ignore)
                canonical = cast(NiObjectT, cache.setdefault(key, obj))
                cache[obj] = canonical
                return canonical

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
        kf_controllers: dict[nif.NiObjectNET, nif.NiKeyframeController] = {}

        def extract_kf_controller(owner: NiObject) -> None:
            if isinstance(owner, nif.NiObjectNET):
                if kf_controller := owner.controllers.discard_type(nif.NiKeyframeController):
                    kf_controller.target = None
                    kf_controllers[owner] = kf_controller

        for root in filter(None, self.roots):
            extract_kf_controller(root)
            for node in root.descendants():
                extract_kf_controller(node)

        # create x.kf output
        output = nif.NiStream()

        # assign root object
        output_root = nif.NiSequenceStreamHelper(extra_data=text_data)
        output.root = output_root

        # assign controllers
        extra_datas: list[nif.NiExtraData] = []
        controllers: list[nif.NiTimeController] = []

        for target, controller in kf_controllers.items():
            extra_data = nif.NiStringExtraData(string_data=target.name)
            extra_datas.append(extra_data)
            controllers.append(controller)

        output_root.extra_datas.extend(extra_datas)
        output_root.controllers.extend(controllers)

        return output

    def attach_keyframe_data(self, kf_data: NiStream) -> None:
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
        controllers_to_attach: dict[str, nif.NiTimeController] = {
            s.string_data: c for s, c in zip(kf_string_datas, kf_root.controllers)
        }

        # merge controllers into target objects of self
        for obj in self.objects_of_type(nif.NiObjectNET):
            new_controller = controllers_to_attach.get(obj.name)
            if new_controller is not None:
                obj.controllers.discard_type(nif.NiKeyframeController)
                obj.controllers.appendleft(new_controller)
                new_controller.target = obj
