from __future__ import annotations

from io import BytesIO
from struct import Struct
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import DTypeLike, NDArray

Byte = Struct("<b")  # int8
UByte = Struct("<B")  # uint8
Short = Struct("<h")  # int16
UShort = Struct("<H")  # uint16
Int = Struct("<i")  # int32
UInt = Struct("<I")  # uint32
Float = Struct("<f")  # float32

ByteDType = np.dtype(Byte.format)
UByteDType = np.dtype(UByte.format)
ShortDType = np.dtype(Short.format)
UShortDType = np.dtype(UShort.format)
IntDType = np.dtype(Int.format)
UIntDType = np.dtype(UInt.format)
FloatDType = np.dtype(Float.format)


class BinaryStream(BytesIO):
    __slots__ = ()

    def read_array(self, shape: int | tuple[int, ...], dtype: DTypeLike) -> NDArray[Any]:
        array = np.empty(shape, dtype)
        # ndarray only gained a typed __buffer__ in numpy 2.0; .data is the
        # portable way to hand the checker something buffer-shaped.
        self.readinto(array.data)
        return array

    def write_array(self, array: NDArray[Any], dtype: DTypeLike) -> None:
        if array.dtype != dtype:
            array = array.astype(dtype, copy=False)
        self.write(np.ascontiguousarray(array).data)

    def read_byte(self) -> int:
        return Byte.unpack(self.read(1))[0]

    def write_byte(self, value: int) -> None:
        self.write(Byte.pack(value))

    def read_bytes(self, *shape: int) -> NDArray[Any]:
        return self.read_array(shape, ByteDType)

    def write_bytes(self, array: NDArray[Any]) -> None:
        self.write_array(array, ByteDType)

    def read_ubyte(self) -> int:
        return UByte.unpack(self.read(1))[0]

    def write_ubyte(self, value: int) -> None:
        self.write(UByte.pack(value))

    def read_ubytes(self, *shape: int) -> NDArray[Any]:
        return self.read_array(shape, UByteDType)

    def write_ubytes(self, array: NDArray[Any]) -> None:
        self.write_array(array, UByteDType)

    def read_short(self) -> int:
        return Short.unpack(self.read(2))[0]

    def write_short(self, value: int) -> None:
        self.write(Short.pack(value))

    def read_shorts(self, *shape: int) -> NDArray[Any]:
        return self.read_array(shape, ShortDType)

    def write_shorts(self, array: NDArray[Any]) -> None:
        self.write_array(array, ShortDType)

    def read_ushort(self) -> int:
        return UShort.unpack(self.read(2))[0]

    def write_ushort(self, value: int) -> None:
        self.write(UShort.pack(value))

    def read_ushorts(self, *shape: int) -> NDArray[Any]:
        return self.read_array(shape, UShortDType)

    def write_ushorts(self, array: NDArray[Any]) -> None:
        self.write_array(array, UShortDType)

    def read_int(self) -> int:
        return Int.unpack(self.read(4))[0]

    def write_int(self, value: int) -> None:
        self.write(Int.pack(value))

    def read_ints(self, *shape: int) -> NDArray[Any]:
        return self.read_array(shape, IntDType)

    def write_ints(self, array: NDArray[Any]) -> None:
        self.write_array(array, IntDType)

    def read_uint(self) -> int:
        return UInt.unpack(self.read(4))[0]

    def write_uint(self, value: int) -> None:
        self.write(UInt.pack(value))

    def read_uints(self, *shape: int) -> NDArray[Any]:
        return self.read_array(shape, UIntDType)

    def write_uints(self, array: NDArray[Any]) -> None:
        self.write_array(array, UIntDType)

    def read_float(self) -> float:
        return Float.unpack(self.read(4))[0]

    def write_float(self, value: float) -> None:
        self.write(Float.pack(value))

    def read_floats(self, *shape: int) -> NDArray[Any]:
        return self.read_array(shape, FloatDType)

    def write_floats(self, array: NDArray[Any]) -> None:
        self.write_array(array, FloatDType)

    def read_str(self) -> str:
        data = self.read(UInt.unpack(self.read(4))[0])
        return data.decode(encoding="cp1252", errors="replace")

    def write_str(self, value: str) -> None:
        data = value.encode(encoding="cp1252", errors="replace")
        self.write(UInt.pack(len(data)) + data)
