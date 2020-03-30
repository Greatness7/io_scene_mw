from io import BytesIO
from struct import Struct

import numpy as np

Byte = Struct("<b")  # int8
UByte = Struct("<B")  # uint8
Short = Struct("<h")  # int16
UShort = Struct("<H")  # uint16
Int = Struct("<i")  # int32
UInt = Struct("<I")  # uint32
Float = Struct("<f")  # float32


class BinaryStream(BytesIO):
    __slots__ = (
        "read_byte",
        "read_bytes",
        "read_ubyte",
        "read_ubytes",
        "read_short",
        "read_shorts",
        "read_ushort",
        "read_ushorts",
        "read_int",
        "read_ints",
        "read_uint",
        "read_uints",
        "read_float",
        "read_floats",
        "read_str",
        "read_strs",
        "write_byte",
        "write_bytes",
        "write_ubyte",
        "write_ubytes",
        "write_short",
        "write_shorts",
        "write_ushort",
        "write_ushorts",
        "write_int",
        "write_ints",
        "write_uint",
        "write_uints",
        "write_float",
        "write_floats",
        "write_str",
        "write_strs",
    )

    def __init__(self, initial_bytes=None):
        super().__init__(initial_bytes)

        (self.read_byte,
         self.write_byte,
         self.read_bytes,
         self.write_bytes) = self.make_read_write_for_struct(Byte)

        (self.read_ubyte,
         self.write_ubyte,
         self.read_ubytes,
         self.write_ubytes) = self.make_read_write_for_struct(UByte)

        (self.read_short,
         self.write_short,
         self.read_shorts,
         self.write_shorts) = self.make_read_write_for_struct(Short)

        (self.read_ushort,
         self.write_ushort,
         self.read_ushorts,
         self.write_ushorts) = self.make_read_write_for_struct(UShort)

        (self.read_int,
         self.write_int,
         self.read_ints,
         self.write_ints) = self.make_read_write_for_struct(Int)

        (self.read_uint,
         self.write_uint,
         self.read_uints,
         self.write_uints) = self.make_read_write_for_struct(UInt)

        (self.read_float,
         self.write_float,
         self.read_floats,
         self.write_floats) = self.make_read_write_for_struct(Float)

        (self.read_str,
         self.write_str,
         self.read_strs,
         self.write_strs) = self.make_read_write_for_string(UInt)

    def make_read_write_for_struct(self, struct):
        # declare these in the local scope for faster name resolution
        read = self.read
        write = self.write
        pack = struct.pack
        unpack = struct.unpack
        size = struct.size
        # these functions are used for efficient read/write of arrays
        empty = np.empty
        dtype = np.dtype(struct.format)
        readinto = self.readinto

        def read_value():
            return unpack(read(size))[0]

        def write_value(value):
            write(pack(value))

        def read_values(*shape):
            array = empty(shape, dtype)
            # noinspection PyTypeChecker
            readinto(array)
            return array

        def write_values(array):
            if array.dtype != dtype:
                array = array.astype(dtype)
            write(array.tobytes())

        return read_value, write_value, read_values, write_values

    def make_read_write_for_string(self, struct):
        # declare these in the local scope for faster name resolutions
        read = self.read
        write = self.write
        pack = struct.pack
        unpack = struct.unpack

        def read_string():
            value = read(*unpack(read(4)))
            return value.decode(errors="surrogateescape")

        def write_string(value):
            value = value.encode(errors="surrogateescape")
            write(pack(len(value)) + value)

        return read_string, write_string, NotImplemented, NotImplemented

    def read_array(self, shape, dtype=np.float32):  # TODO remove this
        array = np.empty(shape, dtype)
        # noinspection PyTypeChecker
        self.readinto(array)
        return array

    def write_array(self, array, dtype=np.float32):  # TODO remove this
        if array.dtype != dtype:
            array = array.astype(dtype)
        self.write(array.tobytes())
