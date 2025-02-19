import bpy


class ObjectProperties(bpy.types.PropertyGroup):
    object_flags: bpy.props.IntProperty(name="Flags", min=0, max=65535, default=2)

    @staticmethod
    def register() -> None:
        bpy.types.Object.mw = bpy.props.PointerProperty(type=ObjectProperties)
        bpy.types.PoseBone.mw = bpy.props.PointerProperty(type=ObjectProperties)

    @staticmethod
    def unregister() -> None:
        pass
