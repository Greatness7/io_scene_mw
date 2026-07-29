import bpy


class ObjectPanel(bpy.types.Panel):
    bl_idname = "OBJECT_PT_MW_ObjectPanel"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "object"
    bl_label = "Morrowind"

    def draw(self, context):
        ob = context.active_object
        layout = self.layout

        layout.prop(ob.mw, "object_flags")

        layout.separator()
        layout.prop(ob.mw, "block_type")
        if ob.mw.block_type == "NiLODNode":
            layout.prop(ob.mw, "lod_center")

        if (ob.parent is not None) and (ob.parent.mw.block_type == "NiLODNode"):
            layout.prop(ob.mw, "lod_far_extent")
