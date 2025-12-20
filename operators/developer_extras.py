import bpy


class CleanMaterials(bpy.types.Operator):
    bl_idname = "scene.mw_clean_materials"
    bl_options = {"REGISTER", "UNDO"}
    bl_label = "Clean Morrowind Materials"

    def execute(self, context):
        for material in list(bpy.data.materials):
            try:
                material.mw.validate()
                if material.users <= 1:
                    bpy.data.materials.remove(material)
            except:
                continue
        return {"FINISHED"}


class ShowRadius(bpy.types.Operator):
    bl_idname = "mesh.mw_show_radius"
    bl_options = {"REGISTER", "UNDO"}
    bl_label = "Create Radius Sphere"
    bl_description = "Create a sphere representing the selected meshes radius."

    def execute(self, context):
        for mesh in context.selected_objects:
            try:
                center, radius = self.get_center_radius(mesh.data.vertices)
            except (AttributeError, ValueError):
                continue

            ob = bpy.data.objects.new("Radius", None)
            ob.empty_display_type = "SPHERE"
            ob.empty_display_size = 1.0
            ob.scale = [radius] * 3
            ob.location = center
            ob.parent = mesh

            context.scene.collection.objects.link(ob)

        return {"FINISHED"}

    @staticmethod
    def get_center_radius(vertices):
        import numpy as np
        from es3.utils.math import get_exact_center_radius

        # convert to numpy array
        array = np.empty((len(vertices), 3))
        vertices.foreach_get("co", array.ravel())

        # calc center and radius
        center, radius = get_exact_center_radius(array)

        return center, radius
