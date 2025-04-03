import bpy
import pathlib

SHADER_PATH = pathlib.Path(__file__).parent / "assets" / "shader.blend"


def execute(mesh):
    name = getattr(mesh.active_material, "name", "")
    image = get_base_texture_image(mesh.active_material)

    material = create_material(name)
    try:  # assign to active index if viable
        mesh.data.materials[mesh.active_material_index] = material
    except IndexError:
        mesh.data.materials.append(material)

    try:
        material.mw.base_texture.image = image
        material.mw.base_texture.layer = mesh.data.uv_layers.active.name
    except AttributeError:
        pass

    return material.mw


def create_material(name=""):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    material.blend_method = "OPAQUE"
    if bpy.app.version < (4, 3, 0):
        material.shadow_method = "NONE"
    material.alpha_threshold = 0.0
    material.use_backface_culling = True
    material.show_transparent_back = False
    material.mw.material = material
    material.mw.reset_nodes()
    return material


def get_base_texture_image(mat):
    try:
        node = mat.node_tree.get_output_node('EEVEE')
        while node.bl_idname != "ShaderNodeTexImage":
            node = node.inputs[0].links[0].from_node
        return node.image
    except (AttributeError, LookupError):
        return None


class NodeTreeWrapper:

    node_tree = property(lambda self: self.material.node_tree)

    nodes = property(lambda self: self.material.node_tree.nodes)
    links = property(lambda self: self.material.node_tree.links)

    shader = property(lambda self: self.nodes["MW Shader"])
    inputs = property(lambda self: self.shader.inputs)

    # NODES
    vertex_color = property(lambda self: self.nodes["Vertex Color"])

    # INPUTS
    ambient_input = property(lambda self: self.inputs["Ambient Color"])
    diffuse_input = property(lambda self: self.inputs["Diffuse Color"])
    specular_input = property(lambda self: self.inputs["Specular Color"])
    emissive_input = property(lambda self: self.inputs["Emissive Color"])

    alpha_input = property(lambda self: self.inputs["Base Alpha"])
    opacity_input = property(lambda self: self.inputs["Diffuse Alpha"])
    alpha_factor_input = property(lambda self: self.inputs["Alpha Factor"])

    # COLORS
    ambient_color = property(lambda self: self.ambient_input.default_value)
    diffuse_color = property(lambda self: self.diffuse_input.default_value)
    specular_color = property(lambda self: self.specular_input.default_value)
    emissive_color = property(lambda self: self.emissive_input.default_value)

    # TEXTURE NODES
    texture_group = property(lambda self: self.nodes["MW Inputs"])
    texture_outputs = property(lambda self: self.texture_group.outputs)

    # TEXTURE SLOTS
    base_texture = property(lambda self: self.texture_slots["Base Texture"])
    dark_texture = property(lambda self: self.texture_slots["Dark Texture"])
    detail_texture = property(lambda self: self.texture_slots["Detail Texture"])
    glow_texture = property(lambda self: self.texture_slots["Glow Texture"])
    decal_0_texture = property(lambda self: self.texture_slots["Decal 0 Texture"])
    decal_1_texture = property(lambda self: self.texture_slots["Decal 1 Texture"])
    decal_2_texture = property(lambda self: self.texture_slots["Decal 2 Texture"])
    decal_3_texture = property(lambda self: self.texture_slots["Decal 3 Texture"])

    @property
    def alpha(self):
        return self.opacity_input.default_value

    @alpha.setter
    def alpha(self, value):
        self.opacity_input.default_value = value

    @property
    def alpha_factor(self):
        return self.alpha_factor_input.default_value

    @alpha_factor.setter
    def alpha_factor(self, value):
        self.alpha_factor_input.default_value = value

    def reset_nodes(self):
        self.nodes.clear()
        self.links.clear()

        # Shader Group
        self.create_shader_group("MW Shader", 0, 0)

        # Texture Groups
        self.create_texture_group("MW Inputs", -200, 0)

        # Vertex Color
        self.create_node("Vertex Color", "ShaderNodeVertexColor", -200, -400)

    def create_shader_group(self, name, x, y):
        group = self.create_node(name, "ShaderNodeGroup", x, y)

        node_tree = bpy.data.node_groups.get(name)
        if node_tree:
            # use previously loaded node tree
            group.node_tree = node_tree
        else:
            # load the node tree from library
            with bpy.data.libraries.load(str(SHADER_PATH), link=False) as (src, dst):
                dst.node_groups.append('MW Shader')
            group.node_tree, = dst.node_groups

        return group

    def create_texture_group(self, name, x, y):
        group = self.create_node(name, 'ShaderNodeGroup', x, y)
        group.node_tree = bpy.data.node_groups.new(name, 'ShaderNodeTree')

        nodes = group.node_tree.nodes
        links = group.node_tree.links

        # make nodes
        base_image = nodes.new('ShaderNodeTexImage')
        dark_image = nodes.new('ShaderNodeTexImage')
        deta_image = nodes.new('ShaderNodeTexImage')
        glow_image = nodes.new('ShaderNodeTexImage')
        dec0_image = nodes.new('ShaderNodeTexImage')
        dec1_image = nodes.new('ShaderNodeTexImage')
        dec2_image = nodes.new('ShaderNodeTexImage')
        dec3_image = nodes.new('ShaderNodeTexImage')

        # name nodes
        base_image.name = base_image.label = "Base Texture"
        dark_image.name = dark_image.label = "Dark Texture"
        deta_image.name = deta_image.label = "Detail Texture"
        glow_image.name = glow_image.label = "Glow Texture"
        dec0_image.name = dec0_image.label = "Decal 0 Texture"
        dec1_image.name = dec1_image.label = "Decal 1 Texture"
        dec2_image.name = dec2_image.label = "Decal 2 Texture"
        dec3_image.name = dec3_image.label = "Decal 3 Texture"

        # texture slots
        for node in nodes:
            slot = self.texture_slots.add()
            slot.node_tree = group.node_tree
            slot.name = node.name

        # outputs
        self.new_output_socket(group, 'NodeSocketColor', "Base Texture")
        self.new_output_socket(group, 'NodeSocketColor', "Dark Texture")
        self.new_output_socket(group, 'NodeSocketColor', "Detail Texture")
        self.new_output_socket(group, 'NodeSocketColor', "Glow Texture")
        self.new_output_socket(group, 'NodeSocketColor', "Decal 0 Texture")
        self.new_output_socket(group, 'NodeSocketColor', "Decal 1 Texture")
        self.new_output_socket(group, 'NodeSocketColor', "Decal 2 Texture")
        self.new_output_socket(group, 'NodeSocketColor', "Decal 3 Texture")
        self.new_output_socket(group, 'NodeSocketColor', "Base Alpha")
        self.new_output_socket(group, 'NodeSocketColor', "Dark Alpha")
        self.new_output_socket(group, 'NodeSocketColor', "Decal 0 Alpha")
        self.new_output_socket(group, 'NodeSocketColor', "Decal 1 Alpha")
        self.new_output_socket(group, 'NodeSocketColor', "Decal 2 Alpha")
        self.new_output_socket(group, 'NodeSocketColor', "Decal 3 Alpha")

        # links
        output = nodes.new('NodeGroupOutput')
        links.new(base_image.outputs["Color"], output.inputs["Base Texture"])
        links.new(dark_image.outputs["Color"], output.inputs["Dark Texture"])
        links.new(deta_image.outputs["Color"], output.inputs["Detail Texture"])
        links.new(glow_image.outputs["Color"], output.inputs["Glow Texture"])
        links.new(dec0_image.outputs["Color"], output.inputs["Decal 0 Texture"])
        links.new(dec1_image.outputs["Color"], output.inputs["Decal 1 Texture"])
        links.new(dec2_image.outputs["Color"], output.inputs["Decal 2 Texture"])
        links.new(dec3_image.outputs["Color"], output.inputs["Decal 3 Texture"])
        links.new(base_image.outputs["Alpha"], output.inputs["Base Alpha"])
        links.new(dark_image.outputs["Alpha"], output.inputs["Dark Alpha"])
        links.new(dec0_image.outputs["Alpha"], output.inputs["Decal 0 Alpha"])
        links.new(dec1_image.outputs["Alpha"], output.inputs["Decal 1 Alpha"])
        links.new(dec2_image.outputs["Alpha"], output.inputs["Decal 2 Alpha"])
        links.new(dec3_image.outputs["Alpha"], output.inputs["Decal 3 Alpha"])

    def create_node(self, name, node_type, x, y):
        node = self.nodes.new(node_type)
        node.name = node.label = name
        node.location = x, y
        return node

    def create_link(self, src_node, dst_node, src_key=0, dst_key=0):
        return self.links.new(src_node.outputs[src_key], dst_node.inputs[dst_key])

    def remove_link(self, socket):
        for link in socket.links:
            self.links.remove(link)

    def validate(self):
        m = self.material
        if not (m and m.use_nodes):
            raise TypeError("Invalid Material: use_nodes not enabled")
        if m.node_tree.get_output_node('EEVEE'):
            raise TypeError("Invalid Material: eevee output override")
        if "MW Shader" not in m.node_tree.nodes:
            raise TypeError("Invalid Material: mw_shader not present")
        return self

    @staticmethod
    def new_output_socket(group, socket_type, name):
        if bpy.app.version >= (4, 0, 0):
            group.node_tree.interface.new_socket(name, in_out='OUTPUT', socket_type=socket_type)
        else:
            group.node_tree.outputs.new(socket_type, name)


class TextureSlot(bpy.types.PropertyGroup):

    node_tree: bpy.props.PointerProperty(type=bpy.types.NodeTree)

    nodes = property(lambda self: self.node_tree.nodes)
    links = property(lambda self: self.node_tree.links)

    outputs = property(lambda self: self.id_data.node_tree.nodes["MW Inputs"].outputs)
    shader_inputs = property(lambda self: self.id_data.node_tree.nodes["MW Shader"].inputs)

    texture_node = property(lambda self: self.nodes[self.name])
    mapping_node = property(lambda self: self.get_chain_from_tx(self.texture_node)[2])

    # -------------
    # Texture Image
    # -------------
    def update_image(self, context):
        tx_node, image = self.texture_node, self.image

        if tx_node.image == image:  # image unchanged
            return
        elif image and not tx_node.image:  # image added
            self.update_links()
        elif tx_node.image and not image:  # image removed
            self.remove_links()

        tx_node.image = image

    image: bpy.props.PointerProperty(
        type=bpy.types.Image,
        update=update_image,
    )

    # ------------
    # UV Map Layer
    # ------------
    def update_layer(self, context):
        tx_node, layer = self.texture_node, self.layer

        if layer:
            self.update_chain_from_tx(tx_node, layer)
        else:
            self.remove_chain_from_tx(tx_node)

    layer: bpy.props.StringProperty(
        update=update_layer,
    )

    # -----------
    # Use Mipmaps
    # -----------
    def get_use_mipmaps(self):
        return self.texture_node.interpolation != 'Closest'

    def set_use_mipmaps(self, value):
        self.texture_node.interpolation = 'Linear' if value else 'Closest'

    use_mipmaps: bpy.props.BoolProperty(
        name="Use Mipmaps",
        description="",
        get=get_use_mipmaps,
        set=set_use_mipmaps,
        options=set(),
    )

    # ----------
    # Use Repeat
    # ----------
    def get_use_repeat(self):
        return self.texture_node.extension == 'REPEAT'

    def set_use_repeat(self, state):
        self.texture_node.extension = 'REPEAT' if state else 'CLIP'

    use_repeat: bpy.props.BoolProperty(
        name="Repeat Image",
        description="",
        get=get_use_repeat,
        set=set_use_repeat,
        options=set(),
    )

    # -------------
    # Utility Funcs
    # -------------
    def create_chain(self):
        # create nodes
        uv_node = self.nodes.new('ShaderNodeUVMap')
        vec_sub = self.nodes.new('ShaderNodeVectorMath')
        mapping = self.nodes.new('ShaderNodeMapping')
        vec_add = self.nodes.new('ShaderNodeVectorMath')

        # set defaults
        vec_sub.operation = 'SUBTRACT'
        vec_sub.inputs[1].default_value = [0.5] * 3
        vec_add.inputs[1].default_value = [0.5] * 3

        # create links
        self.links.new(uv_node.outputs[0], vec_sub.inputs[0])
        self.links.new(vec_sub.outputs[0], mapping.inputs[0])
        self.links.new(mapping.outputs[0], vec_add.inputs[0])

        return uv_node, vec_sub, mapping, vec_add

    def update_chain_from_tx(self, tx_node, uv_name):
        # find an existing uv node for this uv map
        for uv_node in self.nodes_of_type('ShaderNodeUVMap'):
            if uv_node.uv_map == uv_name:
                break
        else:  # no existing node for uv map found
            uv_node, *_ = self.create_chain()
            uv_node.uv_map = uv_name

        vec_add = self.get_chain_from_uv(uv_node)[-1]
        self.links.new(vec_add.outputs[0], tx_node.inputs[0])

    def remove_chain_from_tx(self, tx_node):
        links = tx_node.inputs[0].links
        if not links:
            # the node chain is already removed
            return

        vec_add = links[0].from_node
        if len(vec_add.outputs[0].links) > 1:
            # there are others using this chain
            self.links.remove(links[0])
        else:
            # we are only link to this uv chain
            for node in self.get_chain_from_tx(tx_node):
                self.nodes.remove(node)

    def update_links(self):
        outputs = self.outputs
        inputs = self.shader_inputs
        links = self.id_data.node_tree.links
        for name in self.active_socket_names():
            try:
                links.new(outputs[name], inputs[name])
            except KeyError:
                pass

    def remove_links(self):
        # outputs = self.outputs
        inputs = self.shader_inputs
        links = self.id_data.node_tree.links
        for name in self.active_socket_names():
            for outer_link in inputs[name].links:
                try:
                    links.remove(outer_link)
                except KeyError:
                    pass

    @staticmethod
    def get_chain_from_uv(uv_node):
        vec_sub = uv_node.outputs[0].links[0].to_node
        mapping = vec_sub.outputs[0].links[0].to_node
        vec_add = mapping.outputs[0].links[0].to_node
        return uv_node, vec_sub, mapping, vec_add

    @staticmethod
    def get_chain_from_tx(tx_node):
        vec_add = tx_node.inputs[0].links[0].from_node
        mapping = vec_add.inputs[0].links[0].from_node
        vec_sub = mapping.inputs[0].links[0].from_node
        uv_node = vec_sub.inputs[0].links[0].from_node
        return uv_node, vec_sub, mapping, vec_add

    def active_socket_names(self):
        outputs = self.texture_node.outputs
        return (link.to_socket.name for o in outputs for link in o.links)

    def nodes_of_type(self, bl_idname):
        return (node for node in self.nodes if node.bl_idname == bl_idname)


class NiMaterialProps(bpy.types.PropertyGroup, NodeTreeWrapper):
    material: bpy.props.PointerProperty(type=bpy.types.Material)
    texture_slots: bpy.props.CollectionProperty(type=TextureSlot)
    alpha_flags: bpy.props.IntProperty()
    shine: bpy.props.FloatProperty()

    # -------------
    # Vertex Colors
    # -------------
    def get_use_vertex_colors(self):
        try:
            link = self.diffuse_input.links[0]
        except IndexError:
            return False
        return self.vertex_color == link.from_node

    def set_use_vertex_colors(self, value):
        if value:
            vc = bpy.context.active_object.data.vertex_colors
            self.vertex_color.layer_name = ( vc.active or vc.new() ).name
            self.links.new(self.diffuse_input, self.vertex_color.outputs["Color"])
            self.links.new(self.opacity_input, self.vertex_color.outputs["Alpha"])
        else:
            self.remove_link(self.inputs["Diffuse Color"])
            self.remove_link(self.inputs["Diffuse Alpha"])

    use_vertex_colors: bpy.props.BoolProperty(
        name="Vertex Colors",
        description="Use vertex colors rather than material colors.",
        get=get_use_vertex_colors,
        set=set_use_vertex_colors,
        options=set(),
    )

    # -----------
    # Alpha Blend
    # -----------
    def get_use_alpha_blend(self):
        return (self.alpha_factor == 1.0) and (self.material.blend_method == "BLEND")

    def set_use_alpha_blend(self, value):
        self.alpha_factor = 1.0 if value else 0.0
        if value:
            self.material.blend_method = "BLEND"
        else:
            self.material.blend_method = "CLIP" if value else "OPAQUE"

    use_alpha_blend: bpy.props.BoolProperty(
        name="Alpha Blend",
        description="Enable Alpha Blend",
        get=get_use_alpha_blend,
        set=set_use_alpha_blend,
        options=set(),
    )

    # ----------
    # Alpha Clip
    # ----------
    def get_use_alpha_clip(self):
        if self.alpha_factor == 0.0:
            return False
        if self.material.blend_method == "CLIP":
            return True
        if self.material.blend_method != "BLEND":
            return False
        return (self.alpha_flags & 512) != 0

    def set_use_alpha_clip(self, value):
        self.alpha_factor = 1.0 if value else 0.0
        if self.material.blend_method != "BLEND":
            self.material.blend_method = "CLIP" if value else "OPAQUE"
        if value:
            self.alpha_flags |= 512
        else:
            self.alpha_flags &= ~512

    use_alpha_clip: bpy.props.BoolProperty(
        name="Alpha Clip",
        description="Enable Alpha Clip",
        get=get_use_alpha_clip,
        set=set_use_alpha_clip,
        options=set(),
    )
