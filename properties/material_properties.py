import bpy

from ..nif_shader import NodeTreeWrapper


class TextureProperties(bpy.types.PropertyGroup):

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
        return self.texture_node.interpolation != "Closest"

    def set_use_mipmaps(self, value):
        self.texture_node.interpolation = "Linear" if value else "Closest"

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
        return self.texture_node.extension == "REPEAT"

    def set_use_repeat(self, state):
        self.texture_node.extension = "REPEAT" if state else "CLIP"

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
        uv_node = self.nodes.new("ShaderNodeUVMap")
        vec_sub = self.nodes.new("ShaderNodeVectorMath")
        mapping = self.nodes.new("ShaderNodeMapping")
        vec_add = self.nodes.new("ShaderNodeVectorMath")

        # set defaults
        vec_sub.operation = "SUBTRACT"
        vec_sub.inputs[1].default_value = [0.5] * 3
        vec_add.inputs[1].default_value = [0.5] * 3

        # create links
        self.links.new(uv_node.outputs[0], vec_sub.inputs[0])
        self.links.new(vec_sub.outputs[0], mapping.inputs[0])
        self.links.new(mapping.outputs[0], vec_add.inputs[0])

        return uv_node, vec_sub, mapping, vec_add

    def update_chain_from_tx(self, tx_node, uv_name):
        # find an existing uv node for this uv map
        for uv_node in self.nodes_of_type("ShaderNodeUVMap"):
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


class MaterialProperties(bpy.types.PropertyGroup, NodeTreeWrapper):
    material: bpy.props.PointerProperty(type=bpy.types.Material)
    texture_slots: bpy.props.CollectionProperty(type=TextureProperties)
    alpha_flags: bpy.props.IntProperty()
    shine: bpy.props.FloatProperty()

    @staticmethod
    def register():
        bpy.types.Material.mw = bpy.props.PointerProperty(type=MaterialProperties)

    @staticmethod
    def unregister():
        pass

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
            self.vertex_color.layer_name = (vc.active or vc.new()).name
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
