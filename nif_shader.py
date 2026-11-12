from contextlib import contextmanager

import bpy


def execute(mesh):
    name = getattr(mesh.active_material, "name", "")
    image = get_base_texture_image(mesh.active_material)

    material = create_material(name)

    try:
        slot = mesh.material_slots[mesh.active_material_index]
    except IndexError:
        mesh.data.materials.append(None)
        slot = mesh.material_slots[-1]

    slot.link = 'OBJECT'
    slot.material = material

    try:
        material.mw.base_texture.image = image
        material.mw.base_texture.layer = mesh.data.uv_layers.active.name
    except AttributeError:
        pass

    return material.mw


def create_material(name=""):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    if bpy.app.version < (4, 3, 0):
        material.shadow_method = "NONE"
    material.use_backface_culling = True
    material.show_transparent_back = False
    material.mw.material = material
    material.mw.reset_nodes()
    material.mw.sync_alpha_state()
    return material


def get_base_texture_image(mat):
    try:
        node = mat.node_tree.get_output_node('EEVEE')
        while node.bl_idname != "ShaderNodeTexImage":
            node = node.inputs[0].links[0].from_node
        return node.image
    except (AttributeError, LookupError):
        return None


def new_socket(node_tree, in_out, socket_type, name, default=None, min_value=None, max_value=None):
    if bpy.app.version >= (4, 0, 0):
        base_type, subtype = socket_type, None
        if socket_type == 'NodeSocketFloatFactor':
            base_type, subtype = 'NodeSocketFloat', 'FACTOR'
        socket = node_tree.interface.new_socket(name, in_out=in_out, socket_type=base_type)
        if subtype is not None:
            socket.subtype = subtype
    else:
        sockets = node_tree.inputs if in_out == 'INPUT' else node_tree.outputs
        socket = sockets.new(socket_type, name)
    if min_value is not None:
        socket.min_value = min_value
    if max_value is not None:
        socket.max_value = max_value
    if default is not None:
        socket.default_value = default
    return socket


def new_node(tree, node_type, location, label=""):
    node = tree.nodes.new(node_type)
    node.location = location
    if label:
        node.name = node.label = label
    return node


def new_group_node(tree, group_tree, location, label=""):
    node = new_node(tree, 'ShaderNodeGroup', location, label)
    node.node_tree = group_tree
    return node


@contextmanager
def framed(tree, label):
    """Collect the nodes created inside the block into a labelled frame."""
    existing = {node.name for node in tree.nodes}
    yield
    created = [node for node in tree.nodes if node.name not in existing]
    # The frame stays at the origin, so parenting keeps each node's location as assigned.
    frame = new_node(tree, 'NodeFrame', (0, 0))
    frame.label = label
    for node in created:
        node.parent = frame


def new_input_portal(tree, name, location):
    """A collapsed Group Input for one socket, placed beside the node that consumes it."""
    node = new_node(tree, 'NodeGroupInput', location)
    node.label = name
    node.hide = True
    return node.outputs[name]


def hide_unlinked_outputs(tree):
    """Trim source nodes down to the sockets in use, so each section can have its own Group Input."""
    for node in tree.nodes:
        if node.bl_idname in {'NodeGroupInput', 'ShaderNodeAttribute', 'ShaderNodeLightPath'}:
            for socket in node.outputs:
                socket.hide = not socket.is_linked


def tree_has_input(tree, name):
    if bpy.app.version >= (4, 0, 0):
        return any(it.name == name and it.in_out == 'INPUT' for it in tree.interface.items_tree)
    return name in tree.inputs


def clamp_factor_sockets(tree):
    """Ensure slider-exposed alpha factor inputs are clamped to 0-1 even on pre-fix MW Shader trees."""
    if tree is None or tree.library is not None:
        return

    names = ("Diffuse Alpha", "Alpha Clip Threshold")

    if bpy.app.version >= (4, 0, 0):
        sockets = [it for it in tree.interface.items_tree if it.in_out == 'INPUT' and it.name in names]
        for socket in sockets:
            socket.subtype = 'FACTOR'
    else:
        sockets = [socket for name in names if (socket := tree.inputs.get(name)) is not None]

    for socket in sockets:
        socket.min_value = 0.0
        socket.max_value = 1.0


def get_current_mw_shader_tree():
    """Keep stale groups aside until MaterialProperties.upgrade() migrates their users."""
    tree = bpy.data.node_groups.get("MW Shader")
    if tree and (not tree_has_input(tree, "Alpha Test Function") or not tree_has_input(tree, "Use Vertex Colors")):
        tree.name = "MW Shader.legacy"
        tree = None
    if tree is not None:
        clamp_factor_sockets(tree)
        return tree
    return build_mw_shader_tree()


def build_srgb_encode_tree():
    """The same piecewise input encoding as tes3_blend's native material recipe."""
    tree = bpy.data.node_groups.new("MW sRGB Encode", 'ShaderNodeTree')
    nodes, links = tree.nodes, tree.links
    new_socket(tree, 'INPUT', 'NodeSocketColor', "Color")
    new_socket(tree, 'OUTPUT', 'NodeSocketColor', "Color")
    group_input = nodes.new('NodeGroupInput')
    group_output = nodes.new('NodeGroupOutput')

    def vector_math(name, operation, column):
        node = nodes.new('ShaderNodeVectorMath')
        node.name = node.label = name
        node.operation = operation
        node.location = column * 200, 0
        return node

    threshold = vector_math("Raw Threshold", 'MAXIMUM', 1)
    threshold.inputs[1].default_value = (0.0031308,) * 3
    # Gamma is component-wise power too, and is available on the Blender 3.6 floor.
    power = nodes.new('ShaderNodeGamma')
    power.name = power.label = "Raw Power"
    power.location = 400, 0
    power.inputs[1].default_value = 1 / 2.4
    curve = vector_math("Raw Curve", 'MULTIPLY_ADD', 3)
    curve.inputs[1].default_value = (1.055,) * 3
    curve.inputs[2].default_value = (-0.055,) * 3
    below = vector_math("Raw Below Threshold", 'SUBTRACT', 2)
    below.location.y = -200
    encoded = vector_math("Raw Color", 'MULTIPLY_ADD', 4)
    encoded.inputs[1].default_value = (-12.92,) * 3
    # p(max(c,t)) - 12.92 * (max(c,t) - c), with no final encoding after combines.
    links.new(group_input.outputs["Color"], threshold.inputs[0])
    links.new(threshold.outputs[0], power.inputs[0])
    links.new(power.outputs[0], curve.inputs[0])
    links.new(threshold.outputs[0], below.inputs[0])
    links.new(group_input.outputs["Color"], below.inputs[1])
    links.new(below.outputs[0], encoded.inputs[0])
    links.new(curve.outputs[0], encoded.inputs[2])
    links.new(encoded.outputs[0], group_output.inputs["Color"])
    group_output.location = 1000, 0
    return tree


def build_srgb_decode_tree():
    """Decode raw NIF constants for display without changing stored/exported colors."""
    tree = bpy.data.node_groups.new("MW sRGB Decode", 'ShaderNodeTree')
    nodes, links = tree.nodes, tree.links
    new_socket(tree, 'INPUT', 'NodeSocketColor', "Color")
    new_socket(tree, 'OUTPUT', 'NodeSocketColor', "Color")
    group_input = nodes.new('NodeGroupInput')
    group_output = nodes.new('NodeGroupOutput')

    threshold = nodes.new('ShaderNodeVectorMath')
    threshold.operation = 'MAXIMUM'
    threshold.inputs[1].default_value = (0.04045,) * 3
    shifted = nodes.new('ShaderNodeVectorMath')
    shifted.operation = 'MULTIPLY_ADD'
    shifted.inputs[1].default_value = (1 / 1.055,) * 3
    shifted.inputs[2].default_value = (0.055 / 1.055,) * 3
    power = nodes.new('ShaderNodeGamma')
    power.inputs[1].default_value = 2.4
    below = nodes.new('ShaderNodeVectorMath')
    below.operation = 'SUBTRACT'
    decoded = nodes.new('ShaderNodeVectorMath')
    decoded.operation = 'MULTIPLY_ADD'
    decoded.inputs[1].default_value = (-1 / 12.92,) * 3
    # high(max(c,t)) - (max(c,t) - c) / 12.92, including the linear dark segment.
    links.new(group_input.outputs["Color"], threshold.inputs[0])
    links.new(threshold.outputs[0], shifted.inputs[0])
    links.new(shifted.outputs[0], power.inputs[0])
    links.new(threshold.outputs[0], below.inputs[0])
    links.new(group_input.outputs["Color"], below.inputs[1])
    links.new(below.outputs[0], decoded.inputs[0])
    links.new(power.outputs[0], decoded.inputs[2])
    links.new(decoded.outputs[0], group_output.inputs["Color"])
    for index, node in enumerate(nodes):
        node.location = index * 200, 0
    group_output.location = 1400, 0
    return tree


def new_preview_raw_factor(tree, location):
    """1 while the object previews raw NIF colors, 0 for display colors."""
    attribute = new_node(tree, 'ShaderNodeAttribute', location)
    attribute.name = "PreviewRawColor"
    attribute.label = "Preview Raw"
    attribute.attribute_type = 'OBJECT'
    attribute.attribute_name = "tes3_preview_raw"
    return attribute.outputs["Fac"]


def build_preview_encode_tree():
    """Texture and vertex colors: linear for display, sRGB encoded for raw preview."""
    encode_tree = bpy.data.node_groups.get("MW sRGB Encode") or build_srgb_encode_tree()

    tree = bpy.data.node_groups.new("MW Preview Encode", 'ShaderNodeTree')
    links = tree.links

    new_socket(tree, 'INPUT', 'NodeSocketColor', "Color", default=(1, 1, 1, 1))
    # Usually the same source as Color; only Detail's unlinked neutral differs between modes.
    new_socket(tree, 'INPUT', 'NodeSocketColor', "Raw Color", default=(1, 1, 1, 1))
    new_socket(tree, 'OUTPUT', 'NodeSocketColor', "Color")

    group_input = new_node(tree, 'NodeGroupInput', (0, 0))
    preview_raw = new_preview_raw_factor(tree, (200, 220))
    encoded = new_group_node(tree, encode_tree, (200, -160))

    mix = new_node(tree, 'ShaderNodeMixRGB', (400, 0))
    links.new(preview_raw, mix.inputs[0])
    links.new(group_input.outputs["Color"], mix.inputs[1])
    links.new(group_input.outputs["Raw Color"], encoded.inputs["Color"])
    links.new(encoded.outputs["Color"], mix.inputs[2])

    group_output = new_node(tree, 'NodeGroupOutput', (600, 0))
    links.new(mix.outputs[0], group_output.inputs["Color"])

    hide_unlinked_outputs(tree)
    return tree


def build_preview_decode_tree():
    """Material colors: sRGB decoded for display, the raw NIF constant for raw preview."""
    decode_tree = bpy.data.node_groups.get("MW sRGB Decode") or build_srgb_decode_tree()

    tree = bpy.data.node_groups.new("MW Preview Decode", 'ShaderNodeTree')
    links = tree.links

    new_socket(tree, 'INPUT', 'NodeSocketColor', "Color", default=(1, 1, 1, 1))
    new_socket(tree, 'OUTPUT', 'NodeSocketColor', "Color")

    group_input = new_node(tree, 'NodeGroupInput', (0, 0))
    preview_raw = new_preview_raw_factor(tree, (200, 220))
    decoded = new_group_node(tree, decode_tree, (200, 80))

    mix = new_node(tree, 'ShaderNodeMixRGB', (400, 0))
    links.new(preview_raw, mix.inputs[0])
    links.new(group_input.outputs["Color"], decoded.inputs["Color"])
    links.new(decoded.outputs["Color"], mix.inputs[1])
    links.new(group_input.outputs["Color"], mix.inputs[2])

    group_output = new_node(tree, 'NodeGroupOutput', (600, 0))
    links.new(mix.outputs[0], group_output.inputs["Color"])

    hide_unlinked_outputs(tree)
    return tree


def build_decal_layer_tree():
    """Blends one decal texture over Color by (Decal Alpha * Diffuse Alpha).

    Diffuse alpha is part of the engine's decal blend, not a preview shortcut: landscape textures
    transition on vertex alpha, and NiMaterialColorController can animate one texture into another.
    """
    preview_encode_tree = bpy.data.node_groups.get("MW Preview Encode") or build_preview_encode_tree()

    tree = bpy.data.node_groups.new("MW Decal Layer", 'ShaderNodeTree')
    links = tree.links

    new_socket(tree, 'INPUT', 'NodeSocketColor', "Color", default=(1, 1, 1, 1))
    new_socket(tree, 'INPUT', 'NodeSocketFloat', "Decal Alpha")
    new_socket(tree, 'INPUT', 'NodeSocketColor', "Decal Texture", default=(1, 1, 1, 1))
    new_socket(tree, 'INPUT', 'NodeSocketFloat', "Diffuse Alpha", default=1.0)
    new_socket(tree, 'OUTPUT', 'NodeSocketColor', "Color")

    group_input = new_node(tree, 'NodeGroupInput', (0, 0))

    factor = new_node(tree, 'ShaderNodeMath', (200, 160))
    factor.operation = 'MULTIPLY'
    links.new(group_input.outputs["Decal Alpha"], factor.inputs[0])
    links.new(group_input.outputs["Diffuse Alpha"], factor.inputs[1])

    decal = new_group_node(tree, preview_encode_tree, (200, -200), "Decal Texture")
    decal.hide = True
    links.new(group_input.outputs["Decal Texture"], decal.inputs["Color"])
    links.new(group_input.outputs["Decal Texture"], decal.inputs["Raw Color"])

    mix = new_node(tree, 'ShaderNodeMix', (400, 0))
    mix.data_type, mix.blend_type = 'RGBA', 'MIX'
    links.new(factor.outputs[0], mix.inputs[0])  # Factor
    links.new(group_input.outputs["Color"], mix.inputs[6])  # A
    links.new(decal.outputs["Color"], mix.inputs[7])  # B

    group_output = new_node(tree, 'NodeGroupOutput', (600, 0))
    links.new(mix.outputs[2], group_output.inputs["Color"])  # Result(Color)

    hide_unlinked_outputs(tree)
    return tree


def build_alpha_test_tree():
    """NiAlphaProperty's alpha test: 1 where Alpha passes Function against Threshold, else 0.

    AlphaTestFunction mirrors about its middle: function 7 - N is the negation of function N.
    So only ALWAYS, LESS, EQUAL and LESS_EQUAL are selected between, then negated for 4-7.
    """
    tree = bpy.data.node_groups.new("MW Alpha Test", 'ShaderNodeTree')
    links = tree.links

    new_socket(tree, 'INPUT', 'NodeSocketFloat', "Alpha", default=1.0)
    new_socket(tree, 'INPUT', 'NodeSocketFloat', "Enabled")
    new_socket(tree, 'INPUT', 'NodeSocketFloat', "Threshold", default=133 / 255)
    new_socket(tree, 'INPUT', 'NodeSocketFloat', "Function", default=4.0)
    new_socket(tree, 'OUTPUT', 'NodeSocketFloat', "Pass")

    def math(operation, location, label, hide=False):
        node = new_node(tree, 'ShaderNodeMath', location, label)
        node.operation = operation
        node.hide = hide
        return node

    def switch(location, label):
        node = new_node(tree, 'ShaderNodeMix', location, label)
        node.data_type = 'FLOAT'
        node.clamp_factor = True
        return node

    alpha_inputs = new_node(tree, 'NodeGroupInput', (0, 0))

    less = math('LESS_THAN', (200, 0), "Less")
    equal = math('COMPARE', (200, -170), "Equal")
    equal.inputs[2].default_value = 0.5 / 255
    greater = math('GREATER_THAN', (200, -360), "Greater")
    for predicate in (less, equal, greater):
        links.new(alpha_inputs.outputs["Alpha"], predicate.inputs[0])
        links.new(alpha_inputs.outputs["Threshold"], predicate.inputs[1])

    less_equal = math('SUBTRACT', (400, -360), "Less Equal")
    less_equal.inputs[0].default_value = 1.0
    links.new(greater.outputs[0], less_equal.inputs[1])

    function_inputs = new_node(tree, 'NodeGroupInput', (400, 300))

    # Start from ALWAYS, then each step replaces the result for a narrower band of functions.
    passed = None
    for step, predicate in enumerate((less, equal, less_equal), 1):
        x = 400 + step * 200

        in_band = math('COMPARE', (x, 10 + step * 50), f"Function {step}-{7 - step}", hide=True)
        in_band.inputs[1].default_value = 3.5
        in_band.inputs[2].default_value = 4 - step
        links.new(function_inputs.outputs["Function"], in_band.inputs[0])

        select = switch((x + 200, 40 - step * 40), f"Select {predicate.label}")
        select.inputs[2].default_value = 1.0
        links.new(in_band.outputs[0], select.inputs[0])
        if passed is not None:
            links.new(passed, select.inputs[2])
        links.new(predicate.outputs[0], select.inputs[3])
        passed = select.outputs[0]

    upper_half = math('GREATER_THAN', (1200, 210), "Function 4-7", hide=True)
    upper_half.inputs[1].default_value = 3.5
    links.new(function_inputs.outputs["Function"], upper_half.inputs[0])

    negated = math('SUBTRACT', (1400, -230), "Negated", hide=True)
    negated.inputs[0].default_value = 1.0
    links.new(passed, negated.inputs[1])

    negate = switch((1600, -80), "Negate 4-7")
    links.new(upper_half.outputs[0], negate.inputs[0])
    links.new(passed, negate.inputs[2])
    links.new(negated.outputs[0], negate.inputs[3])

    enabled_input = new_node(tree, 'NodeGroupInput', (1600, 20))

    test_pass = switch((1800, -80), "Enabled")
    test_pass.inputs[2].default_value = 1.0
    links.new(enabled_input.outputs["Enabled"], test_pass.inputs[0])
    links.new(negate.outputs[0], test_pass.inputs[3])

    group_output = new_node(tree, 'NodeGroupOutput', (2000, -80))
    links.new(test_pass.outputs[0], group_output.inputs["Pass"])

    hide_unlinked_outputs(tree)
    return tree


def build_blend_mode_tree():
    """Composites Shader over the background as NiAlphaProperty's blending and testing would."""
    tree = bpy.data.node_groups.new("MW Blend Mode", 'ShaderNodeTree')
    links = tree.links

    new_socket(tree, 'INPUT', 'NodeSocketShader', "Shader")
    new_socket(tree, 'INPUT', 'NodeSocketFloat', "Opacity", default=1.0)
    new_socket(tree, 'INPUT', 'NodeSocketFloat', "Test Pass", default=1.0)
    new_socket(tree, 'INPUT', 'NodeSocketFloat', "Blend Enabled")
    new_socket(tree, 'INPUT', 'NodeSocketFloat', "Additive")
    new_socket(tree, 'OUTPUT', 'NodeSocketShader', "Shader")

    def group_input(name, location):
        return new_input_portal(tree, name, location)

    def background(location):
        """Whatever is behind the surface, passed through untouched."""
        node = new_node(tree, 'ShaderNodeBsdfTransparent', location, "Background")
        node.inputs["Color"].default_value = (1, 1, 1, 1)
        return node.outputs[0]

    def mix_shader(location, label, factor, shader1, shader2):
        node = new_node(tree, 'ShaderNodeMixShader', location, label)
        links.new(factor, node.inputs[0])
        links.new(shader1, node.inputs[1])
        links.new(shader2, node.inputs[2])
        return node.outputs[0]

    # Standard: Shader * Opacity + Background * (1 - Opacity)
    standard = mix_shader(
        (200, 0),
        "Standard",
        group_input("Opacity", (0, 20)),
        background((0, -30)),
        group_input("Shader", (0, -130)),
    )

    # Additive: Shader * Opacity + Background
    # An unlinked shader input contributes nothing, so mixing against it scales Shader by Opacity.
    scaled = new_node(tree, 'ShaderNodeMixShader', (200, -300), "Scaled")
    links.new(group_input("Opacity", (0, -340)), scaled.inputs[0])
    links.new(group_input("Shader", (0, -390)), scaled.inputs[2])

    additive = new_node(tree, 'ShaderNodeAddShader', (400, -300), "Additive")
    links.new(scaled.outputs[0], additive.inputs[0])
    links.new(background((200, -460)), additive.inputs[1])

    blended = mix_shader(
        (600, -150),
        "Blend Mode",
        group_input("Additive", (400, 40)),
        standard,
        additive.outputs[0],
    )
    surface = mix_shader(
        (800, -150),
        "Blend Enabled",
        group_input("Blend Enabled", (600, 20)),
        group_input("Shader", (600, -30)),
        blended,
    )
    # A failed alpha test leaves only the background.
    tested = mix_shader(
        (1000, -150),
        "Alpha Test",
        group_input("Test Pass", (800, 60)),
        background((800, -10)),
        surface,
    )

    group_output = new_node(tree, 'NodeGroupOutput', (1200, -150))
    links.new(tested, group_output.inputs["Shader"])

    hide_unlinked_outputs(tree)
    return tree


def build_mw_shader_tree():
    """Procedural equivalent of the old assets/shader.blend 'MW Shader' node group.

    Reads left to right: Textures -> Decals -> Diffuse / Emission -> Surface -> Blending, with
    Alpha underneath. Each section has its own Group Input showing only the sockets it uses.
    """
    preview_encode_tree = bpy.data.node_groups.get("MW Preview Encode") or build_preview_encode_tree()
    preview_decode_tree = bpy.data.node_groups.get("MW Preview Decode") or build_preview_decode_tree()
    decal_layer_tree = bpy.data.node_groups.get("MW Decal Layer") or build_decal_layer_tree()
    alpha_test_tree = bpy.data.node_groups.get("MW Alpha Test") or build_alpha_test_tree()
    blend_mode_tree = bpy.data.node_groups.get("MW Blend Mode") or build_blend_mode_tree()

    tree = bpy.data.node_groups.new("MW Shader", 'ShaderNodeTree')
    links = tree.links

    new_socket(tree, 'INPUT', 'NodeSocketColor', "Base Texture", default=(1, 1, 1, 1))
    new_socket(tree, 'INPUT', 'NodeSocketColor', "Detail Texture", default=(0.5, 0.5, 0.5, 1))
    new_socket(tree, 'INPUT', 'NodeSocketColor', "Dark Texture", default=(1, 1, 1, 1))
    new_socket(tree, 'INPUT', 'NodeSocketFloatFactor', "Decal 0 Alpha")
    new_socket(tree, 'INPUT', 'NodeSocketColor', "Decal 0 Texture", default=(1, 1, 1, 1))
    new_socket(tree, 'INPUT', 'NodeSocketFloatFactor', "Decal 1 Alpha")
    new_socket(tree, 'INPUT', 'NodeSocketColor', "Decal 1 Texture", default=(1, 1, 1, 1))
    new_socket(tree, 'INPUT', 'NodeSocketFloatFactor', "Decal 2 Alpha")
    new_socket(tree, 'INPUT', 'NodeSocketColor', "Decal 2 Texture", default=(1, 1, 1, 1))
    new_socket(tree, 'INPUT', 'NodeSocketFloatFactor', "Decal 3 Alpha")
    new_socket(tree, 'INPUT', 'NodeSocketColor', "Decal 3 Texture", default=(1, 1, 1, 1))
    new_socket(tree, 'INPUT', 'NodeSocketFloatFactor', "Diffuse Alpha", default=1.0, min_value=0.0, max_value=1.0)
    new_socket(tree, 'INPUT', 'NodeSocketColor', "Ambient Color", default=(1, 1, 1, 1))
    new_socket(tree, 'INPUT', 'NodeSocketColor', "Diffuse Color", default=(1, 1, 1, 1))
    new_socket(tree, 'INPUT', 'NodeSocketColor', "Specular Color", default=(0, 0, 0, 1))
    new_socket(tree, 'INPUT', 'NodeSocketColor', "Emissive Color", default=(0, 0, 0, 1))
    new_socket(tree, 'INPUT', 'NodeSocketColor', "Glow Texture", default=(0, 0, 0, 1))
    # Keep this socket at the old Alpha Factor index. Blender preserves group-node
    # links/defaults by interface index when a material migrates to a new node tree.
    new_socket(tree, 'INPUT', 'NodeSocketFloat', "Alpha Blend Enabled")
    new_socket(tree, 'INPUT', 'NodeSocketFloat', "Base Alpha", default=1.0)
    new_socket(tree, 'INPUT', 'NodeSocketFloat', "Dark Alpha", default=1.0)
    new_socket(tree, 'INPUT', 'NodeSocketFloat', "Alpha Clip Enabled")
    new_socket(
        tree,
        'INPUT',
        'NodeSocketFloatFactor',
        "Alpha Clip Threshold",
        default=133 / 255,
        min_value=0.0,
        max_value=1.0,
    )
    # New sockets must remain after the legacy interface prefix; see the index note above.
    new_socket(tree, 'INPUT', 'NodeSocketFloat', "Additive Blend Enabled")
    new_socket(tree, 'INPUT', 'NodeSocketFloat', "Alpha Test Function", default=4.0)
    # Detail's unlinked linear 0.5 is a synthetic neutral, not a decoded texture sample.
    # A separate raw-mode source keeps it neutral without changing old defaults or indices.
    neutral_detail = ((0.5 + 0.055) / 1.055) ** 2.4
    new_socket(tree, 'INPUT', 'NodeSocketColor', "Raw Detail Texture", default=(neutral_detail,) * 3 + (1,))
    # Material constants remain raw NIF values on their original sockets (including
    # animation targets). Vertex colors are already linear and have their own source.
    new_socket(tree, 'INPUT', 'NodeSocketColor', "Vertex Color", default=(1, 1, 1, 1))
    new_socket(tree, 'INPUT', 'NodeSocketFloat', "Use Vertex Colors")

    def texture_color(inputs, name, location, raw_name=None):
        """Texture and vertex colors are linear here; raw preview shows them sRGB encoded."""
        node = new_group_node(tree, preview_encode_tree, location, name)
        node.hide = True
        links.new(inputs.outputs[name], node.inputs["Color"])
        links.new(inputs.outputs[raw_name or name], node.inputs["Raw Color"])
        return node.outputs["Color"]

    def material_color(inputs, name, location):
        """Material constants are raw NIF values; display shows them sRGB decoded."""
        node = new_group_node(tree, preview_decode_tree, location, name)
        node.hide = True
        links.new(inputs.outputs[name], node.inputs["Color"])
        return node.outputs["Color"]

    def vector_math(operation, location):
        node = new_node(tree, 'ShaderNodeVectorMath', location)
        node.operation = operation
        return node

    with framed(tree, "Textures"):
        inputs = new_node(tree, 'NodeGroupInput', (0, 0))
        base = texture_color(inputs, "Base Texture", (200, 5))
        detail = texture_color(inputs, "Detail Texture", (200, -40), raw_name="Raw Detail Texture")
        dark = texture_color(inputs, "Dark Texture", (600, -190))

        # Base, Detail, Dark: (Base * Detail * 2) * Dark
        base_detail = vector_math('MULTIPLY', (400, 0))
        links.new(base, base_detail.inputs[0])
        links.new(detail, base_detail.inputs[1])

        base_detail_x2 = vector_math('SCALE', (600, 0))
        base_detail_x2.inputs["Scale"].default_value = 2.0
        links.new(base_detail.outputs[0], base_detail_x2.inputs[0])

        base_combined = vector_math('MULTIPLY', (800, 0))
        links.new(base_detail_x2.outputs[0], base_combined.inputs[0])
        links.new(dark, base_combined.inputs[1])

    with framed(tree, "Decals"):
        # Decal: 4x shared Decal Layer subgroup, chained
        color = base_combined.outputs[0]
        for i in range(4):
            inputs = new_node(tree, 'NodeGroupInput', (1100 + i * 200, -220))
            layer = new_group_node(tree, decal_layer_tree, (1300 + i * 200, 0), f"Decal {i}")
            links.new(color, layer.inputs["Color"])
            links.new(inputs.outputs[f"Decal {i} Alpha"], layer.inputs["Decal Alpha"])
            links.new(inputs.outputs[f"Decal {i} Texture"], layer.inputs["Decal Texture"])
            links.new(inputs.outputs["Diffuse Alpha"], layer.inputs["Diffuse Alpha"])
            color = layer.outputs[0]

    with framed(tree, "Diffuse"):
        inputs = new_node(tree, 'NodeGroupInput', (2200, 340))

        # V is either decoded material diffuse or the separately converted vertex color.
        vertex_or_material = new_node(tree, 'ShaderNodeMixRGB', (2600, 340), "Diffuse Source")
        links.new(inputs.outputs["Use Vertex Colors"], vertex_or_material.inputs[0])
        links.new(material_color(inputs, "Diffuse Color", (2400, 340)), vertex_or_material.inputs[1])
        links.new(texture_color(inputs, "Vertex Color", (2400, 295)), vertex_or_material.inputs[2])

        # Diffuse = T × V, under Blender lighting.
        diffuse_color = vector_math('MULTIPLY', (2800, 160))
        links.new(vertex_or_material.outputs[0], diffuse_color.inputs[0])
        links.new(color, diffuse_color.inputs[1])

        diffuse_bsdf = new_node(tree, 'ShaderNodeBsdfDiffuse', (3000, 160))
        links.new(diffuse_color.outputs[0], diffuse_bsdf.inputs["Color"])

    with framed(tree, "Emission"):
        inputs = new_node(tree, 'NodeGroupInput', (2200, -240))

        # Self-illumination = T × E + G, keeping emissive RGB and texture modulation.
        emissive_color = vector_math('MULTIPLY', (2800, -160))
        links.new(color, emissive_color.inputs[0])
        links.new(material_color(inputs, "Emissive Color", (2400, -255)), emissive_color.inputs[1])

        glow = vector_math('ADD', (3000, -160))
        links.new(emissive_color.outputs[0], glow.inputs[0])
        links.new(texture_color(inputs, "Glow Texture", (2800, -330)), glow.inputs[1])

        emission = new_node(tree, 'ShaderNodeEmission', (3200, -160))
        links.new(glow.outputs[0], emission.inputs["Color"])

        # Game emission is visible, but does not become a light source for other objects.
        gate = new_node(tree, 'ShaderNodeLightPath', (3000, -330), "Emission Gate")
        links.new(gate.outputs["Is Camera Ray"], emission.inputs["Strength"])

    surface = new_node(tree, 'ShaderNodeAddShader', (3400, 20), "Surface")
    links.new(diffuse_bsdf.outputs[0], surface.inputs[0])
    links.new(emission.outputs[0], surface.inputs[1])

    with framed(tree, "Alpha"):
        # Alpha: opacity = Base Alpha * Dark Alpha * Diffuse Alpha
        inputs = new_node(tree, 'NodeGroupInput', (2800, -520))
        base_dark_alpha = new_node(tree, 'ShaderNodeMath', (3000, -520))
        base_dark_alpha.operation = 'MULTIPLY'
        links.new(inputs.outputs["Base Alpha"], base_dark_alpha.inputs[0])
        links.new(inputs.outputs["Dark Alpha"], base_dark_alpha.inputs[1])

        inputs = new_node(tree, 'NodeGroupInput', (3000, -720))
        opacity = new_node(tree, 'ShaderNodeMath', (3200, -520), "Opacity")
        opacity.operation = 'MULTIPLY'
        links.new(base_dark_alpha.outputs[0], opacity.inputs[0])
        links.new(inputs.outputs["Diffuse Alpha"], opacity.inputs[1])

        alpha_test = new_group_node(tree, alpha_test_tree, (3400, -640), "Alpha Test")
        links.new(opacity.outputs[0], alpha_test.inputs["Alpha"])
        links.new(inputs.outputs["Alpha Clip Enabled"], alpha_test.inputs["Enabled"])
        links.new(inputs.outputs["Alpha Clip Threshold"], alpha_test.inputs["Threshold"])
        links.new(inputs.outputs["Alpha Test Function"], alpha_test.inputs["Function"])

    with framed(tree, "Blending"):
        inputs = new_node(tree, 'NodeGroupInput', (3660, -700))
        inputs.width = 180  # "Additive Blend Enabled"
        blend = new_group_node(tree, blend_mode_tree, (3900, -420), "Blend Mode")
        links.new(surface.outputs[0], blend.inputs["Shader"])
        links.new(opacity.outputs[0], blend.inputs["Opacity"])
        links.new(alpha_test.outputs[0], blend.inputs["Test Pass"])
        links.new(inputs.outputs["Alpha Blend Enabled"], blend.inputs["Blend Enabled"])
        links.new(inputs.outputs["Additive Blend Enabled"], blend.inputs["Additive"])

        material_output = new_node(tree, 'ShaderNodeOutputMaterial', (4100, -420))
        links.new(blend.outputs[0], material_output.inputs[0])

    hide_unlinked_outputs(tree)
    return tree


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
    vertex_input = property(lambda self: self.inputs["Vertex Color"])
    specular_input = property(lambda self: self.inputs["Specular Color"])
    emissive_input = property(lambda self: self.inputs["Emissive Color"])

    alpha_input = property(lambda self: self.inputs["Base Alpha"])
    opacity_input = property(lambda self: self.inputs["Diffuse Alpha"])
    alpha_blend_enabled_input = property(lambda self: self.inputs["Alpha Blend Enabled"])
    additive_blend_enabled_input = property(lambda self: self.inputs["Additive Blend Enabled"])
    alpha_clip_enabled_input = property(lambda self: self.inputs["Alpha Clip Enabled"])
    alpha_threshold_input = property(lambda self: self.inputs["Alpha Clip Threshold"])
    alpha_test_function_input = property(lambda self: self.inputs["Alpha Test Function"])

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
    def alpha_blend_enabled(self):
        return self.alpha_blend_enabled_input.default_value

    @alpha_blend_enabled.setter
    def alpha_blend_enabled(self, value):
        self.alpha_blend_enabled_input.default_value = value

    @property
    def additive_blend_enabled(self):
        return self.additive_blend_enabled_input.default_value

    @additive_blend_enabled.setter
    def additive_blend_enabled(self, value):
        self.additive_blend_enabled_input.default_value = value

    @property
    def alpha_threshold(self):
        return self.alpha_threshold_input.default_value

    @alpha_threshold.setter
    def alpha_threshold(self, value):
        self.alpha_threshold_input.default_value = value

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
        group.node_tree = get_current_mw_shader_tree()

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
