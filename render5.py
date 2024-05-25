import re
import jax
import numpy as onp
from PIL import Image
from jax import numpy as jp
from renderer.types import Vec3f, Vec2f
from renderer import Model as RendererMesh
from renderer import ModelObject as Instance
from renderer import LightParameters as Light
from renderer.geometry import rotation_matrix
from renderer import CameraParameters as Camera
from renderer import ShadowParameters as Shadow
from renderer import Renderer, transpose_for_display
from numpngw import write_apng

# Function to create a model from file content and textures
def make_model(fileContent: list[str], diffuse_map, specular_map) -> RendererMesh:
    verts, norms, uv, faces, faces_norm, faces_uv = [], [], [], [], [], []
    _float, _integer, _one_vertex = re.compile(r"(-?\d+\.?\d*(?:e[+-]\d+)?)"), re.compile(r"\d+"), re.compile(r"\d+/\d*/\d*")

    for line in fileContent:
        if line.startswith("v "):
            verts.append(tuple(map(float, _float.findall(line, 2)[:3])))
        elif line.startswith("vn "):
            norms.append(tuple(map(float, _float.findall(line, 2)[:3])))
        elif line.startswith("vt "):
            uv.append(tuple(map(float, _float.findall(line, 2)[:2])))
        elif line.startswith("f "):
            face, face_norm, face_uv = [], [], []
            vertices = _one_vertex.findall(line)
            assert len(vertices) == 3, f"Expected 3 vertices, got {len(vertices)}"
            for vertex in vertices:
                v, vt, vn = list(map(int, _integer.findall(vertex)))
                face.append(v - 1)
                face_norm.append(vn - 1)
                face_uv.append(vt - 1)
            faces.append(face)
            faces_norm.append(face_norm)
            faces_uv.append(face_uv)

    return RendererMesh(
        verts=jp.array(verts),
        norms=jp.array(norms),
        uvs=jp.array(uv),
        faces=jp.array(faces),
        faces_norm=jp.array(faces_norm),
        faces_uv=jp.array(faces_uv),
        diffuse_map=jax.numpy.swapaxes(diffuse_map, 0, 1)[:, ::-1, :],
        specular_map=jax.numpy.swapaxes(specular_map, 0, 1)[:, ::-1],
    )

# Function to load a TGA file
def load_tga(path: str):
    image = Image.open(path)
    width, height = image.size
    buffer = onp.zeros((width, height, 3))
    for y in range(height):
        for x in range(width):
            buffer[y, x] = onp.array(image.getpixel((x, y)))
    return jp.array(buffer, dtype=jp.single)

# Load model and textures
obj_path, texture_path, spec_path = "plant.obj", "diff.tga", "spec.tga"
texture, specular_map = load_tga(texture_path) / 255, load_tga(spec_path)[..., 0]
model = make_model(open(obj_path, 'r').readlines(), texture, specular_map)

canvas_width, canvas_height, frames, rotation_axis = 1920, 1080, 30, "Y"
rotation_axis = dict(X=(1., 0., 0.), Y=(0., 1., 0.), Z=(0., 0., 1.))[rotation_axis]
degrees = jax.lax.iota(float, frames) * 360. / frames

# Function to render instances
@jax.default_matmul_precision("float32")
def render_instances(instances, width, height, camera, light=None, shadow=None, camera_target=None, enable_shadow=True):
    if light is None:
        light = Light(direction=jp.array([0.57735, -0.57735, 0.57735]), ambient=0.1, diffuse=0.85, specular=0.05)
    if shadow is None and enable_shadow:
        assert camera_target is not None, 'camera_target is None'
        shadow = Shadow(centre=camera_target)
    elif not enable_shadow:
        shadow = None

    img = Renderer.get_camera_image(objects=instances, light=light, camera=camera, width=width, height=height, shadow_param=shadow, colour_default=jp.zeros(3, dtype=jp.single))
    return jax.lax.clamp(0., img, 1.)

# Function to generate compiled rendering
def gen_compile_render(instances):
    eye, center, up = jp.array((0, 0, 3.)), jp.array((0, 0, 0)), jp.array((0, 1, 0))
    camera = Camera(viewWidth=canvas_width, viewHeight=canvas_height, position=eye, target=center, up=up)
    
    def render(batched_instances):
        def _render(instances):
            _render = jax.jit(render_instances, static_argnames=("width", "height", "enable_shadow"), inline=True)
            img = _render(instances=instances, width=canvas_width, height=canvas_height, camera=camera, camera_target=center)
            return transpose_for_display((img * 255).astype(jp.uint8))

        return jax.jit(jax.vmap(_render))(batched_instances)

    render_compiled = jax.jit(render).lower(instances).compile()
    def wrap(batched_instances):
        return list(map(onp.asarray, jax.device_get(render_compiled(batched_instances))))
    return wrap

# Rotate model
def rotate(model, rotation_axis, degree):
    instance = Instance(model=model)
    return instance.replace_with_orientation(rotation_matrix=rotation_matrix(rotation_axis, degree))

batch_rotation = jax.jit(jax.vmap(lambda degree: rotate(model, rotation_axis, degree))).lower(degrees).compile()
instances = [batch_rotation(degrees)]
render_with_states = gen_compile_render(instances)
images = render_with_states(instances)

write_apng('animation.png', images, delay=1/30.)
