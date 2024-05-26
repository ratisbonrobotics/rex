import re
import jax
import numpy as onp
from PIL import Image
from jax import numpy as jp
from renderer import LightParameters
from renderer.geometry import rotation_matrix
from renderer import CameraParameters
from renderer import ShadowParameters
from renderer import Renderer, transpose_for_display
from numpngw import write_apng
from typing import NamedTuple, TypeAlias, Optional
from jaxtyping import Array, Float, Integer, Bool
from renderer.geometry import transform_matrix_from_rotation

Vec2f: TypeAlias = Float[Array, "2"]
Vec3f: TypeAlias = Float[Array, "3"]
ModelMatrix: TypeAlias = Float[Array, "4 4"]
Vec4f: TypeAlias = Float[Array, "4"]

BoolV: TypeAlias = Bool[Array, ""]
FaceIndices: TypeAlias = Integer[Array, "faces 3"]
Vertices: TypeAlias = Float[Array, "vertices 3"]
Normals: TypeAlias = Float[Array, "normals 3"]
UVCoordinates: TypeAlias = Float[Array, "uv_counts 2"]
Texture: TypeAlias = Float[Array, "textureWidth textureHeight channel"]
SpecularMap: TypeAlias = Float[Array, "textureWidth textureHeight"]
NormalMap: TypeAlias = Float[Array, "textureWidth textureHeight 3"]

FALSE_ARRAY: BoolV = jax.lax.full((), False, dtype=jax.numpy.bool_)

class Model(NamedTuple):
    verts: Vertices
    norms: Normals
    uvs: UVCoordinates
    faces: FaceIndices
    faces_norm: FaceIndices
    faces_uv: FaceIndices

    diffuse_map: Texture
    specular_map: SpecularMap

    @classmethod
    def create(cls, verts: Vertices, norms: Normals, uvs: UVCoordinates, faces: FaceIndices, diffuse_map: Texture, specular_map: Optional[SpecularMap] = None) -> "Model":
        if specular_map is None:
            specular_map = jax.lax.full(diffuse_map.shape[:2], 2.0)
        return cls(verts=verts, norms=norms, uvs=uvs, faces=faces, faces_norm=faces, faces_uv=faces, diffuse_map=diffuse_map, specular_map=specular_map)

class ModelObject(NamedTuple):
    model: Model
    local_scaling: Vec3f = jax.numpy.ones(3)
    transform: ModelMatrix = jax.numpy.identity(4)
    double_sided: BoolV = FALSE_ARRAY

    def replace_with_position(self, position: Vec3f) -> "ModelObject":
        return self._replace(transform=self.transform.at[:3, 3].set(position))

    def replace_with_orientation(self, orientation: Optional[Vec4f] = None, rotation_matrix: Optional[Float[Array, "3 3"]] = None) -> "ModelObject":
        if rotation_matrix is None:
            if orientation is None:
                orientation = jax.numpy.array((0.0, 0.0, 0.0, 1.0))
            rotation_matrix = transform_matrix_from_rotation(orientation)

        return self._replace(transform=self.transform.at[:3, :3].set(rotation_matrix))

    def replace_with_local_scaling(self, local_scaling: Vec3f) -> "ModelObject":
        return self._replace(local_scaling=local_scaling)

    def replace_with_double_sided(self, double_sided: BoolV) -> "ModelObject":
        return self._replace(double_sided=double_sided)

# Load model and textures
obj_path, texture_path, spec_path = "african_head.obj", "african_head_diffuse.tga", "african_head_spec.tga"
image = Image.open(texture_path)
width, height = image.size
texture = onp.zeros((width, height, 3))
for y in range(height):
    for x in range(width):
        texture[y, x] = onp.array(image.getpixel((x, y)))
texture = jp.array(texture, dtype=jp.single) / 255

image = Image.open(spec_path)
specular_map = onp.zeros((width, height, 3))
for y in range(height):
    for x in range(width):
        specular_map[y, x] = onp.array(image.getpixel((x, y)))
specular_map = jp.array(specular_map, dtype=jp.single)[..., 0]

verts, norms, uv, faces, faces_norm, faces_uv = [], [], [], [], [], []
_float, _integer, _one_vertex = re.compile(r"(-?\d+\.?\d*(?:e[+-]\d+)?)"), re.compile(r"\d+"), re.compile(r"\d+/\d*/\d*")

with open(obj_path, 'r') as file:
    for line in file:
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

model = Model(verts=jp.array(verts), norms=jp.array(norms), uvs=jp.array(uv), faces=jp.array(faces), faces_norm=jp.array(faces_norm), faces_uv=jp.array(faces_uv), diffuse_map=jax.numpy.swapaxes(texture, 0, 1)[:, ::-1, :], specular_map=jax.numpy.swapaxes(specular_map, 0, 1)[:, ::-1])

canvas_width, canvas_height, frames, rotation_axis = 1920, 1080, 30, "Y"
rotation_axis = dict(X=(1., 0., 0.), Y=(0., 1., 0.), Z=(0., 0., 1.))[rotation_axis]
degrees = jax.lax.iota(float, frames) * 360. / frames

eye, center, up = jp.array((0, 0, 3.)), jp.array((0, 0, 0)), jp.array((0, 1, 0))
camera = CameraParameters(viewWidth=canvas_width, viewHeight=canvas_height, position=eye, target=center, up=up)
light = LightParameters(direction=jp.array([0.57735, -0.57735, 0.57735]), ambient=0.1, diffuse=0.85, specular=0.05)
shadow = ShadowParameters(centre=center)

@jax.default_matmul_precision("float32")
def render_instances(instances, width, height, camera, light, shadow):
    img = Renderer.get_camera_image(objects=instances, light=light, camera=camera, width=width, height=height, shadow_param=shadow, colour_default=jp.zeros(3, dtype=jp.single))
    return jax.lax.clamp(0., img, 1.)

def rotate(model, rotation_axis, degree):
    instance = ModelObject(model=model)
    return instance.replace_with_orientation(rotation_matrix=rotation_matrix(rotation_axis, degree))

batch_rotation = jax.jit(jax.vmap(lambda degree: rotate(model, rotation_axis, degree))).lower(degrees).compile()
instances = [batch_rotation(degrees)]

@jax.jit
def render(batched_instances):
    def _render(instances):
        _render = jax.jit(render_instances, static_argnames=("width", "height"), inline=True)
        img = _render(instances=instances, width=canvas_width, height=canvas_height, camera=camera, light=light, shadow=shadow)
        return transpose_for_display((img * 255).astype(jp.uint8))

    return jax.jit(jax.vmap(_render))(batched_instances)

render_compiled = jax.jit(render).lower(instances).compile()
images = list(map(onp.asarray, jax.device_get(render_compiled(instances))))

write_apng('animation.png', images, delay=1/30.)

# ffmpeg -i animation.png intermediate.gif
# gifsicle --optimize=3 --delay=5 intermediate.gif > output.gif