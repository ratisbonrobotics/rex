import functools
from typing import Iterable, NamedTuple, Optional

import jax
from jax import numpy as jp
import numpy as onp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from PIL import Image

from renderer import CameraParameters as Camera
from renderer import LightParameters as Light
from renderer import Model as RendererMesh
from renderer import ModelObject as Instance
from renderer import ShadowParameters as Shadow
from renderer import Renderer, merge_objects, transpose_for_display
from renderer.geometry import rotation_matrix


import re

from renderer.types import Vec3f, Vec2f

#@title ## Download and Load Models
def make_model(fileContent: list[str], diffuse_map, specular_map) -> RendererMesh:
    verts: list[Vec3f] = []
    norms: list[Vec3f] = []
    uv: list[Vec2f] = []
    faces: list[list[int]] = []
    faces_norm: list[list[int]] = []
    faces_uv: list[list[int]] = []

    _float = re.compile(r"(-?\d+\.?\d*(?:e[+-]\d+)?)")
    _integer = re.compile(r"\d+")
    _one_vertex = re.compile(r"\d+/\d*/\d*")
    for line in fileContent:
        if line.startswith("v "):
            vert: Vec3f = tuple(map(float, _float.findall(line, 2)[:3]))
            verts.append(vert)
        elif line.startswith("vn "):
            norm: Vec3f = tuple(map(float, _float.findall(line, 2)[:3]))
            norms.append(norm)
        elif line.startswith("vt "):
            uv_coord: Vec2f = tuple(map(float, _float.findall(line, 2)[:2]))
            uv.append(uv_coord)
        elif line.startswith("f "):
            face: list[int] = []
            face_norm: list[int] = []
            face_uv: list[int] = []

            vertices: list[str] = _one_vertex.findall(line)
            assert len(vertices) == 3, ("Expected 3 vertices, "
                                        f"(got {len(vertices)}")
            for vertex in _one_vertex.findall(line):
                indices: list[int] = list(map(int, _integer.findall(vertex)))
                assert len(indices) == 3, ("Expected 3 indices (v/vt/vn), "
                                           f"got {len(indices)}")
                v, vt, vn = indices
                # indexed from 1 in Wavefront Obj
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

def load_tga(path: str):
    image: Image.Image = Image.open(path)
    width, height = image.size
    buffer = onp.zeros((width, height, 3))

    for y in range(height):
        for x in range(width):
            buffer[y, x] = onp.array(image.getpixel((x, y)))

    texture = jp.array(buffer, dtype=jp.single)

    return texture

obj_path: str = "african_head.obj"
texture_path: str = "african_head_diffuse.tga"
spec_path: str = "african_head_spec.tga"
normal_tangent_path: str = "african_head_nm_tangent.tga"

texture = load_tga(texture_path) / 255
specular_map = load_tga(spec_path)[..., 0]
model: RendererMesh = make_model(open(obj_path, 'r').readlines(), texture, specular_map)

canvas_width: int = 1920 #@param {type:"integer"}
canvas_height: int = 1080 #@param {type:"integer"}
frames: int = 30 #@param {type:"slider", min:1, max:32, step:1}
rotation_axis = "Y" #@param ["X", "Y", "Z"]
rotation_degrees: float = 360.

rotation_axis = dict(
    X=(1., 0., 0.),
    Y=(0., 1., 0.),
    Z=(0., 0., 1.),
)[rotation_axis]

degrees = jax.lax.iota(float, frames) * rotation_degrees / frames

@jax.default_matmul_precision("float32")
def render_instances(
  instances: list[Instance],
  width: int,
  height: int,
  camera: Camera,
  light: Optional[Light] = None,
  shadow: Optional[Shadow] = None,
  camera_target: Optional[jp.ndarray] = None,
  enable_shadow: bool = True,
) -> jp.ndarray:
  """Renders an RGB array of sequence of instances.

  Rendered result is not transposed with `transpose_for_display`; it is in
  floating numbers in [0, 1], not `uint8` in [0, 255].
  """
  if light is None:
    direction = jp.array([0.57735, -0.57735, 0.57735])
    light = Light(
        direction=direction,
        ambient=0.1,
        diffuse=0.85,
        specular=0.05,
    )
  if shadow is None and enable_shadow:
    assert camera_target is not None, 'camera_target is None'
    shadow = Shadow(centre=camera_target)
  elif not enable_shadow:
    shadow = None

  img = Renderer.get_camera_image(
    objects=instances,
    light=light,
    camera=camera,
    width=width,
    height=height,
    shadow_param=shadow,
    colour_default=jp.zeros(3, dtype=jp.single),
  )
  arr = jax.lax.clamp(0., img, 1.)

  return arr

def gen_compile_render(instances):
  """Return batched states."""
  # parameters
  # camera
  eye = jp.array((0, 0, 3.))
  center = jp.array((0, 0, 0))
  up = jp.array((0, 1, 0))
  camera: Camera = Camera(viewWidth=canvas_width, viewHeight=canvas_height, position=eye, target=center, up=up)


  def render(batched_instances) -> jp.ndarray:
    def _render(instances) -> jp.ndarray:
      _render = jax.jit(
        render_instances,
        static_argnames=("width", "height", "enable_shadow"),
        inline=True,
      )
      img = _render(instances=instances, width=canvas_width, height=canvas_height, camera=camera, camera_target=center)
      arr = transpose_for_display((img * 255).astype(jp.uint8))

      return arr

    # render
    _render_batch = jax.jit(jax.vmap(_render))
    images = _render_batch(batched_instances)

    return images

  def copy_back_images(images: jp.ndarray) -> list[onp.ndarray]:
    # copy back
    images_in_device = jax.device_get(images)

    frames: list[onp.ndarray] = list(map(onp.asarray, images_in_device))

    return frames


  render_compiled = jax.jit(render).lower(instances).compile()

  def wrap(batched_instances) -> list[onp.ndarray]:
    images = render_compiled(batched_instances)

    return copy_back_images(images)

  return wrap

def rotate(model: RendererMesh, rotation_axis: tuple[float, float, float], degree: float) -> Instance:
  instance = Instance(model=model)
  instance = instance.replace_with_orientation(rotation_matrix=rotation_matrix(rotation_axis, degree))

  return instance

gen_model = lambda degree: rotate(model, rotation_axis, degree)
print("Compile time")
batch_rotation = jax.jit(jax.vmap(gen_model)).lower(degrees).compile()

print("\nExecution time")
instances = [batch_rotation(degrees)]

print("Compile and generate rendering function")
render_with_states = gen_compile_render(instances)
print()
print("Render time")
images = render_with_states(instances)

from numpngw import write_apng
write_apng('animation.png', images, delay=1/30.)