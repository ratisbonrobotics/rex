import jax
import jax.numpy as jnp
from jax import jit, vmap
from PIL import Image
from functools import partial
import numpy as np

def parse_obj_file(file_path):
    vertices = []
    texture_coords = []
    faces = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                vertex = [float(coord) for coord in line.split()[1:]]
                vertices.append(vertex)
            elif line.startswith('vt '):
                texture_coord = [float(coord) for coord in line.split()[1:]]
                texture_coords.append(texture_coord)
            elif line.startswith('f '):
                face = []
                for vertex_index in line.split()[1:]:
                    indices = vertex_index.split('/')
                    vertex_index = int(indices[0]) - 1
                    texture_index = int(indices[1]) - 1 if len(indices) > 1 else None
                    face.append((vertex_index, texture_index))
                faces.append(face)

    return jnp.array(vertices), jnp.array(texture_coords), jnp.array(faces)

@partial(jit, static_argnums=(7, 8))
def render_triangle(v0, v1, v2, vt0, vt1, vt2, texture, width, height):
    # Convert vertex coordinates to screen space
    v0 = jnp.array([(v0[0] + 1) * width / 2, (1 - v0[1]) * height / 2, v0[2]])
    v1 = jnp.array([(v1[0] + 1) * width / 2, (1 - v1[1]) * height / 2, v1[2]])
    v2 = jnp.array([(v2[0] + 1) * width / 2, (1 - v2[1]) * height / 2, v2[2]])

    # Create a grid of all pixels
    x = jnp.arange(width)
    y = jnp.arange(height)
    xx, yy = jnp.meshgrid(x, y)
    pixels = jnp.stack([xx, yy], axis=-1)

    def process_pixel(pixel):
        # Inlined barycentric_coordinates
        v0v1 = v1[:2] - v0[:2]
        v0v2 = v2[:2] - v0[:2]
        v0p = pixel - v0[:2]
        d00 = jnp.dot(v0v1, v0v1)
        d01 = jnp.dot(v0v1, v0v2)
        d11 = jnp.dot(v0v2, v0v2)
        d20 = jnp.dot(v0p, v0v1)
        d21 = jnp.dot(v0p, v0v2)
        denom = d00 * d11 - d01 * d01
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        barycentric = jnp.array([u, v, w])

        # Inlined is_inside_triangle
        inside = jnp.all(barycentric >= 0) & jnp.all(barycentric <= 1)

        depth = jnp.dot(barycentric, jnp.array([v0[2], v1[2], v2[2]]))
        tx = jnp.dot(barycentric, jnp.array([vt0[0], vt1[0], vt2[0]]))
        ty = jnp.dot(barycentric, jnp.array([vt0[1], vt1[1], vt2[1]]))

        # Inlined sample_texture
        tx = jnp.clip(tx * texture.shape[1], 0, texture.shape[1] - 1).astype(jnp.int32)
        ty = jnp.clip((1 - ty) * texture.shape[0], 0, texture.shape[0] - 1).astype(jnp.int32)
        color = texture[ty, tx]

        return jnp.where(inside, color, jnp.array([0, 0, 0])), jnp.where(inside, depth, -jnp.inf)

    colors, depths = vmap(vmap(process_pixel))(pixels)
    return colors, depths

@jit
def update_buffers(image, depth_buffer, colors, depths):
    mask = depths > depth_buffer
    new_image = jnp.where(mask[..., None], colors, image)
    new_depth_buffer = jnp.where(mask, depths, depth_buffer)
    return new_image, new_depth_buffer

@partial(jit, static_argnums=(4, 5))
def render_triangles(vertices, texture_coords, faces, texture, width, height):
    image = jnp.zeros((height, width, 3), dtype=jnp.float32)
    depth_buffer = jnp.full((height, width), -jnp.inf)

    def render_face(carry, face):
        image, depth_buffer = carry
        v0, v1, v2 = vertices[face[:, 0]]
        vt0, vt1, vt2 = texture_coords[face[:, 1]]

        colors, depths = render_triangle(v0, v1, v2, vt0, vt1, vt2, texture, width, height)
        new_image, new_depth_buffer = update_buffers(image, depth_buffer, colors, depths)
        return (new_image, new_depth_buffer), None

    (final_image, _), _ = jax.lax.scan(render_face, (image, depth_buffer), faces)
    return final_image

def main():
    vertices, texture_coords, faces = parse_obj_file('african_head.obj')
    texture = jnp.array(Image.open('african_head_diffuse.tga')).astype(jnp.float32) / 255.0
    
    image = render_triangles(vertices, texture_coords, faces, texture, 800, 600)
    
    # Convert the JAX array to a NumPy array, then to a PIL Image
    numpy_image = (np.array(image) * 255).astype(np.uint8)
    output_image = Image.fromarray(numpy_image)
    output_image.save('output.png')

if __name__ == '__main__':
    main()