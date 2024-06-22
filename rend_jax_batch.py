import jax
import jax.numpy as jnp
from jax import vmap, jit
from functools import partial
from PIL import Image
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

@partial(jit, static_argnums=(3, 4))
def render_triangle(vertices, texture_coords, face, width, height):
    v0, v1, v2 = [vertices[i] for i, _ in face]
    vt0, vt1, vt2 = [texture_coords[i] for _, i in face if i is not None]

    # Convert vertex coordinates to screen space
    v0 = ((v0[0] + 1) * width / 2, (1 - v0[1]) * height / 2, v0[2])
    v1 = ((v1[0] + 1) * width / 2, (1 - v1[1]) * height / 2, v1[2])
    v2 = ((v2[0] + 1) * width / 2, (1 - v2[1]) * height / 2, v2[2])

    # Calculate edge function coefficients
    e01 = v1[1] - v0[1], v0[0] - v1[0], v1[0] * v0[1] - v0[0] * v1[1]
    e12 = v2[1] - v1[1], v1[0] - v2[0], v2[0] * v1[1] - v1[0] * v2[1]
    e20 = v0[1] - v2[1], v2[0] - v0[0], v0[0] * v2[1] - v2[0] * v0[1]

    # Generate x and y coordinates using meshgrid
    x, y = jnp.meshgrid(jnp.arange(width), jnp.arange(height))

    # Calculate barycentric coordinates
    w0 = e12[0] * (x + 0.5) + e12[1] * (y + 0.5) + e12[2]
    w1 = e20[0] * (x + 0.5) + e20[1] * (y + 0.5) + e20[2]
    w2 = e01[0] * (x + 0.5) + e01[1] * (y + 0.5) + e01[2]

    # Create a mask for pixels inside the triangle
    mask = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)

    # Calculate depth and texture coordinates
    w = w0 + w1 + w2
    w0, w1, w2 = w0 / w, w1 / w, w2 / w
    depth = w0 * v0[2] + w1 * v1[2] + w2 * v2[2]
    tx = w0 * vt0[0] + w1 * vt1[0] + w2 * vt2[0]
    ty = 1 - (w0 * vt0[1] + w1 * vt1[1] + w2 * vt2[1])

    return mask, depth, tx, ty

@partial(jit, static_argnums=(3, 4))
def render_triangles_batch(vertices, texture_coords, faces, width, height):
    render_triangle_vmap = vmap(render_triangle, in_axes=(None, None, 0, None, None))
    mask, depth, tx, ty = render_triangle_vmap(vertices, texture_coords, faces, width, height)

    def combine_triangles(carry, x):
        a_mask, a_depth, a_tx, a_ty = carry
        b_mask, b_depth, b_tx, b_ty = x
        
        update_mask = b_mask & (b_depth > a_depth)
        new_mask = a_mask | b_mask
        new_depth = jnp.where(update_mask, b_depth, a_depth)
        new_tx = jnp.where(update_mask, b_tx, a_tx)
        new_ty = jnp.where(update_mask, b_ty, a_ty)
        
        return (new_mask, new_depth, new_tx, new_ty), None

    initial_values = (
        jnp.zeros((height, width), dtype=bool),
        jnp.full((height, width), -jnp.inf),
        jnp.zeros((height, width)),
        jnp.zeros((height, width))
    )

    (final_mask, final_depth, final_tx, final_ty), _ = jax.lax.scan(
        combine_triangles,
        initial_values,
        (mask, depth, tx, ty)
    )

    return final_mask, final_depth, final_tx, final_ty

@partial(jit, static_argnums=(3, 4, 5))
def render_model(vertices, texture_coords, faces, width, height, batch_size, texture):
    num_batches = (faces.shape[0] + batch_size - 1) // batch_size
    
    def render_and_update(buffers, i):
        depth_buffer, update_mask, tx_buffer, ty_buffer = buffers
        start = i * batch_size
        batch_faces = jax.lax.dynamic_slice(faces, (start, 0, 0), (batch_size, faces.shape[1], faces.shape[2]))
        
        new_mask, new_depth, new_tx, new_ty = render_triangles_batch(vertices, texture_coords, batch_faces, width, height)
        
        new_pixels = new_mask & (new_depth > depth_buffer)
        depth_buffer = jnp.where(new_pixels, new_depth, depth_buffer)
        update_mask = update_mask | new_pixels
        tx_buffer = jnp.where(new_pixels, new_tx, tx_buffer)
        ty_buffer = jnp.where(new_pixels, new_ty, ty_buffer)
        
        return (depth_buffer, update_mask, tx_buffer, ty_buffer), None
    
    initial_buffers = (
        jnp.full((height, width), -jnp.inf),
        jnp.zeros((height, width), dtype=bool),
        jnp.zeros((height, width)),
        jnp.zeros((height, width))
    )
    
    (final_depth_buffer, final_update_mask, final_tx_buffer, final_ty_buffer), _ = jax.lax.scan(
        render_and_update, initial_buffers, jnp.arange(num_batches)
    )
    
    tx = jnp.where(final_update_mask, final_tx_buffer, 0)
    ty = jnp.where(final_update_mask, final_ty_buffer, 0)

    color = texture[jnp.clip(ty * texture.shape[0], 0, texture.shape[0] - 1).astype(jnp.int32),
                    jnp.clip(tx * texture.shape[1], 0, texture.shape[1] - 1).astype(jnp.int32)]

    image = jnp.where(final_update_mask[:, :, jnp.newaxis], color, jnp.zeros_like(color))
    
    return image

def main():
    vertices, texture_coords, faces = parse_obj_file('african_head.obj')
    texture = jnp.array(Image.open('african_head_diffuse.tga'))
    
    width, height = 800, 600
    batch_size = 10  # Adjust this value based on your available memory
    
    image = render_model(vertices, texture_coords, faces, width, height, batch_size, texture)
    
    Image.fromarray(np.array(image)).save('output.png')

if __name__ == '__main__':
    main()