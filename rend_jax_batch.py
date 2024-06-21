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

    def update_buffers(carry, inputs):
        depth_buffer, update_mask, tx_buffer, ty_buffer = carry
        mask_i, depth_i, tx_i, ty_i = inputs

        update_mask_i = mask_i & (depth_i > depth_buffer)
        new_depth_buffer = jnp.where(update_mask_i, depth_i, depth_buffer)
        new_update_mask = update_mask | update_mask_i
        new_tx_buffer = jnp.where(update_mask_i, tx_i, tx_buffer)
        new_ty_buffer = jnp.where(update_mask_i, ty_i, ty_buffer)

        return (new_depth_buffer, new_update_mask, new_tx_buffer, new_ty_buffer), None

    depth_buffer = jnp.full((height, width), float('-inf'))
    update_mask = jnp.zeros((height, width), dtype=bool)
    tx_buffer = jnp.zeros((height, width))
    ty_buffer = jnp.zeros((height, width))

    (depth_buffer, update_mask, tx_buffer, ty_buffer), _ = jax.lax.scan(update_buffers, (depth_buffer, update_mask, tx_buffer, ty_buffer), (mask, depth, tx, ty))

    return update_mask, tx_buffer, ty_buffer

def apply_texture(update_mask, tx_buffer, ty_buffer, texture):
    tx = jnp.where(update_mask, tx_buffer, 0)
    ty = jnp.where(update_mask, ty_buffer, 0)

    color = texture[jnp.clip(ty * texture.shape[0], 0, texture.shape[0] - 1).astype(jnp.int32),
                    jnp.clip(tx * texture.shape[1], 0, texture.shape[1] - 1).astype(jnp.int32)]

    return jnp.where(update_mask[:, :, jnp.newaxis], color, jnp.zeros_like(color))

def main():
    vertices, texture_coords, faces = parse_obj_file('african_head.obj')
    texture = jnp.array(Image.open('african_head_diffuse.tga'))
    
    width, height = 800, 600
    batch_size = 10  # Adjust this value based on your available memory
    
    final_update_mask = jnp.zeros((height, width), dtype=bool)
    final_tx_buffer = jnp.zeros((height, width))
    final_ty_buffer = jnp.zeros((height, width))
    
    for i in range(0, len(faces), batch_size):
        batch_faces = faces[i:i+batch_size]
        update_mask, tx_buffer, ty_buffer = render_triangles_batch(vertices, texture_coords, batch_faces, width, height)
        
        final_update_mask = final_update_mask | update_mask
        final_tx_buffer = jnp.where(update_mask, tx_buffer, final_tx_buffer)
        final_ty_buffer = jnp.where(update_mask, ty_buffer, final_ty_buffer)
    
    image = apply_texture(final_update_mask, final_tx_buffer, final_ty_buffer, texture)
    
    Image.fromarray(np.array(image)).save('output.png')

if __name__ == '__main__':
    main()