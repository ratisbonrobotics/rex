import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
from PIL import Image
import numpy as np

def parse_obj_file(file_path):
    vertices, texture_coords, faces = [], [], []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '): vertices.append([float(coord) for coord in line.split()[1:]])
            elif line.startswith('vt '): texture_coords.append([float(coord) for coord in line.split()[1:]])
            elif line.startswith('f '):
                face = []
                for vertex_index in line.split()[1:]:
                    indices = vertex_index.split('/')
                    face.append((int(indices[0]) - 1, int(indices[1]) - 1 if len(indices) > 1 else None))
                faces.append(face)
    return jnp.array(vertices), jnp.array(texture_coords), jnp.array(faces)

@partial(jit, static_argnums=(3, 4))
def rasterize_triangle(vertices, texture_coords, face, width, height):
    v0, v1, v2 = [vertices[i] for i, _ in face]
    vt0, vt1, vt2 = [texture_coords[i] for _, i in face if i is not None]

    # Convert to screen space
    v0 = ((v0[0] + 1) * width / 2, (1 - v0[1]) * height / 2, v0[2])
    v1 = ((v1[0] + 1) * width / 2, (1 - v1[1]) * height / 2, v1[2])
    v2 = ((v2[0] + 1) * width / 2, (1 - v2[1]) * height / 2, v2[2])

    def edge_function(a, b, c):
        return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])

    def inside_triangle(p):
        w0 = edge_function(v1, v2, p)
        w1 = edge_function(v2, v0, p)
        w2 = edge_function(v0, v1, p)
        return (w0 >= 0) & (w1 >= 0) & (w2 >= 0)

    x, y = jnp.meshgrid(jnp.arange(width), jnp.arange(height))
    points = jnp.stack([x, y], axis=-1)

    mask = vmap(vmap(inside_triangle))(points)

    area = edge_function(v0, v1, v2)
    w0 = vmap(vmap(lambda p: edge_function(v1, v2, p)))(points) / area
    w1 = vmap(vmap(lambda p: edge_function(v2, v0, p)))(points) / area
    w2 = 1 - w0 - w1

    depth = w0 * v0[2] + w1 * v1[2] + w2 * v2[2]
    tx = w0 * vt0[0] + w1 * vt1[0] + w2 * vt2[0]
    ty = 1 - (w0 * vt0[1] + w1 * vt1[1] + w2 * vt2[1])

    return mask, depth, tx, ty

@partial(jit, static_argnums=(3, 4))
def render_model(vertices, texture_coords, faces, width, height, texture):
    depth_buffer = jnp.full((height, width), -jnp.inf)
    color_buffer = jnp.zeros((height, width, 3), dtype=jnp.uint8)

    def render_face(buffers, face):
        depth_buffer, color_buffer = buffers
        mask, depth, tx, ty = rasterize_triangle(vertices, texture_coords, face, width, height)
        
        update = mask & (depth > depth_buffer)
        depth_buffer = jnp.where(update, depth, depth_buffer)
        
        tx_clipped = jnp.clip(tx * texture.shape[1], 0, texture.shape[1] - 1).astype(jnp.int32)
        ty_clipped = jnp.clip(ty * texture.shape[0], 0, texture.shape[0] - 1).astype(jnp.int32)
        color = texture[ty_clipped, tx_clipped]
        
        color_buffer = jnp.where(update[:, :, jnp.newaxis], color, color_buffer)
        
        return (depth_buffer, color_buffer), None

    (_, final_color_buffer), _ = jax.lax.scan(render_face, (depth_buffer, color_buffer), faces)
    
    return final_color_buffer

def main():
    vertices, texture_coords, faces = parse_obj_file('african_head.obj')
    texture = jnp.array(Image.open('african_head_diffuse.tga'))
    
    width, height = 800, 600
    
    image = render_model(vertices, texture_coords, faces, width, height, texture)
    
    Image.fromarray(np.array(image)).save('output.png')

if __name__ == '__main__':
    main()