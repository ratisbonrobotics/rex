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

def create_projection_matrix(fov, aspect_ratio, near, far):
    f = 1 / jnp.tan(fov / 2)
    return jnp.array([
        [f / aspect_ratio, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, -1, 0]
    ])

def create_view_matrix(eye, center, up):
    f = (center - eye) / jnp.linalg.norm(center - eye)
    s = jnp.cross(f, up) / jnp.linalg.norm(jnp.cross(f, up))
    u = jnp.cross(s, f)
    
    return jnp.array([
        [s[0], s[1], s[2], -jnp.dot(s, eye)],
        [u[0], u[1], u[2], -jnp.dot(u, eye)],
        [-f[0], -f[1], -f[2], jnp.dot(f, eye)],
        [0, 0, 0, 1]
    ])

def create_model_matrix(scale, rotation, translation):
    scale = jnp.array(scale)
    rotation = jnp.array(rotation)
    translation = jnp.array(translation)
    
    scale_matrix = jnp.diag(jnp.concatenate([scale, jnp.array([1])]))
    
    rx, ry, rz = rotation
    rotation_x = jnp.array([
        [1, 0, 0, 0],
        [0, jnp.cos(rx), -jnp.sin(rx), 0],
        [0, jnp.sin(rx), jnp.cos(rx), 0],
        [0, 0, 0, 1]
    ])
    rotation_y = jnp.array([
        [jnp.cos(ry), 0, jnp.sin(ry), 0],
        [0, 1, 0, 0],
        [-jnp.sin(ry), 0, jnp.cos(ry), 0],
        [0, 0, 0, 1]
    ])
    rotation_z = jnp.array([
        [jnp.cos(rz), -jnp.sin(rz), 0, 0],
        [jnp.sin(rz), jnp.cos(rz), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    rotation_matrix = rotation_z @ rotation_y @ rotation_x
    
    translation_matrix = jnp.eye(4)
    translation_matrix = translation_matrix.at[:3, 3].set(translation)
    
    return translation_matrix @ rotation_matrix @ scale_matrix

@partial(jit, static_argnums=(3, 4))
def rasterize_triangle(vertices, texture_coords, face, width, height):
    v0, v1, v2 = [vertices[i] for i, _ in face]
    vt0, vt1, vt2 = [texture_coords[i] for _, i in face if i is not None]

    # Perspective division
    v0 = v0[:3] / v0[3]
    v1 = v1[:3] / v1[3]
    v2 = v2[:3] / v2[3]

    # Convert to screen space
    v0 = ((v0[0] + 1) * width / 2, (1 - v0[1]) * height / 2, v0[2])
    v1 = ((v1[0] + 1) * width / 2, (1 - v1[1]) * height / 2, v1[2])
    v2 = ((v2[0] + 1) * width / 2, (1 - v2[1]) * height / 2, v2[2])

    def edge_function(a, b, c):
        return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])

    x, y = jnp.meshgrid(jnp.arange(width), jnp.arange(height))
    points = jnp.stack([x, y], axis=-1)

    area = edge_function(v0, v1, v2)
    w0 = vmap(vmap(lambda p: edge_function(v1, v2, p)))(points) / area
    w1 = vmap(vmap(lambda p: edge_function(v2, v0, p)))(points) / area
    w2 = 1 - w0 - w1

    mask = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
    depth = w0 * v0[2] + w1 * v1[2] + w2 * v2[2]
    tx = w0 * vt0[0] + w1 * vt1[0] + w2 * vt2[0]
    ty = 1 - (w0 * vt0[1] + w1 * vt1[1] + w2 * vt2[1])

    return mask, depth, tx, ty

@partial(jit, static_argnums=(3, 4))
def render_model(vertices, texture_coords, faces, width, height, texture, mvp_matrix):
    depth_buffer = jnp.full((height, width), jnp.inf)
    color_buffer = jnp.zeros((height, width, 3), dtype=jnp.uint8)

    # Apply MVP matrix to vertices
    vertices_homogeneous = jnp.pad(vertices, ((0, 0), (0, 1)), constant_values=1)
    vertices_transformed = jnp.dot(vertices_homogeneous, mvp_matrix.T)

    def render_face(buffers, face):
        depth_buffer, color_buffer = buffers
        mask, depth, tx, ty = rasterize_triangle(vertices_transformed, texture_coords, face, width, height)
        
        update = mask & (depth < depth_buffer)
        depth_buffer = jnp.where(update, depth, depth_buffer)
        
        tx_clipped = jnp.clip(tx * texture.shape[1], 0, texture.shape[1] - 1).astype(jnp.int32)
        ty_clipped = jnp.clip(ty * texture.shape[0], 0, texture.shape[0] - 1).astype(jnp.int32)
        color = texture[ty_clipped, tx_clipped]
        
        color_buffer = jnp.where(update[:, :, jnp.newaxis], color, color_buffer)
        
        return (depth_buffer, color_buffer), None

    (_, final_color_buffer), _ = jax.lax.scan(render_face, (depth_buffer, color_buffer), faces)
    
    return final_color_buffer

def main():
    # Load African head model
    vertices1, texture_coords1, faces1 = parse_obj_file('african_head.obj')
    texture1 = jnp.array(Image.open('african_head_diffuse.tga'))

    # Load African head model
    vertices2, texture_coords2, faces2 = parse_obj_file('african_head.obj')
    texture2 = jnp.array(Image.open('african_head_diffuse.tga'))
    
    width, height = 800, 600
    aspect_ratio = width / height
    fov = jnp.radians(45)
    near = 0.1
    far = 100.0
    
    view_matrix = create_view_matrix(eye=jnp.array([0, 0, 3]), center=jnp.array([0, 0, 0]), up=jnp.array([0, 1, 0]))
    projection_matrix = create_projection_matrix(fov, aspect_ratio, near, far)
    
    # Create two different model matrices
    model_matrix1 = create_model_matrix(scale=[1, 1, 1], rotation=[0, 1, 0], translation=[0, 0, -3])
    model_matrix2 = create_model_matrix(scale=[1, 1, 1], rotation=[0, 2, 0], translation=[0, 0, -3])
    
    mvp_matrix1 = projection_matrix @ view_matrix @ model_matrix1
    mvp_matrix2 = projection_matrix @ view_matrix @ model_matrix2
    
    # Vmap the render_model function
    batched_render_model = vmap(render_model, in_axes=(0, 0, 0, None, None, 0, 0))
    
    # Render both models in parallel
    images = batched_render_model(
        jnp.stack([vertices1, vertices2]),
        jnp.stack([texture_coords1, texture_coords2]),
        jnp.stack([faces1, faces2]),
        width,
        height,
        jnp.stack([texture1, texture2]),
        jnp.stack([mvp_matrix1, mvp_matrix2])
    )
    
    # Save both images
    Image.fromarray(np.array(images[0])).save('output_1.png')
    Image.fromarray(np.array(images[1])).save('output_2.png')

if __name__ == '__main__':
    main()