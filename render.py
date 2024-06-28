import jax
import jax.numpy as jnp
from jax import jit, vmap
from PIL import Image
import numpy as np

# Constants
WIDTH, HEIGHT = 800, 600

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

@jit
def rasterize_triangle(vertices, texture_coords, face):
    v0, v1, v2 = [vertices[i] for i, _ in face]
    vt0, vt1, vt2 = [texture_coords[i] for _, i in face if i is not None]

    # Perspective division
    v0 = v0[:3] / v0[3]
    v1 = v1[:3] / v1[3]
    v2 = v2[:3] / v2[3]

    # Convert to screen space
    v0 = ((v0[0] + 1) * WIDTH / 2, (1 - v0[1]) * HEIGHT / 2, v0[2])
    v1 = ((v1[0] + 1) * WIDTH / 2, (1 - v1[1]) * HEIGHT / 2, v1[2])
    v2 = ((v2[0] + 1) * WIDTH / 2, (1 - v2[1]) * HEIGHT / 2, v2[2])

    def edge_function(a, b, c):
        return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])

    x, y = jnp.meshgrid(jnp.arange(WIDTH), jnp.arange(HEIGHT))
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

@jit
def render_scene(vertices_list, texture_coords_list, faces_list, textures, mvp_matrices):
    depth_buffer = jnp.full((HEIGHT, WIDTH), jnp.inf)
    color_buffer = jnp.zeros((HEIGHT, WIDTH, 3), dtype=jnp.uint8)

    def render_model(depth_buffer, color_buffer, vertices, texture_coords, faces, texture, mvp_matrix):
        # Apply MVP matrix to vertices
        vertices_homogeneous = jnp.pad(vertices, ((0, 0), (0, 1)), constant_values=1)
        vertices_transformed = jnp.dot(vertices_homogeneous, mvp_matrix.T)

        def render_face(buffers, face):
            depth_buffer, color_buffer = buffers
            mask, depth, tx, ty = rasterize_triangle(vertices_transformed, texture_coords, face)
            
            update = mask & (depth < depth_buffer)
            depth_buffer = jnp.where(update, depth, depth_buffer)
            
            tx_clipped = jnp.clip(tx * texture.shape[1], 0, texture.shape[1] - 1).astype(jnp.int32)
            ty_clipped = jnp.clip(ty * texture.shape[0], 0, texture.shape[0] - 1).astype(jnp.int32)
            color = texture[ty_clipped, tx_clipped]
            
            color_buffer = jnp.where(update[:, :, jnp.newaxis], color, color_buffer)
            
            return (depth_buffer, color_buffer), None

        (depth_buffer, color_buffer), _ = jax.lax.scan(render_face, (depth_buffer, color_buffer), faces)
        
        return depth_buffer, color_buffer

    for vertices, texture_coords, faces, texture, mvp_matrix in zip(vertices_list, texture_coords_list, faces_list, textures, mvp_matrices):
        depth_buffer, color_buffer = render_model(depth_buffer, color_buffer, vertices, texture_coords, faces, texture, mvp_matrix)

    return color_buffer

def main():
    # Load models
    vertices1, texture_coords1, faces1 = parse_obj_file('drone.obj')
    texture1 = jnp.array(Image.open('drone.png').convert('RGB'))

    vertices2, texture_coords2, faces2 = parse_obj_file('african_head.obj')
    texture2 = jnp.array(Image.open('african_head_diffuse.tga').convert('RGB'))
    
    aspect_ratio = WIDTH / HEIGHT
    fov = jnp.radians(45)
    near = 0.1
    far = 100.0
    
    view_matrix = create_view_matrix(eye=jnp.array([0, 0, 3]), center=jnp.array([0, 0, 0]), up=jnp.array([0, 1, 0]))
    projection_matrix = create_projection_matrix(fov, aspect_ratio, near, far)
    
    # Create model matrices for each object and each output image
    model_matrices = jnp.array([
        [create_model_matrix(scale=[0.1, 0.1, 0.1], rotation=[0, 1, 0], translation=[-0.5, 0, -3]),  # Drone in output 1
         create_model_matrix(scale=[1.0, 1.0, 1.0], rotation=[0, 2, 0], translation=[0.5, 0, -4])],  # African head in output 1
        [create_model_matrix(scale=[0.1, 0.1, 0.1], rotation=[0, 3, 0], translation=[-0.5, 0, -3]),  # Drone in output 2
         create_model_matrix(scale=[1.0, 1.0, 1.0], rotation=[0, 4, 0], translation=[0.5, 0, -4])]   # African head in output 2
    ])
    
    mvp_matrices = jnp.einsum('ij,nkjl->nkil', projection_matrix @ view_matrix, model_matrices)
    
    # Vmap the render_scene function
    batched_render_scene = jit(vmap(render_scene, in_axes=(None, None, None, None, 0)))
    
    # Render all scenes at once
    images = batched_render_scene(
        [vertices1, vertices2],
        [texture_coords1, texture_coords2],
        [faces1, faces2],
        [texture1, texture2],
        mvp_matrices
    )
    
    # Save the images
    for i, image in enumerate(images):
        Image.fromarray(np.array(image)).save(f'output_{i+1}.png')

if __name__ == '__main__':
    main()