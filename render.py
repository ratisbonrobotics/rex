import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

def load_obj(file_path):
    vertices = []
    triangles = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('v '):
                vertex = list(map(float, line.split()[1:]))
                vertices.append(vertex)
            elif line.startswith('f '):
                face = []
                for vertex_info in line.split()[1:]:
                    vertex_index = int(vertex_info.split('/')[0]) - 1
                    face.append(vertex_index)
                
                for i in range(1, len(face) - 1):
                    triangles.append([face[0], face[i], face[i + 1]])

    vertices = np.array(vertices, dtype=np.float32)
    triangles = np.array(triangles, dtype=np.int32)

    return vertices, triangles

def compute_projection_matrix(fov, aspect_ratio, near, far):
    f = 1.0 / jnp.tan(jnp.radians(fov) / 2.0)
    return jnp.array([
        [f / aspect_ratio, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (far + near) / (near - far), (2.0 * far * near) / (near - far)],
        [0.0, 0.0, -1.0, 0.0]
    ])

def compute_view_matrix(camera_pos, camera_lookat, camera_up):
    z_axis = camera_lookat - camera_pos
    z_axis = z_axis / jnp.linalg.norm(z_axis)
    x_axis = jnp.cross(camera_up, z_axis)
    x_axis = x_axis / jnp.linalg.norm(x_axis)
    y_axis = jnp.cross(z_axis, x_axis)
    return jnp.stack([x_axis, y_axis, z_axis, camera_pos], axis=1)

def rasterize(vertices, triangles, proj_matrix, view_matrix, image_width, image_height):
    # Filler implementation that results in a white image
    return [(x, y) for y in range(image_height) for x in range(image_width)]

# Load the 3D model
vertices, triangles = load_obj("bunny.obj")

# Set up the camera and rendering parameters
camera_pos = jnp.array([0.0, 0.0, 1.0])
camera_lookat = jnp.array([0.0, 0.0, 0.0])
camera_up = jnp.array([0.0, 1.0, 0.0])
fov = 60.0
aspect_ratio = 1.0
image_width = 800
image_height = int(image_width / aspect_ratio)

# Compute the projection and view matrices
proj_matrix = compute_projection_matrix(fov, aspect_ratio, 0.1, 100.0)
view_matrix = compute_view_matrix(camera_pos, camera_lookat, camera_up)

# Rasterize the scene
rasterized_triangles = rasterize(vertices, triangles, proj_matrix, view_matrix, image_width, image_height)

# Create the final image
image = np.zeros((image_height, image_width), dtype=np.float32)
for triangle in rasterized_triangles:
    x, y = triangle
    image[y, x] = 1.0

# Save the rendered image as PNG
Image.fromarray((image * 255).astype(np.uint8)).save("output.png")
