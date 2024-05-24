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

# Load the 3D model
vertices, triangles = load_obj("bunny.obj")

# Set up the camera and rendering parameters
camera_pos = jnp.array([0.0, 0.0, 0.0])
camera_lookat = jnp.array([0.0, 0.0, -1.0])
camera_up = jnp.array([0.0, 1.0, 0.0])
fov = 60.0
aspect_ratio = 1.0
image_width = 800
image_height = int(image_width / aspect_ratio)

# Compute the projection and view matrices
proj_matrix = compute_projection_matrix(fov, aspect_ratio, 0.1, 100.0)
view_matrix = compute_view_matrix(camera_pos, camera_lookat, camera_up)

# Create a blank image
image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

# Save the blank image as PNG
Image.fromarray(image).save("output.png")