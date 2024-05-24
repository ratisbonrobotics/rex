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

@jax.jit
def compute_projection_matrix(fov, aspect_ratio, near, far):
    f = 1.0 / jnp.tan(jnp.radians(fov) / 2.0)
    return jnp.array([
        [f / aspect_ratio, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (far + near) / (near - far), (2.0 * far * near) / (near - far)],
        [0.0, 0.0, -1.0, 0.0]
    ])

@jax.jit
def compute_view_matrix(camera_pos, camera_lookat, camera_up):
    z_axis = camera_lookat - camera_pos
    z_axis = z_axis / jnp.linalg.norm(z_axis)
    x_axis = jnp.cross(camera_up, z_axis)
    x_axis = x_axis / jnp.linalg.norm(x_axis)
    y_axis = jnp.cross(z_axis, x_axis)
    return jnp.stack([x_axis, y_axis, z_axis, camera_pos], axis=1)

@jax.jit
def rasterize(vertices, triangles, proj_matrix, view_matrix, image_width, image_height):
    # Transform vertices from world space to screen space
    vertices_homogeneous = jnp.hstack((vertices, jnp.ones((vertices.shape[0], 1))))
    vertices_view = jnp.dot(view_matrix, vertices_homogeneous.T).T
    vertices_view_homogeneous = jnp.hstack((vertices_view, jnp.ones((vertices_view.shape[0], 1))))
    vertices_proj = jnp.dot(proj_matrix, vertices_view_homogeneous.T).T
    vertices_proj = vertices_proj[:, :3] / vertices_proj[:, 3:]
    vertices_screen = jnp.clip(vertices_proj[:, :2], -1.0, 1.0)
    vertices_screen = (vertices_screen + 1.0) * 0.5
    vertices_screen = vertices_screen.at[:, 0].multiply(image_width)
    vertices_screen = vertices_screen.at[:, 1].multiply(image_height)
    vertices_screen = vertices_screen.astype(jnp.int32)

    # Rasterize triangles
    rasterized_triangles = []
    for triangle in triangles:
        v0, v1, v2 = vertices_screen[triangle]
        xmin = jnp.min(jnp.array([v0[0], v1[0], v2[0]]))
        xmax = jnp.max(jnp.array([v0[0], v1[0], v2[0]]))
        ymin = jnp.min(jnp.array([v0[1], v1[1], v2[1]]))
        ymax = jnp.max(jnp.array([v0[1], v1[1], v2[1]]))

        for x in range(xmin, xmax + 1):
            for y in range(ymin, ymax + 1):
                # Compute barycentric coordinates
                v0v1 = v1 - v0
                v0v2 = v2 - v0
                v0p = jnp.array([x, y]) - v0
                denom = v0v1[0] * v0v2[1] - v0v1[1] * v0v2[0]
                if denom == 0:
                    continue
                u = (v0p[0] * v0v2[1] - v0p[1] * v0v2[0]) / denom
                v = (v0p[1] * v0v1[0] - v0p[0] * v0v1[1]) / denom
                if u >= 0 and v >= 0 and u + v <= 1:
                    rasterized_triangles.append((x, y))

    return rasterized_triangles

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

# Rasterize the scene
rasterized_triangles = rasterize(vertices, triangles, proj_matrix, view_matrix, image_width, image_height)

# Create the final image
image = np.zeros((image_height, image_width), dtype=np.float32)
for triangle in rasterized_triangles:
    x, y = triangle
    image[y, x] = 1.0

# Save the rendered image as PNG
Image.fromarray((image * 255).astype(np.uint8)).save("output.png")