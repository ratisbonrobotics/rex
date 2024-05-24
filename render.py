import jax.numpy as jnp
from jax import jit
from PIL import Image

def load_obj(file_path):
    vertices = []
    faces = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                vertex = [float(x) for x in line.split()[1:]]
                vertices.append(vertex)
            elif line.startswith('f '):
                face = [int(x.split('/')[0]) - 1 for x in line.split()[1:]]
                faces.append(face)
    return jnp.array(vertices), jnp.array(faces)

def project_vertices(vertices):
    return vertices[:, :2]

def rasterize_triangle(vertices, image, color):
    v0, v1, v2 = vertices
    xmin = jnp.ceil(jnp.minimum(v0[0], jnp.minimum(v1[0], v2[0]))).astype(int)
    xmax = jnp.floor(jnp.maximum(v0[0], jnp.maximum(v1[0], v2[0]))).astype(int)
    ymin = jnp.ceil(jnp.minimum(v0[1], jnp.minimum(v1[1], v2[1]))).astype(int)
    ymax = jnp.floor(jnp.maximum(v0[1], jnp.maximum(v1[1], v2[1]))).astype(int)

    for x in range(xmin, xmax + 1):
        for y in range(ymin, ymax + 1):
            if inside_triangle(x, y, vertices):
                image = image.at[y, x].set(color)

    return image

def inside_triangle(x, y, vertices):
    v0, v1, v2 = vertices
    area = jnp.abs(v0[0] * (v1[1] - v2[1]) + v1[0] * (v2[1] - v0[1]) + v2[0] * (v0[1] - v1[1])) / 2
    s = (v0[1] * v2[0] - v0[0] * v2[1] + (v2[1] - v0[1]) * x + (v0[0] - v2[0]) * y) / (2 * area)
    t = (v0[0] * v1[1] - v0[1] * v1[0] + (v0[1] - v1[1]) * x + (v1[0] - v0[0]) * y) / (2 * area)
    return s >= 0 and t >= 0 and (1 - s - t) >= 0

def render(vertices, faces, width, height, color):
    image = jnp.zeros((height, width, 3))
    projected_vertices = project_vertices(vertices)

    for face in faces:
        triangle_vertices = jnp.array([projected_vertices[i] for i in face])
        image = rasterize_triangle(triangle_vertices, image, color)

    return image

# Usage example
obj_file = 'bunny.obj'
vertices, faces = load_obj(obj_file)
width, height = 800, 600
color = jnp.array([255, 0, 0])  # Red color

rendered_image = render(vertices, faces, width, height, color)

# Save the rendered image
image = Image.fromarray(rendered_image.astype(jnp.uint8), mode='RGB')
image.save('rendered_image.png')