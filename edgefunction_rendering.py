import numpy as np
from PIL import Image
import random

def parse_obj_file(file_path):
    vertices = []
    faces = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                vertex = [float(coord) for coord in line.split()[1:]]
                vertices.append(vertex)
            elif line.startswith('f '):
                face = [int(vertex_index.split('/')[0]) - 1 for vertex_index in line.split()[1:]]
                faces.append(face)

    return np.array(vertices), np.array(faces)

def edge_function(v0, v1, p):
    return (p[0] - v0[0]) * (v1[1] - v0[1]) - (p[1] - v0[1]) * (v1[0] - v0[0])

def render_triangles(vertices, faces, width, height):
    image = Image.new('RGB', (width, height), color='black')
    depth_buffer = np.full((width, height), float('-inf'))  # Initialize depth buffer with negative infinity

    for face in faces:
        v0, v1, v2 = [vertices[i] for i in face]

        # Convert vertex coordinates to screen space
        v0 = ((v0[0] + 1) * width / 2, (1 - v0[1]) * height / 2, v0[2])
        v1 = ((v1[0] + 1) * width / 2, (1 - v1[1]) * height / 2, v1[2])
        v2 = ((v2[0] + 1) * width / 2, (1 - v2[1]) * height / 2, v2[2])

        # Generate random color for the triangle
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        # Calculate bounding box
        min_x = max(0, int(min(v0[0], v1[0], v2[0])))
        max_x = min(width - 1, int(max(v0[0], v1[0], v2[0])))
        min_y = max(0, int(min(v0[1], v1[1], v2[1])))
        max_y = min(height - 1, int(max(v0[1], v1[1], v2[1])))

        # Rasterize the triangle
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                p = (x, y)
                w0 = edge_function(v1, v2, p)
                w1 = edge_function(v2, v0, p)
                w2 = edge_function(v0, v1, p)

                if w0 >= 0 and w1 >= 0 and w2 >= 0:
                    # Calculate barycentric coordinates
                    area = edge_function(v0, v1, v2)
                    w0 /= area
                    w1 /= area
                    w2 /= area

                    # Interpolate depth value
                    depth = w0 * v0[2] + w1 * v1[2] + w2 * v2[2]

                    # Perform depth test
                    if depth > depth_buffer[x, y]:
                        depth_buffer[x, y] = depth
                        image.putpixel((x, y), color)

    return image

def main():
    input_file = 'obj/african_head.obj'
    output_file = 'output.png'
    width, height = 800, 600

    vertices, faces = parse_obj_file(input_file)
    image = render_triangles(vertices, faces, width, height)
    image.save(output_file)

if __name__ == '__main__':
    main()