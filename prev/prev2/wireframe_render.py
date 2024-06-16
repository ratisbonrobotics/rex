import numpy as np
from PIL import Image

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

def draw_line(image, x1, y1, x2, y2, color):
    width, height = image.size
    x1, y1 = max(0, min(x1, width - 1)), max(0, min(y1, height - 1))
    x2, y2 = max(0, min(x2, width - 1)), max(0, min(y2, height - 1))

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        if 0 <= x1 < width and 0 <= y1 < height:
            image.putpixel((x1, y1), color)
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

def render_wireframe(vertices, faces, width, height):
    image = Image.new('RGB', (width, height), color='white')

    for face in faces:
        for i in range(len(face)):
            v1 = vertices[face[i]]
            v2 = vertices[face[(i + 1) % len(face)]]

            x1, y1 = int((v1[0] + 1) * width / 2), int((v1[1] + 1) * height / 2)
            x2, y2 = int((v2[0] + 1) * width / 2), int((v2[1] + 1) * height / 2)

            draw_line(image, x1, y1, x2, y2, (0, 0, 0))

    return image

def main():
    input_file = 'obj/african_head.obj'
    output_file = 'output.png'
    width, height = 800, 600

    vertices, faces = parse_obj_file(input_file)
    image = render_wireframe(vertices, faces, width, height)
    image.save(output_file)

if __name__ == '__main__':
    main()