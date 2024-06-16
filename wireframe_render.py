import numpy as np
from PIL import Image, ImageDraw

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

def render_wireframe(vertices, faces, width, height):
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)

    for face in faces:
        for i in range(len(face)):
            v1 = vertices[face[i]]
            v2 = vertices[face[(i + 1) % len(face)]]

            x1, y1 = int((v1[0] + 1) * width / 2), int((v1[1] + 1) * height / 2)
            x2, y2 = int((v2[0] + 1) * width / 2), int((v2[1] + 1) * height / 2)

            draw.line((x1, y1, x2, y2), fill='black', width=1)

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