import numpy as np
from PIL import Image

def parse_obj_file(file_path):
    vertices, texture_coords, faces = [], [], []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                vertices.append([float(coord) for coord in line.split()[1:]])
            elif line.startswith('vt '):
                texture_coords.append([float(coord) for coord in line.split()[1:]])
            elif line.startswith('f '):
                faces.append([tuple(int(idx) - 1 if idx else None for idx in vertex.split('/')) for vertex in line.split()[1:]])
    return np.array(vertices), np.array(texture_coords), np.array(faces)

def edge_function(v0, v1, p):
    return (p[0] - v0[0]) * (v1[1] - v0[1]) - (p[1] - v0[1]) * (v1[0] - v0[0])

def render_triangles(vertices, texture_coords, faces, texture, width, height):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    depth_buffer = np.full((width, height), float('-inf'))
    for face in faces:
        v0, v1, v2 = [((vertices[i][0] + 1) * width / 2, (1 - vertices[i][1]) * height / 2, vertices[i][2]) for i, *_ in face]
        vt0, vt1, vt2 = [texture_coords[i] for _, i, *_ in face if i is not None]
        min_x, max_x = max(0, int(min(v0[0], v1[0], v2[0]))), min(width - 1, int(max(v0[0], v1[0], v2[0])))
        min_y, max_y = max(0, int(min(v0[1], v1[1], v2[1]))), min(height - 1, int(max(v0[1], v1[1], v2[1])))
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                w0, w1, w2 = edge_function(v1, v2, (x, y)), edge_function(v2, v0, (x, y)), edge_function(v0, v1, (x, y))
                if w0 >= 0 and w1 >= 0 and w2 >= 0:
                    area = edge_function(v0, v1, v2)
                    w0, w1, w2 = w0 / area, w1 / area, w2 / area
                    depth = w0 * v0[2] + w1 * v1[2] + w2 * v2[2]
                    if depth > depth_buffer[x, y]:
                        depth_buffer[x, y] = depth
                        tx, ty = w0 * vt0[0] + w1 * vt1[0] + w2 * vt2[0], 1 - (w0 * vt0[1] + w1 * vt1[1] + w2 * vt2[1])
                        image[y, x] = texture[int(ty * texture.shape[0]), int(tx * texture.shape[1])]
    return image

def main():
    vertices, texture_coords, faces = parse_obj_file('obj/african_head.obj')
    texture = np.array(Image.open('prev/african_head_diffuse.tga'))
    image = render_triangles(vertices, texture_coords, faces, texture, 800, 600)
    Image.fromarray(image).save('output.png')

if __name__ == '__main__':
    main()