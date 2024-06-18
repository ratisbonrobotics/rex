import numpy as np
from PIL import Image

def parse_obj_file(file_path):
    vertices = []
    texture_coords = []
    faces = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                vertex = [float(coord) for coord in line.split()[1:]]
                vertices.append(vertex)
            elif line.startswith('vt '):
                texture_coord = [float(coord) for coord in line.split()[1:]]
                texture_coords.append(texture_coord)
            elif line.startswith('f '):
                face = []
                for vertex_index in line.split()[1:]:
                    indices = vertex_index.split('/')
                    vertex_index = int(indices[0]) - 1
                    texture_index = int(indices[1]) - 1 if len(indices) > 1 else None
                    face.append((vertex_index, texture_index))
                faces.append(face)

    return np.array(vertices), np.array(texture_coords), np.array(faces)

def edge_function(v0, v1, ps):
    return (ps[:, 0] - v0[0]) * (v1[1] - v0[1]) - (ps[:, 1] - v0[1]) * (v1[0] - v0[0])

def render_face(face, vertices, texture_coords, texture, width, height, depth_buffer, image):
    v0, v1, v2 = [vertices[i] for i, _ in face]
    vt0, vt1, vt2 = [texture_coords[i] for _, i in face]

    v0 = ((v0[0] + 1) * width / 2, (1 - v0[1]) * height / 2, v0[2])
    v1 = ((v1[0] + 1) * width / 2, (1 - v1[1]) * height / 2, v1[2])
    v2 = ((v2[0] + 1) * width / 2, (1 - v2[1]) * height / 2, v2[2])

    min_x = max(0, int(min(v0[0], v1[0], v2[0])))
    max_x = min(width - 1, int(max(v0[0], v1[0], v2[0])))
    min_y = max(0, int(min(v0[1], v1[1], v2[1])))
    max_y = min(height - 1, int(max(v0[1], v1[1], v2[1])))

    xs, ys = np.meshgrid(np.arange(min_x, max_x + 1), np.arange(min_y, max_y + 1))
    xs, ys = xs.flatten(), ys.flatten()
    ps = np.vstack((xs, ys)).T

    w0 = edge_function(v1, v2, ps)
    w1 = edge_function(v2, v0, ps)
    w2 = edge_function(v0, v1, ps)

    area = edge_function(v0, v1, np.array([v2]))
    w0 /= area
    w1 /= area
    w2 /= area

    mask = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
    valid_ps = ps[mask]
    valid_w0, valid_w1, valid_w2 = w0[mask], w1[mask], w2[mask]

    depth = valid_w0 * v0[2] + valid_w1 * v1[2] + valid_w2 * v2[2]
    xs, ys = valid_ps[:, 0], valid_ps[:, 1]

    update_mask = depth > depth_buffer[ys, xs]
    ys_update, xs_update = ys[update_mask], xs[update_mask]
    depth_buffer[ys_update, xs_update] = depth[update_mask]

    tx = (valid_w0[update_mask] * vt0[0] + valid_w1[update_mask] * vt1[0] + valid_w2[update_mask] * vt2[0]) * texture.shape[1]
    ty = (1 - (valid_w0[update_mask] * vt0[1] + valid_w1[update_mask] * vt1[1] + valid_w2[update_mask] * vt2[1])) * texture.shape[0]

    tx = np.clip(tx.astype(int), 0, texture.shape[1] - 1)
    ty = np.clip(ty.astype(int), 0, texture.shape[0] - 1)

    image[ys_update, xs_update] = texture[ty, tx]

def main():
    vertices, texture_coords, faces = parse_obj_file('african_head.obj')
    texture = np.array(Image.open('african_head_diffuse.tga'))

    width,height = 800, 600
    image = np.zeros((height, width, 3), dtype=np.uint8)
    depth_buffer = np.full((height, width), float('-inf'))

    for face in faces:
        render_face(face, vertices, texture_coords, texture, width, height, depth_buffer, image)

    Image.fromarray(image).save('output.png')

if __name__ == '__main__':
    main()