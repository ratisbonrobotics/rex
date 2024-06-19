import torch
from PIL import Image
import numpy as np

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

    return torch.tensor(vertices), torch.tensor(texture_coords), torch.tensor(faces)

def render_triangles(vertices, texture_coords, faces, texture, width, height):
    image = torch.zeros((height, width, 3), dtype=torch.uint8)
    depth_buffer = torch.full((height, width), float('-inf'))

    for face in faces:
        v0, v1, v2 = vertices[face[:, 0]]
        vt0, vt1, vt2 = texture_coords[face[:, 1]]

        # Convert vertex coordinates to screen space
        v0 = ((v0[0] + 1) * width / 2, (1 - v0[1]) * height / 2, v0[2])
        v1 = ((v1[0] + 1) * width / 2, (1 - v1[1]) * height / 2, v1[2])
        v2 = ((v2[0] + 1) * width / 2, (1 - v2[1]) * height / 2, v2[2])

        # Calculate bounding box
        min_x, max_x = int(min(v0[0], v1[0], v2[0])), int(max(v0[0], v1[0], v2[0]))
        min_y, max_y = int(min(v0[1], v1[1], v2[1])), int(max(v0[1], v1[1], v2[1]))

        # Calculate edge function coefficients
        e01 = v1[1] - v0[1], v0[0] - v1[0], v1[0] * v0[1] - v0[0] * v1[1]
        e12 = v2[1] - v1[1], v1[0] - v2[0], v2[0] * v1[1] - v1[0] * v2[1]
        e20 = v0[1] - v2[1], v2[0] - v0[0], v0[0] * v2[1] - v2[0] * v0[1]

        # Generate x and y coordinates using meshgrid
        x, y = torch.meshgrid(torch.arange(max(min_x, 0), min(max_x + 1, width)),
                              torch.arange(max(min_y, 0), min(max_y + 1, height)))

        # Calculate barycentric coordinates using broadcasting
        w0 = e12[0] * (x + 0.5) + e12[1] * (y + 0.5) + e12[2]
        w1 = e20[0] * (x + 0.5) + e20[1] * (y + 0.5) + e20[2]
        w2 = e01[0] * (x + 0.5) + e01[1] * (y + 0.5) + e01[2]

        # Create a mask for pixels inside the triangle
        mask = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)

        # Calculate depth and texture coordinates using broadcasting
        w = w0 + w1 + w2
        w0, w1, w2 = w0 / w, w1 / w, w2 / w
        depth = w0 * v0[2] + w1 * v1[2] + w2 * v2[2]
        tx = w0 * vt0[0] + w1 * vt1[0] + w2 * vt2[0]
        ty = 1 - (w0 * vt0[1] + w1 * vt1[1] + w2 * vt2[1])

        # Update depth buffer and image for pixels inside the triangle
        update_mask = mask & (depth > depth_buffer[y, x])
        depth_buffer[y[update_mask], x[update_mask]] = depth[update_mask]
        color = texture[torch.clip(ty[update_mask] * texture.shape[0], 0, texture.shape[0] - 1).long(),
                        torch.clip(tx[update_mask] * texture.shape[1], 0, texture.shape[1] - 1).long()]
        image[y[update_mask], x[update_mask]] = color

    return image

def main():
    vertices, texture_coords, faces = parse_obj_file('african_head.obj')
    texture = torch.from_numpy(np.array(Image.open('african_head_diffuse.tga')))
    
    image = render_triangles(vertices, texture_coords, faces, texture, 800, 600)
    Image.fromarray(image.numpy()).save('output.png')

if __name__ == '__main__':
    main()