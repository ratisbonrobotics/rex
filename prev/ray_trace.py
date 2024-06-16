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

def ray_triangle_intersection(ray_origin, ray_direction, v0, v1, v2):
    e1 = v1 - v0
    e2 = v2 - v0
    h = np.cross(ray_direction, e2)
    a = np.dot(e1, h)

    if -1e-6 < a < 1e-6:
        return None

    f = 1.0 / a
    s = ray_origin - v0
    u = f * np.dot(s, h)

    if u < 0.0 or u > 1.0:
        return None

    q = np.cross(s, e1)
    v = f * np.dot(ray_direction, q)

    if v < 0.0 or u + v > 1.0:
        return None

    t = f * np.dot(e2, q)

    if t > 1e-6:
        return t, u, v

    return None

def render_triangles(vertices, texture_coords, faces, texture, width, height):
    image = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            # Calculate ray direction
            ray_direction = np.array([
                (2 * (x + 0.5) / width - 1) * (width / height),
                (1 - 2 * (y + 0.5) / height),
                -1.0
            ])
            ray_direction /= np.linalg.norm(ray_direction)

            # Find the nearest intersection
            nearest_t = float('inf')
            nearest_face = None
            nearest_uv = None

            for face in faces:
                v0, v1, v2 = [vertices[i] for i, _ in face]
                intersection = ray_triangle_intersection(np.array([0, 0, 0]), ray_direction, v0, v1, v2)

                if intersection is not None:
                    t, u, v = intersection
                    if t < nearest_t:
                        nearest_t = t
                        nearest_face = face
                        nearest_uv = (u, v)

            if nearest_face is not None:
                vt0, vt1, vt2 = [texture_coords[i] for _, i in nearest_face if i is not None]
                tx = nearest_uv[0] * vt1[0] + nearest_uv[1] * vt2[0] + (1 - nearest_uv[0] - nearest_uv[1]) * vt0[0]
                ty = 1 - (nearest_uv[0] * vt1[1] + nearest_uv[1] * vt2[1] + (1 - nearest_uv[0] - nearest_uv[1]) * vt0[1])

                color = texture[int(ty * texture.shape[0]), int(tx * texture.shape[1])]
                image[y, x] = color

    return image

def main():
    vertices, texture_coords, faces = parse_obj_file('obj/african_head.obj')
    texture = np.array(Image.open('prev/african_head_diffuse.tga'))
    
    image = render_triangles(vertices, texture_coords, faces, texture, 800, 600)
    Image.fromarray(image).save('output.png')

if __name__ == '__main__':
    main()