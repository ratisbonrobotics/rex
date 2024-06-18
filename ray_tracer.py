import numpy as np
from PIL import Image

def parse_obj_file(file_path):
    vertices = []
    normals = []
    faces = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                vertex = [float(coord) for coord in line.split()[1:]]
                vertices.append(vertex)
            elif line.startswith('vn '):
                normal = [float(coord) for coord in line.split()[1:]]
                normals.append(normal)
            elif line.startswith('f '):
                face = []
                for vertex_index in line.split()[1:]:
                    indices = vertex_index.split('/')
                    vertex_index = int(indices[0]) - 1
                    normal_index = int(indices[2]) - 1 if len(indices) > 2 and indices[2] else None
                    face.append((vertex_index, normal_index))
                faces.append(face)

    return np.array(vertices), np.array(normals), np.array(faces)

def ray_intersect_triangle(orig, dir, vert0, vert1, vert2):
    # Moller-Trumbore intersection algorithm
    EPSILON = 0.0000001
    edge1 = vert1 - vert0
    edge2 = vert2 - vert0
    h = np.cross(dir, edge2)
    a = np.dot(edge1, h)
    if -EPSILON < a < EPSILON:
        return False, None
    f = 1.0 / a
    s = orig - vert0
    u = f * np.dot(s, h)
    if u < 0.0 or u > 1.0:
        return False, None
    q = np.cross(s, edge1)
    v = f * np.dot(dir, q)
    if v < 0.0 or u + v > 1.0:
        return False, None
    t = f * np.dot(edge2, q)
    if t > EPSILON:
        return True, t
    else:
        return False, None

def ray_trace(vertices, normals, faces, orig, light_dir, width, height):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for x in range(width):
        for y in range(height):
            # Convert pixel to view space
            px = (x / width) * 2 - 1
            py = (y / height) * 2 - 1
            dir = np.array([px, py, -1])  # Simple perspective projection
            dir = dir / np.linalg.norm(dir)  # Normalize direction

            closest_t = np.inf
            final_color = np.array([0, 0, 0])
            for face in faces:
                verts = [vertices[i] for i, _ in face]
                if len(face[0]) > 1 and face[0][1] is not None:
                    norms = [normals[n] for _, n in face]
                else:
                    # Calculate normal from vertices if no normals in OBJ
                    norm = np.cross(verts[1] - verts[0], verts[2] - verts[0])
                    norms = [norm, norm, norm]
                hit, t = ray_intersect_triangle(orig, dir, np.array(verts[0]), np.array(verts[1]), np.array(verts[2]))
                if hit and t < closest_t:
                    closest_t = t
                    # Simple diffuse lighting calculation
                    norm = np.mean(norms, axis=0)
                    norm = norm / np.linalg.norm(norm)
                    light_intensity = max(0, np.dot(norm, -light_dir))
                    final_color = np.clip(light_intensity * 255, 0, 255)

            image[y, x] = final_color
    return image

def main():
    vertices, normals, faces = parse_obj_file('african_head.obj')
    orig = np.array([0, 0, 3])  # Camera position
    light_dir = np.array([0, 0, -1])  # Light direction
    image = ray_trace(vertices, normals, faces, orig, light_dir, 800, 600)
    Image.fromarray(image).save('output_raytraced.png')

if __name__ == '__main__':
    main()
