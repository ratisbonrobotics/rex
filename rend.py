import numpy as np
from PIL import Image

def identMat4f():
    return np.eye(4, dtype=np.float32)

def multMat4f(a, b):
    return np.dot(a, b)

def translMat4f(tx, ty, tz):
    return np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ], dtype=np.float32)

def xRotMat4f(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

def yRotMat4f(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

def zRotMat4f(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [c, -s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

def scaleMat4f(sx, sy, sz):
    return np.array([
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

def modelMat4f(tx, ty, tz, rx, ry, rz, sx, sy, sz):
    modelmatrix = identMat4f()
    modelmatrix = multMat4f(translMat4f(tx, ty, tz), modelmatrix)
    modelmatrix = multMat4f(multMat4f(multMat4f(xRotMat4f(rx), yRotMat4f(ry)), zRotMat4f(rz)), modelmatrix)
    modelmatrix = multMat4f(scaleMat4f(sx, sy, sz), modelmatrix)
    return modelmatrix

def create_projection_matrix(fov, aspect_ratio, near, far):
    f = 1 / np.tan(fov / 2)
    return np.array([
        [f / aspect_ratio, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, -1, 0]
    ], dtype=np.float32)

def create_view_matrix(eye, center, up):
    f = (center - eye) / np.linalg.norm(center - eye)
    s = np.cross(f, up) / np.linalg.norm(np.cross(f, up))
    u = np.cross(s, f)
    
    return np.array([
        [s[0], s[1], s[2], -np.dot(s, eye)],
        [u[0], u[1], u[2], -np.dot(u, eye)],
        [-f[0], -f[1], -f[2], np.dot(f, eye)],
        [0, 0, 0, 1]
    ], dtype=np.float32)

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

def edge_function(v0, v1, p):
    return (p[0] - v0[0]) * (v1[1] - v0[1]) - (p[1] - v0[1]) * (v1[0] - v0[0])

def render_triangles(vertices, texture_coords, faces, texture, width, height, model_matrix, view_matrix, projection_matrix):
    image = Image.new('RGB', (width, height), color='black')
    depth_buffer = np.full((width, height), float('inf'))  # Initialize with positive infinity

    view_model_matrix = np.dot(view_matrix, model_matrix)

    for face in faces:
        v0, v1, v2 = [vertices[i] for i, _ in face]
        vt0, vt1, vt2 = [texture_coords[i] for _, i in face if i is not None]

        # Apply view-model matrix to vertices
        v0 = np.dot(view_model_matrix, np.append(v0, 1))
        v1 = np.dot(view_model_matrix, np.append(v1, 1))
        v2 = np.dot(view_model_matrix, np.append(v2, 1))

        # Apply projection matrix
        v0 = np.dot(projection_matrix, v0)
        v1 = np.dot(projection_matrix, v1)
        v2 = np.dot(projection_matrix, v2)

        # Perspective division
        v0 = v0[:3] / v0[3]
        v1 = v1[:3] / v1[3]
        v2 = v2[:3] / v2[3]

        # Convert vertex coordinates to screen space
        v0 = ((v0[0] + 1) * width / 2, (1 - v0[1]) * height / 2, v0[2])
        v1 = ((v1[0] + 1) * width / 2, (1 - v1[1]) * height / 2, v1[2])
        v2 = ((v2[0] + 1) * width / 2, (1 - v2[1]) * height / 2, v2[2])

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

                    # Perform depth test (note the change in comparison)
                    if depth < depth_buffer[x, y]:
                        depth_buffer[x, y] = depth

                        # Interpolate texture coordinates
                        tx = w0 * vt0[0] + w1 * vt1[0] + w2 * vt2[0]
                        ty = 1 - (w0 * vt0[1] + w1 * vt1[1] + w2 * vt2[1])  # Invert v-coordinate

                        # Sample texture color
                        color = texture.getpixel((int(tx * texture.width) % texture.width, 
                                                  int(ty * texture.height) % texture.height))
                        image.putpixel((x, y), color)

    return image

def main():
    input_file = 'african_head.obj'
    texture_file = 'african_head_diffuse.tga'
    output_file = 'output.png'
    width, height = 800, 600

    vertices, texture_coords, faces = parse_obj_file(input_file)
    texture = Image.open(texture_file)
    
    # Create matrices
    model_matrix = modelMat4f(0, 0, 0, 0, np.pi/4, 0, 1, 1, 1)
    view_matrix = create_view_matrix(np.array([0, 0, 3]), np.array([0, 0, 0]), np.array([0, 1, 0]))
    projection_matrix = create_projection_matrix(np.radians(45), width/height, 0.1, 100.0)
    
    image = render_triangles(vertices, texture_coords, faces, texture, width, height, 
                             model_matrix, view_matrix, projection_matrix)
    image.save(output_file)

if __name__ == '__main__':
    main()