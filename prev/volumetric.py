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

def volumetric_rendering(vertices, texture_coords, faces, texture, width, height):
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Define camera parameters
    camera_pos = np.array([0, 0, 2])
    camera_dir = np.array([0, 0, -1])
    camera_up = np.array([0, 1, 0])
    camera_right = np.cross(camera_dir, camera_up)
    camera_up = np.cross(camera_right, camera_dir)

    # Define volume dimensions
    volume_min = np.min(vertices, axis=0)
    volume_max = np.max(vertices, axis=0)
    volume_size = volume_max - volume_min

    # Cast rays through the volume
    for y in range(height):
        for x in range(width):
            # Calculate ray direction
            u = (x - width / 2) / (width / 2)
            v = (height / 2 - y) / (height / 2)
            ray_dir = camera_dir + u * camera_right + v * camera_up
            ray_dir /= np.linalg.norm(ray_dir)

            # Define ray parameters
            ray_origin = camera_pos
            ray_step = 0.01
            ray_length = 0.0
            color = np.zeros(3)
            opacity = 0.0

            # Traverse the volume
            while ray_length < np.max(volume_size):
                ray_pos = ray_origin + ray_length * ray_dir

                # Check if the ray is inside the volume
                if np.all(ray_pos >= volume_min) and np.all(ray_pos <= volume_max):
                    # Find the intersected face
                    for face in faces:
                        v0, v1, v2 = [vertices[i] for i, _ in face]
                        vt0, vt1, vt2 = [texture_coords[i] for _, i in face if i is not None]

                        # Check if the ray intersects the face
                        edge1 = v1 - v0
                        edge2 = v2 - v0
                        h = np.cross(ray_dir, edge2)
                        a = np.dot(edge1, h)
                        if a > -1e-6 and a < 1e-6:
                            continue

                        f = 1.0 / a
                        s = ray_origin - v0
                        u = f * np.dot(s, h)
                        if u < 0.0 or u > 1.0:
                            continue

                        q = np.cross(s, edge1)
                        v = f * np.dot(ray_dir, q)
                        if v < 0.0 or u + v > 1.0:
                            continue

                        t = f * np.dot(edge2, q)
                        if t > 1e-6:
                            # Calculate barycentric coordinates
                            w = 1 - u - v
                            tx = w * vt0[0] + u * vt1[0] + v * vt2[0]
                            ty = 1 - (w * vt0[1] + u * vt1[1] + v * vt2[1])  # Invert v-coordinate

                            # Sample texture color
                            tex_color = texture[int(ty * texture.shape[0]), int(tx * texture.shape[1])]
                            color += (1 - opacity) * tex_color * ray_step
                            opacity += (1 - opacity) * ray_step

                            break

                ray_length += ray_step

            image[y, x] = color

    return image

def main():
    vertices, texture_coords, faces = parse_obj_file('obj/african_head.obj')
    texture = np.array(Image.open('prev/african_head_diffuse.tga'))
    
    image = volumetric_rendering(vertices, texture_coords, faces, texture, 800, 600)
    Image.fromarray(image).save('output.png')

if __name__ == '__main__':
    main()