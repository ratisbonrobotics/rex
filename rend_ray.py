from PIL import Image
import random
import math

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

    return vertices, texture_coords, faces

def ray_triangle_intersection(ray_origin, ray_direction, v0, v1, v2):
    epsilon = 1e-6
    edge1 = [v1[i] - v0[i] for i in range(3)]
    edge2 = [v2[i] - v0[i] for i in range(3)]
    h = cross_product(ray_direction, edge2)
    a = dot_product(edge1, h)

    if -epsilon < a < epsilon:
        return None

    f = 1.0 / a
    s = [ray_origin[i] - v0[i] for i in range(3)]
    u = f * dot_product(s, h)

    if u < 0.0 or u > 1.0:
        return None

    q = cross_product(s, edge1)
    v = f * dot_product(ray_direction, q)

    if v < 0.0 or u + v > 1.0:
        return None

    t = f * dot_product(edge2, q)

    if t > epsilon:
        return t, u, v

    return None

def cross_product(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ]

def dot_product(a, b):
    return sum(a[i] * b[i] for i in range(3))

def probabilistic_ray_casting(vertices, texture_coords, faces, texture, width, height, num_samples):
    image = [[(0, 0, 0) for _ in range(width)] for _ in range(height)]

    for _ in range(num_samples):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)

        ray_origin = [0, 0, -3]
        ray_direction = [
            (x - width / 2) / (width / 2),
            -(y - height / 2) / (height / 2),
            1
        ]
        ray_direction = normalize(ray_direction)

        closest_t = float('inf')
        closest_face = None
        closest_uv = None

        for face in faces:
            v0, v1, v2 = [vertices[i] for i, _ in face]
            result = ray_triangle_intersection(ray_origin, ray_direction, v0, v1, v2)

            if result and result[0] < closest_t:
                closest_t = result[0]
                closest_face = face
                closest_uv = result[1:]

        if closest_face:
            u, v = closest_uv
            w = 1 - u - v
            vt0, vt1, vt2 = [texture_coords[i] for _, i in closest_face if i is not None]

            tx = u * vt0[0] + v * vt1[0] + w * vt2[0]
            ty = 1 - (u * vt0[1] + v * vt1[1] + w * vt2[1])

            tx = max(0, min(int(tx * texture.width), texture.width - 1))
            ty = max(0, min(int(ty * texture.height), texture.height - 1))
            color = texture.getpixel((tx, ty))

            image[y][x] = color

    return image

def normalize(v):
    length = math.sqrt(sum(x * x for x in v))
    return [x / length for x in v]

import time

def main():
    start_time = time.time()

    vertices, texture_coords, faces = parse_obj_file('african_head.obj')
    texture = Image.open('african_head_diffuse.tga')
    
    width, height = 400, 300
    num_samples = width * height // 200  # Adjust this value to balance speed and quality
    
    print(f"Starting render with {num_samples} samples...")
    
    image = probabilistic_ray_casting(vertices, texture_coords, faces, texture, width, height, num_samples)
    
    # Convert the image list to a PIL Image
    output_image = Image.new('RGB', (width, height))
    for y in range(height):
        for x in range(width):
            output_image.putpixel((x, y), image[y][x])
    
    output_image.save('output.png')

    end_time = time.time()
    render_time = end_time - start_time
    print(f"Rendering completed in {render_time:.2f} seconds")

if __name__ == '__main__':
    main()