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

    return vertices, texture_coords, faces

def render_triangles(vertices, texture_coords, faces, texture, width, height):
    image = [[(0, 0, 0) for _ in range(width)] for _ in range(height)]
    depth_buffer = [[float('-inf') for _ in range(width)] for _ in range(height)]

    for face in faces:
        v0, v1, v2 = [vertices[i] for i, _ in face]
        vt0, vt1, vt2 = [texture_coords[i] for _, i in face if i is not None]

        # Convert vertex coordinates to screen space
        v0 = ((v0[0] + 1) * width / 2, (1 - v0[1]) * height / 2, v0[2])
        v1 = ((v1[0] + 1) * width / 2, (1 - v1[1]) * height / 2, v1[2])
        v2 = ((v2[0] + 1) * width / 2, (1 - v2[1]) * height / 2, v2[2])

        # Calculate bounding box
        min_x, max_x = int(min(v0[0], v1[0], v2[0])), int(max(v0[0], v1[0], v2[0]))
        min_y, max_y = int(min(v0[1], v1[1], v2[1])), int(max(v0[1], v1[1], v2[1]))

        # Calculate edge function coefficients
        e01 = (v1[1] - v0[1], v0[0] - v1[0], v1[0] * v0[1] - v0[0] * v1[1])
        e12 = (v2[1] - v1[1], v1[0] - v2[0], v2[0] * v1[1] - v1[0] * v2[1])
        e20 = (v0[1] - v2[1], v2[0] - v0[0], v0[0] * v2[1] - v2[0] * v0[1])

        for y in range(max(min_y, 0), min(max_y + 1, height)):
            for x in range(max(min_x, 0), min(max_x + 1, width)):
                # Calculate barycentric coordinates
                w0 = e12[0] * (x + 0.5) + e12[1] * (y + 0.5) + e12[2]
                w1 = e20[0] * (x + 0.5) + e20[1] * (y + 0.5) + e20[2]
                w2 = e01[0] * (x + 0.5) + e01[1] * (y + 0.5) + e01[2]

                if w0 >= 0 and w1 >= 0 and w2 >= 0:
                    # Pixel is inside the triangle
                    w = w0 + w1 + w2
                    w0, w1, w2 = w0 / w, w1 / w, w2 / w
                    depth = w0 * v0[2] + w1 * v1[2] + w2 * v2[2]

                    if depth > depth_buffer[y][x]:
                        depth_buffer[y][x] = depth

                        # Calculate texture coordinates
                        tx = w0 * vt0[0] + w1 * vt1[0] + w2 * vt2[0]
                        ty = 1 - (w0 * vt0[1] + w1 * vt1[1] + w2 * vt2[1])

                        # Sample texture
                        tx = max(0, min(int(tx * texture.width), texture.width - 1))
                        ty = max(0, min(int(ty * texture.height), texture.height - 1))
                        color = texture.getpixel((tx, ty))

                        image[y][x] = color

    return image

def main():
    vertices, texture_coords, faces = parse_obj_file('african_head.obj')
    texture = Image.open('african_head_diffuse.tga')
    
    image = render_triangles(vertices, texture_coords, faces, texture, 800, 600)
    
    # Convert the image list to a PIL Image
    output_image = Image.new('RGB', (800, 600))
    for y in range(600):
        for x in range(800):
            output_image.putpixel((x, y), image[y][x])
    
    output_image.save('output.png')

if __name__ == '__main__':
    main()