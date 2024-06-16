import jax.numpy as jnp
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

    return jnp.array(vertices), jnp.array(texture_coords), jnp.array(faces)

def edge_function(v0, v1, p):
    return (p[0] - v0[0]) * (v1[1] - v0[1]) - (p[1] - v0[1]) * (v1[0] - v0[0])

def render_triangles(vertices, texture_coords, faces, texture, width, height):
    image = jnp.zeros((height, width, 3), dtype=jnp.uint8)
    depth_buffer = jnp.full((width, height), float('-inf'))

    for face in faces:
        v0, v1, v2 = [vertices[i] for i, _ in face]
        vt0, vt1, vt2 = [texture_coords[i] for _, i in face if i is not None]

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

                    # Perform depth test
                    if depth > depth_buffer[x, y]:
                        depth_buffer = depth_buffer.at[x, y].set(depth)

                        # Interpolate texture coordinates
                        tx = w0 * vt0[0] + w1 * vt1[0] + w2 * vt2[0]
                        ty = 1 - (w0 * vt0[1] + w1 * vt1[1] + w2 * vt2[1])  # Invert v-coordinate

                        # Sample texture color
                        color = texture[int(ty * texture.shape[0]), int(tx * texture.shape[1])]
                        image = image.at[y, x].set(color)

    return image

def main():
    vertices, texture_coords, faces = parse_obj_file('obj/african_head.obj')
    texture = jnp.array(Image.open('prev/african_head_diffuse.tga'))
    
    image = render_triangles(vertices, texture_coords, faces, texture, 800, 600)
    Image.fromarray(image.astype(jnp.uint8)).save('output.png')

if __name__ == '__main__':
    main()