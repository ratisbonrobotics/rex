import numpy as np
from PIL import Image
from numba import jit

def parse_obj_file(file_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

@jit(nopython=True)
def convert_to_screen_space(v: np.ndarray, width: int, height: int) -> tuple:
    return ((v[0] + 1) * width / 2, (1 - v[1]) * height / 2, v[2])

@jit(nopython=True)
def calculate_bounding_box(v0: tuple, v1: tuple, v2: tuple) -> tuple[int, int, int, int]:
    min_x, max_x = int(min(v0[0], v1[0], v2[0])), int(max(v0[0], v1[0], v2[0]))
    min_y, max_y = int(min(v0[1], v1[1], v2[1])), int(max(v0[1], v1[1], v2[1]))
    return min_x, max_x, min_y, max_y

@jit(nopython=True)
def calculate_edge_function_coefficients(v0: tuple, v1: tuple, v2: tuple) -> tuple[tuple, tuple, tuple]:
    e01 = v1[1] - v0[1], v0[0] - v1[0], v1[0] * v0[1] - v0[0] * v1[1]
    e12 = v2[1] - v1[1], v1[0] - v2[0], v2[0] * v1[1] - v1[0] * v2[1]
    e20 = v0[1] - v2[1], v2[0] - v0[0], v0[0] * v2[1] - v2[0] * v0[1]
    return e01, e12, e20

def generate_coordinates(min_x: int, max_x: int, min_y: int, max_y: int, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.meshgrid(np.arange(max(min_x, 0), min(max_x + 1, width)), np.arange(max(min_y, 0), min(max_y + 1, height)))
    return x, y

@jit(nopython=True)
def calculate_barycentric_coordinates(x: np.ndarray, y: np.ndarray, e01: tuple, e12: tuple, e20: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w0 = e12[0] * (x + 0.5) + e12[1] * (y + 0.5) + e12[2]
    w1 = e20[0] * (x + 0.5) + e20[1] * (y + 0.5) + e20[2]
    w2 = e01[0] * (x + 0.5) + e01[1] * (y + 0.5) + e01[2]
    return w0, w1, w2

@jit(nopython=True)
def create_triangle_mask(w0: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    return (w0 >= 0) & (w1 >= 0) & (w2 >= 0)

@jit(nopython=True)
def calculate_depth_and_texture_coordinates(w0: np.ndarray, w1: np.ndarray, w2: np.ndarray, v0: tuple, v1: tuple, v2: tuple, vt0: tuple, vt1: tuple, vt2: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = w0 + w1 + w2
    w0, w1, w2 = w0 / w, w1 / w, w2 / w
    depth = w0 * v0[2] + w1 * v1[2] + w2 * v2[2]
    tx = w0 * vt0[0] + w1 * vt1[0] + w2 * vt2[0]
    ty = 1 - (w0 * vt0[1] + w1 * vt1[1] + w2 * vt2[1])
    return depth, tx, ty

def update_depth_buffer_and_image(depth_buffer: np.ndarray, image: np.ndarray, depth: np.ndarray,
                                  tx: np.ndarray, ty: np.ndarray, mask: np.ndarray, texture: np.ndarray,
                                  x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    update_mask = mask & (depth > depth_buffer[y, x])
    depth_buffer[y[update_mask], x[update_mask]] = depth[update_mask]
    color = texture[np.clip(ty[update_mask] * texture.shape[0], 0, texture.shape[0] - 1).astype(int),
                    np.clip(tx[update_mask] * texture.shape[1], 0, texture.shape[1] - 1).astype(int)]
    image[y[update_mask], x[update_mask]] = color
    return depth_buffer, image

def render_triangles(vertices: np.ndarray, texture_coords: np.ndarray, faces: np.ndarray, texture: np.ndarray, width: int, height: int) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    depth_buffer = np.full((height, width), float('-inf'))

    for face in faces:
        v0, v1, v2 = [vertices[i] for i, _ in face]
        vt0, vt1, vt2 = [texture_coords[i] for _, i in face if i is not None]

        # Convert vertex coordinates to screen space
        v0 = convert_to_screen_space(v0, width, height)
        v1 = convert_to_screen_space(v1, width, height)
        v2 = convert_to_screen_space(v2, width, height)

        # Calculate bounding box
        min_x, max_x, min_y, max_y = calculate_bounding_box(v0, v1, v2)

        # Calculate edge function coefficients
        e01, e12, e20 = calculate_edge_function_coefficients(v0, v1, v2)

        # Generate x and y coordinates
        x, y = generate_coordinates(min_x, max_x, min_y, max_y, width, height)

        # Calculate barycentric coordinates
        w0, w1, w2 = calculate_barycentric_coordinates(x, y, e01, e12, e20)

        # Create a mask for pixels inside the triangle
        mask = create_triangle_mask(w0, w1, w2)

        # Calculate depth and texture coordinates
        depth, tx, ty = calculate_depth_and_texture_coordinates(w0, w1, w2, v0, v1, v2, vt0, vt1, vt2)

        # Update depth buffer and image for pixels inside the triangle
        depth_buffer, image = update_depth_buffer_and_image(depth_buffer, image, depth, tx, ty, mask, texture, x, y)

    return image

def main():
    vertices, texture_coords, faces = parse_obj_file('african_head.obj')
    texture = np.array(Image.open('african_head_diffuse.tga'))
    
    image = render_triangles(vertices, texture_coords, faces, texture, 800, 600)
    Image.fromarray(image).save('output.png')

if __name__ == '__main__':
    main()