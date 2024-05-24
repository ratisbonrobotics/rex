import jax
import jax.numpy as jnp
from typing import Tuple, List
import numpy as np  # Add this import for NumPy conversion

def load_obj(file_path: str) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Loads a 3D model from an .obj file.

    Args:
        file_path: Path to the .obj file.

    Returns:
        A tuple containing the vertices and faces of the model.
    """
    vertices = []
    faces = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertices.append([float(x) for x in line.split()[1:]])
            elif line.startswith('f '):
                faces.append([int(x.split('/')[0]) - 1 for x in line.split()[1:]])
    return jnp.array(vertices), jnp.array(faces)

def orthographic_projection(vertices: jnp.ndarray) -> jnp.ndarray:
    """Performs orthographic projection on 3D vertices.

    Args:
        vertices: A (N, 3) array of 3D vertices.

    Returns:
        A (N, 2) array of 2D projected vertices.
    """
    return vertices[:, :2]

def rasterize(vertices: jnp.ndarray, faces: jnp.ndarray,
              width: int, height: int) -> jnp.ndarray:
    """Rasterizes the 3D model onto a 2D image plane.

    Args:
        vertices: A (N, 2) array of 2D projected vertices.
        faces: A (M, 3) array of triangle faces.
        width: Width of the output image.
        height: Height of the output image.

    Returns:
        A (height, width) binary image representing the rasterized model.
    """
    image = jnp.zeros((height, width), dtype=bool)

    # Iterate through each triangle face
    for face in faces:
        # Extract the 2D coordinates of the triangle's vertices
        v1, v2, v3 = vertices[face]

        # Determine the bounding box of the triangle
        min_x = int(max(0, min(v1[0], v2[0], v3[0])))
        max_x = int(min(width - 1, max(v1[0], v2[0], v3[0])))
        min_y = int(max(0, min(v1[1], v2[1], v3[1])))
        max_y = int(min(height - 1, max(v1[1], v2[1], v3[1])))

        # Iterate through each pixel in the bounding box
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                # Perform a point-in-triangle test
                p = jnp.array([x, y])
                d1 = (p[1] - v2[1]) * (v3[0] - v2[0]) - (p[0] - v2[0]) * (v3[1] - v2[1])
                d2 = (p[1] - v3[1]) * (v1[0] - v3[0]) - (p[0] - v3[0]) * (v1[1] - v3[1])
                d3 = (p[1] - v1[1]) * (v2[0] - v1[0]) - (p[0] - v1[0]) * (v2[1] - v1[1])

                # If the point is inside the triangle, set the pixel value to 1
                if (d1 >= 0 and d2 >= 0 and d3 >= 0) or (d1 <= 0 and d2 <= 0 and d3 <= 0):
                    image = image.at[y, x].set(True)

    return image

# Example usage:
if __name__ == "__main__":
    vertices, faces = load_obj('bunny.obj')
    vertices_2d = orthographic_projection(vertices)
    image = rasterize(vertices_2d, faces, width=512, height=512)
    
    # Convert JAX array to NumPy array
    np_image = np.array(image, dtype=np.uint8) * 255  # Ensure it's 0 or 255 for binary image

    # Save or display the image using a suitable library
    # For example, using matplotlib:
    import matplotlib.pyplot as plt
    plt.imshow(np_image, cmap='gray')
    plt.savefig("out.png")
