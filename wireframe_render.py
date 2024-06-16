import numpy as np
from PIL import Image, ImageDraw
from collections import defaultdict

def load_obj(filename):
  """Loads vertices and faces from an .obj file."""
  vertices = []
  faces = []
  with open(filename, 'r') as f:
    for line in f:
      if line.startswith('v '):
        vertices.append([float(x) for x in line.split()[1:]])
      elif line.startswith('f '):
        faces.append([int(x.split('/')[0]) - 1 for x in line.split()[1:]])
  return np.array(vertices), faces

def project_vertices(vertices):
  """Projects 3D vertices to 2D using a simple orthogonal projection."""
  return vertices[:, :2]

def draw_wireframe(faces, vertices_2d, img_size=(512, 512)):
  """Draws a wireframe image."""
  img = Image.new("RGB", img_size, "white")
  draw = ImageDraw.Draw(img)
  for face in faces:
    for i in range(len(face)):
      v1 = vertices_2d[face[i]]
      v2 = vertices_2d[face[(i + 1) % len(face)]]
      draw.line((v1[0], v1[1], v2[0], v2[1]), fill="black")
  return img

def normalize_vertices(vertices, img_size=(512, 512)):
  """Normalizes vertices to fit within the image size."""
  min_coords = np.min(vertices, axis=0)
  max_coords = np.max(vertices, axis=0)
  scale = (np.array(img_size) * 0.8) / (max_coords - min_coords)
  vertices = (vertices - min_coords) * scale + (np.array(img_size) * 0.1)
  return vertices

def render_wireframe(obj_filename, output_filename="output.png"):
  """Renders a wireframe image from an .obj file."""
  vertices, faces = load_obj(obj_filename)
  vertices_2d = project_vertices(vertices)
  vertices_2d = normalize_vertices(vertices_2d)
  img = draw_wireframe(faces, vertices_2d)
  img.save(output_filename)

# Example usage:
render_wireframe("obj/african_head.obj", "output.png")