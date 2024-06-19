import numpy as np
import math

class Vec3f:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Vec3f(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vec3f(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other):
        if isinstance(other, Vec3f):
            return self.x * other.x + self.y * other.y + self.z * other.z
        else:
            return Vec3f(self.x * other, self.y * other, self.z * other)

    def __truediv__(self, other):
        return Vec3f(self.x / other, self.y / other, self.z / other)

    def __getitem__(self, index):
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        elif index == 2:
            return self.z
        else:
            raise IndexError("Index out of range")

    def __setitem__(self, index, value):
        if index == 0:
            self.x = value
        elif index == 1:
            self.y = value
        elif index == 2:
            self.z = value
        else:
            raise IndexError("Index out of range")

    def norm(self):
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalize(self, l=1.0):
        norm = self.norm()
        self.x = self.x * (l / norm)
        self.y = self.y * (l / norm)
        self.z = self.z * (l / norm)
        return self

def cross(v1, v2):
    return Vec3f(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x)

class Light:
    def __init__(self, position, intensity):
        self.position = position
        self.intensity = intensity

class Material:
    def __init__(self, color=Vec3f()):
        self.diffuse_color = color

class Sphere:
    def __init__(self, center, radius, material):
        self.center = center
        self.radius = radius
        self.material = material

    def ray_intersect(self, orig, dir):
        L = self.center - orig
        tca = L * dir
        d2 = L * L - tca * tca
        if d2 > self.radius * self.radius:
            return False, 0.0
        thc = math.sqrt(self.radius * self.radius - d2)
        t0 = tca - thc
        t1 = tca + thc
        if t0 < 0:
            t0 = t1
        if t0 < 0:
            return False, 0.0
        return True, t0

def scene_intersect(orig, dir, spheres):
    spheres_dist = float("inf")
    hit = Vec3f()
    N = Vec3f()
    material = Material()
    for sphere in spheres:
        res, dist_i = sphere.ray_intersect(orig, dir)
        if res and dist_i < spheres_dist:
            spheres_dist = dist_i
            hit = orig + dir * dist_i
            N = (hit - sphere.center).normalize()
            material = sphere.material
    return spheres_dist < 1000, hit, N, material

def cast_ray(orig, dir, spheres, lights):
    hit, point, N, material = scene_intersect(orig, dir, spheres)
    if not hit:
        return Vec3f(0.2, 0.7, 0.8)  # background color

    diffuse_light_intensity = 0.0
    for light in lights:
        light_dir = (light.position - point).normalize()
        diffuse_light_intensity += light.intensity * max(0.0, light_dir * N)
    return material.diffuse_color * diffuse_light_intensity

def render(spheres, lights):
    width = 1024
    height = 768
    fov = math.pi / 2.0
    framebuffer = [Vec3f() for _ in range(width * height)]

    for j in range(height):
        for i in range(width):
            x = (2 * (i + 0.5) / width - 1) * math.tan(fov / 2.0) * width / height
            y = -(2 * (j + 0.5) / height - 1) * math.tan(fov / 2.0)
            dir = Vec3f(x, y, -1).normalize()
            framebuffer[i + j * width] = cast_ray(Vec3f(0, 0, 0), dir, spheres, lights)

    with open("out.ppm", "wb") as f:
        f.write(f"P6\n{width} {height}\n255\n".encode())
        for pixel in framebuffer:
            for color in pixel:
                f.write(bytes([int(255 * max(0.0, min(1.0, color)))]))

def main():
    ivory = Material(Vec3f(0.4, 0.4, 0.3))
    red_rubber = Material(Vec3f(0.3, 0.1, 0.1))

    spheres = [
        Sphere(Vec3f(-3, 0, -16), 2, ivory),
        Sphere(Vec3f(-1.0, -1.5, -12), 2, red_rubber),
        Sphere(Vec3f(1.5, -0.5, -18), 3, red_rubber),
        Sphere(Vec3f(7, 5, -18), 4, ivory)
    ]

    lights = [Light(Vec3f(-20, 20, 20), 1.5)]

    render(spheres, lights)

if __name__ == "__main__":
    main()