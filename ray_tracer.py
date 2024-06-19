import numpy as np
import math

def normalize(v):
    return v / np.linalg.norm(v)

def ray_sphere_intersect(orig, dir, center, radius):
    L = center - orig
    tca = np.dot(L, dir)
    d2 = np.dot(L, L) - tca * tca
    if d2 > radius * radius:
        return False, 0.0
    thc = math.sqrt(radius * radius - d2)
    t0 = tca - thc
    t1 = tca + thc
    if t0 < 0:
        t0 = t1
    if t0 < 0:
        return False, 0.0
    return True, t0

def scene_intersect(orig, dir, spheres):
    spheres_dist = float("inf")
    hit = np.zeros(3)
    N = np.zeros(3)
    material = np.zeros(3)
    for sphere in spheres:
        res, dist_i = ray_sphere_intersect(orig, dir, sphere[0], sphere[1])
        if res and dist_i < spheres_dist:
            spheres_dist = dist_i
            hit = orig + dir * dist_i
            N = normalize(hit - sphere[0])
            material = sphere[2]
    return spheres_dist < 1000, hit, N, material

def cast_ray(orig, dir, spheres, lights):
    hit, point, N, material = scene_intersect(orig, dir, spheres)
    if not hit:
        return np.array([0.2, 0.7, 0.8])  # background color

    diffuse_light_intensity = 0.0
    for light in lights:
        light_dir = normalize(light[0] - point)
        diffuse_light_intensity += light[1] * max(0.0, np.dot(light_dir, N))
    return material * diffuse_light_intensity

def render(spheres, lights):
    width = 1024
    height = 768
    fov = math.pi / 2.0
    framebuffer = np.zeros((height, width, 3))

    for j in range(height):
        for i in range(width):
            x = (2 * (i + 0.5) / width - 1) * math.tan(fov / 2.0) * width / height
            y = -(2 * (j + 0.5) / height - 1) * math.tan(fov / 2.0)
            dir = normalize(np.array([x, y, -1]))
            framebuffer[j, i] = cast_ray(np.zeros(3), dir, spheres, lights)

    framebuffer = np.clip(framebuffer, 0, 1)
    framebuffer = (framebuffer * 255).astype(np.uint8)
    with open("out.ppm", "wb") as f:
        f.write(f"P6\n{width} {height}\n255\n".encode())
        f.write(framebuffer.tobytes())

def main():
    ivory = np.array([0.4, 0.4, 0.3])
    red_rubber = np.array([0.3, 0.1, 0.1])

    spheres = [
        (np.array([-3, 0, -16]), 2, ivory),
        (np.array([-1.0, -1.5, -12]), 2, red_rubber),
        (np.array([1.5, -0.5, -18]), 3, red_rubber),
        (np.array([7, 5, -18]), 4, ivory)
    ]

    lights = [(np.array([-20, 20, 20]), 1.5)]

    render(spheres, lights)

if __name__ == "__main__":
    main()