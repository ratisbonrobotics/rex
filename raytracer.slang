struct Vec3f
{
    float x, y, z;

    __init(float x = 0, float y = 0, float z = 0)
    {
        this.x = x;
        this.y = y;
        this.z = z;
    }

    Vec3f operator+(Vec3f v) { return Vec3f(x + v.x, y + v.y, z + v.z); }
    Vec3f operator-(Vec3f v) { return Vec3f(x - v.x, y - v.y, z - v.z); }
    Vec3f operator*(float f) { return Vec3f(x * f, y * f, z * f); }
    float dot(Vec3f v) { return x * v.x + y * v.y + z * v.z; }
    Vec3f cross(Vec3f v) { return Vec3f(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x); }
    Vec3f normalize() { float l = sqrt(x * x + y * y + z * z); return Vec3f(x / l, y / l, z / l); }
}

struct Vec2f
{
    float u, v;
}

struct Triangle
{
    Vec3f v[3];
    Vec2f uv[3];
    Vec3f normal;
}

struct Ray
{
    Vec3f origin;
    Vec3f direction;
}

bool ray_triangle_intersect(Ray ray, Triangle triangle, out float t, out float u, out float v)
{
    let edge1 = triangle.v[1] - triangle.v[0];
    let edge2 = triangle.v[2] - triangle.v[0];
    let h = ray.direction.cross(edge2);
    let a = edge1.dot(h);

    if (a > -1e-5 && a < 1e-5) return false;

    let f = 1.0 / a;
    let s = ray.origin - triangle.v[0];
    u = f * s.dot(h);

    if (u < 0.0 || u > 1.0) return false;

    let q = s.cross(edge1);
    v = f * ray.direction.dot(q);

    if (v < 0.0 || u + v > 1.0) return false;

    t = f * edge2.dot(q);

    return t > 1e-5;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void ray_trace_kernel(
    uniform StructuredBuffer<Triangle> triangles : register(t0),
    uniform Texture2D<float3> texture : register(t1),
    uniform RWTexture2D<float3> output : register(u0),
    uniform int num_triangles,
    uniform int2 dimensions,
    uint3 dispatchThreadID : SV_DispatchThreadID)
{
    let x = dispatchThreadID.x;
    let y = dispatchThreadID.y;

    if (x >= uint(dimensions.x) || y >= uint(dimensions.y)) return;

    let aspect_ratio = float(dimensions.x) / dimensions.y;
    let fov = 3.14159 / 4.0;
    let tan_fov = tan(fov / 2.0);

    let camera_x = ((2.0 * (x + 0.5) / dimensions.x - 1.0) * aspect_ratio) * tan_fov;
    let camera_y = (1.0 - 2.0 * (y + 0.5) / dimensions.y) * tan_fov;

    Ray ray;
    ray.origin = Vec3f(0, 0, 3);
    ray.direction = Vec3f(camera_x, camera_y, -1).normalize();

    Vec3f color = Vec3f(0.2, 0.2, 0.2);
    float closest_t = 3.402823466e+38; // FLT_MAX

    for (int i = 0; i < num_triangles; i++)
    {
        float t, u, v;
        if (ray_triangle_intersect(ray, triangles[i], t, u, v) && t < closest_t)
        {
            closest_t = t;

            float w = 1.0 - u - v;

            float tex_u = w * triangles[i].uv[0].u + u * triangles[i].uv[1].u + v * triangles[i].uv[2].u;
            float tex_v = w * triangles[i].uv[0].v + u * triangles[i].uv[1].v + v * triangles[i].uv[2].v;

            Vec3f tex_color = texture.SampleLevel(texture.GetSampler(), float2(tex_u, tex_v), 0).xyz;

            Vec3f light_dir = Vec3f(1, 1, 1).normalize();
            float diffuse = max(0.0, triangles[i].normal.dot(light_dir));

            color = tex_color * (0.3 + 0.7 * diffuse);
        }
    }

    output[int2(x, y)] = float3(color.x, color.y, color.z);
}