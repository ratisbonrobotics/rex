#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <cfloat>

#define CHECK_CUDA(call) { cudaError_t err = call; if (err != cudaSuccess) { printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); exit(1); } }

struct Vec3f {
    float x, y, z;
};

struct Vec2f {
    float u, v;
};

struct Triangle {
    Vec3f v[3];
    Vec2f uv[3];
};

__device__ float edge_function(float x0, float y0, float x1, float y1, float x2, float y2) {
    return (x2 - x0) * (y1 - y0) - (x1 - x0) * (y2 - y0);
}

__device__ Vec3f barycentric(float x, float y, const Vec3f& v0, const Vec3f& v1, const Vec3f& v2) {
    Vec3f u = {
        (v2.x - v0.x) * (v1.y - v0.y) - (v2.y - v0.y) * (v1.x - v0.x),
        (v2.x - v0.x) * (y - v0.y) - (v2.y - v0.y) * (x - v0.x),
        (x - v0.x) * (v1.y - v0.y) - (y - v0.y) * (v1.x - v0.x)
    };
    if (abs(u.x) < 1e-5) return {-1, 1, 1};
    return {1.f - (u.y + u.z) / u.x, u.y / u.x, u.z / u.x};
}

__global__ void rasterize_kernel(Triangle* triangles, int num_triangles, unsigned char* texture, int tex_width, int tex_height, unsigned char* output, float* depth_buffer, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float min_depth = FLT_MAX;
    Vec3f color = {0, 0, 0};

    for (int i = 0; i < num_triangles; i++) {
        Triangle tri = triangles[i];

        Vec3f bc_screen = barycentric(x, y, tri.v[0], tri.v[1], tri.v[2]);
        if (bc_screen.x < 0 || bc_screen.y < 0 || bc_screen.z < 0) continue;

        float depth = bc_screen.x * tri.v[0].z + bc_screen.y * tri.v[1].z + bc_screen.z * tri.v[2].z;

        if (depth < depth_buffer[y * width + x]) {
            depth_buffer[y * width + x] = depth;

            float u = bc_screen.x * tri.uv[0].u + bc_screen.y * tri.uv[1].u + bc_screen.z * tri.uv[2].u;
            float v = bc_screen.x * tri.uv[0].v + bc_screen.y * tri.uv[1].v + bc_screen.z * tri.uv[2].v;

            int tex_x = u * tex_width;
            int tex_y = v * tex_height;

            if (tex_x >= 0 && tex_x < tex_width && tex_y >= 0 && tex_y < tex_height) {
                color.x = texture[(tex_y * tex_width + tex_x) * 3 + 0] / 255.0f;
                color.y = texture[(tex_y * tex_width + tex_x) * 3 + 1] / 255.0f;
                color.z = texture[(tex_y * tex_width + tex_x) * 3 + 2] / 255.0f;
            } else {
                // Simple lighting if texture coordinates are out of bounds
                Vec3f normal = {0, 0, -1}; // Simplified normal
                Vec3f light_dir = {0, 0, -1}; // Light coming from the camera
                float intensity = max(0.f, normal.x * light_dir.x + normal.y * light_dir.y + normal.z * light_dir.z);
                color = {intensity, intensity, intensity};
            }
        }
    }

    output[(y * width + x) * 3 + 0] = static_cast<unsigned char>(color.x * 255);
    output[(y * width + x) * 3 + 1] = static_cast<unsigned char>(color.y * 255);
    output[(y * width + x) * 3 + 2] = static_cast<unsigned char>(color.z * 255);
}

void viewport_transform(std::vector<Triangle>& triangles, int width, int height) {
    float scale = std::min(width, height) * 0.8f;  // Scale to 80% of the smaller dimension
    float offsetX = width / 2.0f;
    float offsetY = height / 2.0f;

    for (auto& tri : triangles) {
        for (int i = 0; i < 3; i++) {
            // Flip z-coordinate to change handedness
            tri.v[i].z = -tri.v[i].z;

            // Scale and center the x and y coordinates
            tri.v[i].x = (tri.v[i].x * scale) + offsetX;
            tri.v[i].y = (-tri.v[i].y * scale) + offsetY;  // Negate y to flip vertically
        }
    }
}



void load_obj(const char* filename, std::vector<Triangle>& triangles) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        printf("Failed to open OBJ file\n");
        return;
    }

    std::vector<Vec3f> vertices;
    std::vector<Vec2f> texcoords;

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string type;
        iss >> type;

        if (type == "v") {
            Vec3f v;
            iss >> v.x >> v.y >> v.z;
            vertices.push_back(v);
        } else if (type == "vt") {
            Vec2f vt;
            iss >> vt.u >> vt.v;
            texcoords.push_back(vt);
        } else if (type == "f") {
            Triangle tri;
            for (int i = 0; i < 3; i++) {
                int v, vt, vn;
                char slash;
                iss >> v >> slash >> vt >> slash >> vn;
                tri.v[i] = vertices[v - 1];
                tri.uv[i] = texcoords[vt - 1];
            }
            triangles.push_back(tri);
        }
    }
}

// Add this function to flip the texture vertically
void flip_texture_vertically(unsigned char* texture, int width, int height, int channels) {
    int row_size = width * channels;
    unsigned char* temp_row = new unsigned char[row_size];
    
    for (int y = 0; y < height / 2; ++y) {
        unsigned char* top_row = texture + y * row_size;
        unsigned char* bottom_row = texture + (height - 1 - y) * row_size;
        
        // Swap rows
        memcpy(temp_row, top_row, row_size);
        memcpy(top_row, bottom_row, row_size);
        memcpy(bottom_row, temp_row, row_size);
    }
    
    delete[] temp_row;
}

int main() {
    const int width = 800;
    const int height = 600;

    std::vector<Triangle> triangles;
    load_obj("african_head.obj", triangles);
    viewport_transform(triangles, width, height);

    printf("Loaded %zu triangles\n", triangles.size());

    int tex_width, tex_height, tex_channels;
    unsigned char* texture = stbi_load("african_head_diffuse.tga", &tex_width, &tex_height, &tex_channels, 3);
    if (!texture) {
        printf("Failed to load texture\n");
        return 1;
    }
    printf("Loaded texture: %dx%d, %d channels\n", tex_width, tex_height, tex_channels);

    // Flip the texture vertically
    flip_texture_vertically(texture, tex_width, tex_height, 3);

    Triangle* d_triangles;
    unsigned char* d_texture;
    unsigned char* d_output;
    float* d_depth_buffer;

    CHECK_CUDA(cudaMalloc(&d_triangles, triangles.size() * sizeof(Triangle)));
    CHECK_CUDA(cudaMalloc(&d_texture, tex_width * tex_height * 3 * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_output, width * height * 3 * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_depth_buffer, width * height * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_triangles, triangles.data(), triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_texture, texture, tex_width * tex_height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_output, 0, width * height * 3 * sizeof(unsigned char))); // Clear output buffer
    CHECK_CUDA(cudaMemset(d_depth_buffer, 0x7f, width * height * sizeof(float))); // Initialize depth buffer to "infinity"

    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);

    rasterize_kernel<<<grid_size, block_size>>>(d_triangles, triangles.size(), d_texture, tex_width, tex_height, d_output, d_depth_buffer, width, height);

    unsigned char* output = new unsigned char[width * height * 3];
    CHECK_CUDA(cudaMemcpy(output, d_output, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    stbi_write_png("output.png", width, height, 3, output, width * 3);

    delete[] output;
    stbi_image_free(texture);
    CHECK_CUDA(cudaFree(d_triangles));
    CHECK_CUDA(cudaFree(d_texture));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_depth_buffer));

    return 0;
}