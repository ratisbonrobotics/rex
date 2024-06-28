#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

struct Vec3f {
    float x, y, z;
};

struct Vec2f {
    float u, v;
};

struct Face {
    int v[3];
    int vt[3];
};

__global__ void renderTriangles(Vec3f* d_vertices, Vec2f* d_texture_coords, Face* d_faces, 
                                unsigned char* d_texture, int texture_width, int texture_height,
                                unsigned char* d_image, float* d_depth_buffer,
                                int width, int height, int num_faces) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_faces) return;

    Face face = d_faces[idx];
    Vec3f v0 = d_vertices[face.v[0]];
    Vec3f v1 = d_vertices[face.v[1]];
    Vec3f v2 = d_vertices[face.v[2]];
    Vec2f vt0 = d_texture_coords[face.vt[0]];
    Vec2f vt1 = d_texture_coords[face.vt[1]];
    Vec2f vt2 = d_texture_coords[face.vt[2]];

    // Convert vertex coordinates to screen space
    v0.x = (v0.x + 1) * width / 2;
    v0.y = (1 - v0.y) * height / 2;
    v1.x = (v1.x + 1) * width / 2;
    v1.y = (1 - v1.y) * height / 2;
    v2.x = (v2.x + 1) * width / 2;
    v2.y = (1 - v2.y) * height / 2;

    // Calculate bounding box
    int min_x = max(0, (int)min(min(v0.x, v1.x), v2.x));
    int max_x = min(width - 1, (int)max(max(v0.x, v1.x), v2.x));
    int min_y = max(0, (int)min(min(v0.y, v1.y), v2.y));
    int max_y = min(height - 1, (int)max(max(v0.y, v1.y), v2.y));

    // Calculate edge function coefficients
    float e01_x = v1.y - v0.y, e01_y = v0.x - v1.x, e01_z = v1.x * v0.y - v0.x * v1.y;
    float e12_x = v2.y - v1.y, e12_y = v1.x - v2.x, e12_z = v2.x * v1.y - v1.x * v2.y;
    float e20_x = v0.y - v2.y, e20_y = v2.x - v0.x, e20_z = v0.x * v2.y - v2.x * v0.y;

    for (int y = min_y; y <= max_y; y++) {
        for (int x = min_x; x <= max_x; x++) {
            // Calculate barycentric coordinates
            float w0 = e12_x * (x + 0.5f) + e12_y * (y + 0.5f) + e12_z;
            float w1 = e20_x * (x + 0.5f) + e20_y * (y + 0.5f) + e20_z;
            float w2 = e01_x * (x + 0.5f) + e01_y * (y + 0.5f) + e01_z;

            if (w0 >= 0 && w1 >= 0 && w2 >= 0) {
                // Pixel is inside the triangle
                float w = w0 + w1 + w2;
                w0 /= w; w1 /= w; w2 /= w;
                float depth = w0 * v0.z + w1 * v1.z + w2 * v2.z;

                if (depth > d_depth_buffer[y * width + x]) {
                    d_depth_buffer[y * width + x] = depth;

                    // Calculate texture coordinates
                    float tx = w0 * vt0.u + w1 * vt1.u + w2 * vt2.u;
                    float ty = 1 - (w0 * vt0.v + w1 * vt1.v + w2 * vt2.v);

                    // Sample texture
                    int tex_x = max(0, min((int)(tx * texture_width), texture_width - 1));
                    int tex_y = max(0, min((int)(ty * texture_height), texture_height - 1));
                    int tex_idx = (tex_y * texture_width + tex_x) * 3;

                    // Set pixel color
                    int img_idx = (y * width + x) * 3;
                    d_image[img_idx] = d_texture[tex_idx];
                    d_image[img_idx + 1] = d_texture[tex_idx + 1];
                    d_image[img_idx + 2] = d_texture[tex_idx + 2];
                }
            }
        }
    }
}

void parseObjFile(const char* filename, std::vector<Vec3f>& vertices, std::vector<Vec2f>& texture_coords, std::vector<Face>& faces) {
    std::ifstream file(filename);
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
            texture_coords.push_back(vt);
        } else if (type == "f") {
            Face f;
            for (int i = 0; i < 3; i++) {
                std::string vertex;
                iss >> vertex;
                sscanf(vertex.c_str(), "%d/%d", &f.v[i], &f.vt[i]);
                f.v[i]--; // OBJ indices start at 1
                f.vt[i]--;
            }
            faces.push_back(f);
        }
    }
}

int main() {
    std::vector<Vec3f> vertices;
    std::vector<Vec2f> texture_coords;
    std::vector<Face> faces;
    parseObjFile("african_head.obj", vertices, texture_coords, faces);

    int width, height, channels;
    unsigned char* texture = stbi_load("african_head_diffuse.tga", &width, &height, &channels, 3);
    if (!texture) {
        fprintf(stderr, "Failed to load texture\n");
        return 1;
    }

    int img_width = 800, img_height = 600;
    unsigned char* image = new unsigned char[img_width * img_height * 3];
    float* depth_buffer = new float[img_width * img_height];
    for (int i = 0; i < img_width * img_height; i++) {
        depth_buffer[i] = -INFINITY;
    }

    // Allocate device memory
    Vec3f* d_vertices;
    Vec2f* d_texture_coords;
    Face* d_faces;
    unsigned char* d_texture;
    unsigned char* d_image;
    float* d_depth_buffer;

    cudaMalloc(&d_vertices, vertices.size() * sizeof(Vec3f));
    cudaMalloc(&d_texture_coords, texture_coords.size() * sizeof(Vec2f));
    cudaMalloc(&d_faces, faces.size() * sizeof(Face));
    cudaMalloc(&d_texture, width * height * 3 * sizeof(unsigned char));
    cudaMalloc(&d_image, img_width * img_height * 3 * sizeof(unsigned char));
    cudaMalloc(&d_depth_buffer, img_width * img_height * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_vertices, vertices.data(), vertices.size() * sizeof(Vec3f), cudaMemcpyHostToDevice);
    cudaMemcpy(d_texture_coords, texture_coords.data(), texture_coords.size() * sizeof(Vec2f), cudaMemcpyHostToDevice);
    cudaMemcpy(d_faces, faces.data(), faces.size() * sizeof(Face), cudaMemcpyHostToDevice);
    cudaMemcpy(d_texture, texture, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_depth_buffer, depth_buffer, img_width * img_height * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (faces.size() + threadsPerBlock - 1) / threadsPerBlock;
    renderTriangles<<<blocksPerGrid, threadsPerBlock>>>(d_vertices, d_texture_coords, d_faces, 
                                                        d_texture, width, height,
                                                        d_image, d_depth_buffer,
                                                        img_width, img_height, faces.size());

    // Copy result back to host
    cudaMemcpy(image, d_image, img_width * img_height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Save image
    stbi_write_png("output.png", img_width, img_height, 3, image, img_width * 3);

    // Free device memory
    cudaFree(d_vertices);
    cudaFree(d_texture_coords);
    cudaFree(d_faces);
    cudaFree(d_texture);
    cudaFree(d_image);
    cudaFree(d_depth_buffer);

    // Free host memory
    delete[] image;
    delete[] depth_buffer;
    stbi_image_free(texture);

    return 0;
}