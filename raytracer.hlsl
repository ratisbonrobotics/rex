#pragma pack_matrix(column_major)
#ifdef SLANG_HLSL_ENABLE_NVAPI
#include "nvHLSLExtns.h"
#endif

#ifndef __DXC_VERSION_MAJOR
    // warning X3557: loop doesn't seem to do anything, forcing loop to unroll
    #pragma warning(disable: 3557)
#endif


#line 67 "raytracer.slang"
struct EntryPointParams_0
{
    int num_triangles_0;
    int2 dimensions_0;
};


#line 31
cbuffer entryPointParams_0 : register(b0)
{
    EntryPointParams_0 entryPointParams_0;
}

#line 1
struct Vec3f_0
{
    float x_0;
    float y_0;
    float z_0;
};


#line 22
struct Vec2f_0
{
    float u_0;
    float v_0;
};


#line 27
struct Triangle_0
{
    Vec3f_0  v_1[int(3)];
    Vec2f_0  uv_0[int(3)];
    Vec3f_0 normal_0;
};


#line 67
StructuredBuffer<Triangle_0 > entryPointParams_triangles_0 : register(t0);


#line 2373 "hlsl.meta.slang"
Texture2D<float3 > entryPointParams_texture_0 : register(t1);


#line 2914
RWTexture2D<float3 > entryPointParams_output_0 : register(u0);


#line 6 "raytracer.slang"
Vec3f_0 make_vec3f_0(float x_1, float y_1, float z_1)
{
    Vec3f_0 v_2;
    v_2.x_0 = x_1;
    v_2.y_0 = y_1;
    v_2.z_0 = z_1;
    return v_2;
}


#line 20
Vec3f_0 normalize_0(Vec3f_0 v_3)
{

#line 20
    float _S1 = v_3.x_0;

#line 20
    float _S2 = v_3.y_0;

#line 20
    float _S3 = v_3.z_0;

#line 20
    float l_0 = sqrt(_S1 * _S1 + _S2 * _S2 + _S3 * _S3);

#line 20
    return make_vec3f_0(v_3.x_0 / l_0, v_3.y_0 / l_0, v_3.z_0 / l_0);
}


#line 16
Vec3f_0 x2D_0(Vec3f_0 a_0, Vec3f_0 b_0)
{

#line 16
    return make_vec3f_0(a_0.x_0 - b_0.x_0, a_0.y_0 - b_0.y_0, a_0.z_0 - b_0.z_0);
}

Vec3f_0 cross_0(Vec3f_0 a_1, Vec3f_0 b_1)
{

#line 19
    return make_vec3f_0(a_1.y_0 * b_1.z_0 - a_1.z_0 * b_1.y_0, a_1.z_0 * b_1.x_0 - a_1.x_0 * b_1.z_0, a_1.x_0 * b_1.y_0 - a_1.y_0 * b_1.x_0);
}


#line 18
float dot_0(Vec3f_0 a_2, Vec3f_0 b_2)
{

#line 18
    return a_2.x_0 * b_2.x_0 + a_2.y_0 * b_2.y_0 + a_2.z_0 * b_2.z_0;
}


#line 34
struct Ray_0
{
    Vec3f_0 origin_0;
    Vec3f_0 direction_0;
};

bool ray_triangle_intersect_0(Ray_0 ray_0, Triangle_0 triangle_0, out float t_0, out float u_1, out float v_4)
{
    Vec3f_0 edge1_0 = x2D_0(triangle_0.v_1[int(1)], triangle_0.v_1[int(0)]);
    Vec3f_0 edge2_0 = x2D_0(triangle_0.v_1[int(2)], triangle_0.v_1[int(0)]);
    Vec3f_0 h_0 = cross_0(ray_0.direction_0, edge2_0);
    float a_3 = dot_0(edge1_0, h_0);

#line 45
    bool _S4;

    if(a_3 > -0.00000999999974738)
    {

#line 47
        _S4 = a_3 < 0.00000999999974738;

#line 47
    }
    else
    {

#line 47
        _S4 = false;

#line 47
    }

#line 47
    if(_S4)
    {

#line 47
        return false;
    }
    float f_0 = 1.0 / a_3;
    Vec3f_0 s_0 = x2D_0(ray_0.origin_0, triangle_0.v_1[int(0)]);
    float _S5 = f_0 * dot_0(s_0, h_0);

#line 51
    u_1 = _S5;

    if(_S5 < 0.0)
    {

#line 53
        _S4 = true;

#line 53
    }
    else
    {

#line 53
        _S4 = u_1 > 1.0;

#line 53
    }

#line 53
    if(_S4)
    {

#line 53
        return false;
    }
    Vec3f_0 q_0 = cross_0(s_0, edge1_0);
    float _S6 = f_0 * dot_0(ray_0.direction_0, q_0);

#line 56
    v_4 = _S6;

    if(_S6 < 0.0)
    {

#line 58
        _S4 = true;

#line 58
    }
    else
    {

#line 58
        _S4 = u_1 + v_4 > 1.0;

#line 58
    }

#line 58
    if(_S4)
    {

#line 58
        return false;
    }
    float _S7 = f_0 * dot_0(edge2_0, q_0);

#line 60
    t_0 = _S7;

    return _S7 > 0.00000999999974738;
}


#line 17
Vec3f_0 mul_scalar_0(Vec3f_0 v_5, float f_1)
{

#line 17
    return make_vec3f_0(v_5.x_0 * f_1, v_5.y_0 * f_1, v_5.z_0 * f_1);
}


#line 67
[numthreads(16, 16, 1)]
void ray_trace_kernel(uint3 SV_DispatchThreadID_0 : SV_DispatchThreadID)
{

#line 75
    uint x_2 = SV_DispatchThreadID_0.x;
    uint y_2 = SV_DispatchThreadID_0.y;

#line 76
    bool _S8;

    if(x_2 >= uint(entryPointParams_0.dimensions_0.x))
    {

#line 78
        _S8 = true;

#line 78
    }
    else
    {

#line 78
        _S8 = y_2 >= uint(entryPointParams_0.dimensions_0.y);

#line 78
    }

#line 78
    if(_S8)
    {

#line 78
        return;
    }


    float tan_fov_0 = tan(0.39269876480102539);

    float camera_x_0 = (2.0 * (float(x_2) + 0.5) / float(entryPointParams_0.dimensions_0.x) - 1.0) * (float(entryPointParams_0.dimensions_0.x) / float(entryPointParams_0.dimensions_0.y)) * tan_fov_0;
    float camera_y_0 = (1.0 - 2.0 * (float(y_2) + 0.5) / float(entryPointParams_0.dimensions_0.y)) * tan_fov_0;

    Ray_0 ray_1;
    ray_1.origin_0 = make_vec3f_0(0.0, 0.0, 3.0);
    ray_1.direction_0 = normalize_0(make_vec3f_0(camera_x_0, camera_y_0, -1.0));

    Vec3f_0 _S9 = make_vec3f_0(0.20000000298023224, 0.20000000298023224, 0.20000000298023224);

#line 123
    uint2 _S10 = uint2(int2(int(x_2), int(y_2)));

#line 115
    Vec3f_0 light_dir_0 = normalize_0(make_vec3f_0(1.0, 1.0, 1.0));

#line 115
    float closest_t_0 = 3.4028234663852886e+38;

#line 115
    Vec3f_0 color_0 = _S9;

#line 115
    int i_0 = int(0);

#line 115
    for(;;)
    {

#line 94
        if(i_0 < entryPointParams_0.num_triangles_0)
        {
        }
        else
        {

#line 94
            break;
        }
        float t_1;

#line 96
        float u_2;

#line 96
        float v_6;
        bool _S11 = ray_triangle_intersect_0(ray_1, entryPointParams_triangles_0.Load(i_0), t_1, u_2, v_6);

#line 97
        if(_S11)
        {

#line 97
            _S8 = t_1 < closest_t_0;

#line 97
        }
        else
        {

#line 97
            _S8 = false;

#line 97
        }

#line 97
        if(_S8)
        {
            float _S12 = t_1;

            float w_0 = 1.0 - u_2 - v_6;

            float tex_u_0 = w_0 * entryPointParams_triangles_0.Load(i_0).uv_0[int(0)].u_0 + u_2 * entryPointParams_triangles_0.Load(i_0).uv_0[int(1)].u_0 + v_6 * entryPointParams_triangles_0.Load(i_0).uv_0[int(2)].u_0;
            float tex_v_0 = w_0 * entryPointParams_triangles_0.Load(i_0).uv_0[int(0)].v_0 + u_2 * entryPointParams_triangles_0.Load(i_0).uv_0[int(1)].v_0 + v_6 * entryPointParams_triangles_0.Load(i_0).uv_0[int(2)].v_0;

            uint width_0;

#line 106
            uint height_0;
            entryPointParams_texture_0.GetDimensions(width_0, height_0);
            int _S13 = int(tex_u_0 * float(width_0));

#line 108
            int _S14 = int((1.0 - tex_v_0) * float(height_0));

#line 108
            int2 tex_coords_0 = int2(_S13, _S14);

#line 108
            bool _S15;

            if(_S13 >= int(0))
            {

#line 110
                _S15 = _S14 >= int(0);

#line 110
            }
            else
            {

#line 110
                _S15 = false;

#line 110
            }

#line 110
            bool _S16;

#line 110
            if(_S15)
            {

#line 110
                _S16 = _S13 < int(width_0);

#line 110
            }
            else
            {

#line 110
                _S16 = false;

#line 110
            }

#line 110
            bool _S17;

#line 110
            if(_S16)
            {

#line 110
                _S17 = _S14 < int(height_0);

#line 110
            }
            else
            {

#line 110
                _S17 = false;

#line 110
            }

#line 110
            Vec3f_0 color_1;

#line 110
            if(_S17)
            {
                float3 tex_color_0 = entryPointParams_texture_0.Load(int3(tex_coords_0, int(0)));

#line 112
                color_1 = mul_scalar_0(make_vec3f_0(tex_color_0.x, tex_color_0.y, tex_color_0.z), 0.30000001192092896 + 0.69999998807907104 * max(0.0, dot_0(entryPointParams_triangles_0.Load(i_0).normal_0, light_dir_0)));

#line 112
            }
            else
            {

#line 112
                color_1 = color_0;

#line 112
            }

#line 112
            closest_t_0 = _S12;

#line 112
            color_0 = color_1;

#line 112
        }

#line 94
        i_0 = i_0 + int(1);

#line 94
    }

#line 123
    entryPointParams_output_0[_S10] = float3(color_0.x_0, color_0.y_0, color_0.z_0);
    return;
}

