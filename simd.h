#pragma once

#include <immintrin.h>
#include <array>

namespace simd
{

struct float16
{
    __m512 xs;

    static float16 zero() { return float16(_mm512_setzero_ps()); }
    static float16 one() { return float16(1.0f); }

    float16() {};

    float16(const float16& another) : float16(another.xs)
    {
    }

    float16& operator=(const float16& another)
    {
        this->xs = another.xs;
        return *this;
    }

    float16(float value) : xs(_mm512_set1_ps(value))
    {
    }

    float16(const std::array<float, 16>& values)
        : xs(_mm512_set_ps(
            values[15],
            values[14],
            values[13],
            values[12],
            values[11],
            values[10],
            values[9],
            values[8],
            values[7],
            values[6],
            values[5],
            values[4],
            values[3],
            values[2],
            values[1],
            values[0]))
    {
    }

    float16(const float* alignedArray)
        : xs(_mm512_load_ps(alignedArray))
    {
    }

    float16(const __m512& value)  : xs(value)
    {
    }

    float16 operator-() const { return _mm512_mul_ps(xs, _mm512_set1_ps(-1.0f)); }

    float16 operator+(const float16& rhs) const { return _mm512_add_ps(xs, rhs.xs); }

    float16 operator-(const float16& rhs) const { return _mm512_sub_ps(xs, rhs.xs); }

    float16 operator*(const float16& rhs) const { return _mm512_mul_ps(xs, rhs.xs); }

    float16 operator*(float rhs) const { return _mm512_mul_ps(xs, _mm512_set1_ps(rhs)); }

    float16 operator/(const float16& rhs) const { return _mm512_div_ps(xs, rhs.xs); }

    float16 operator/(float rhs) const { return _mm512_div_ps(xs, _mm512_set1_ps(rhs)); }

    float16 operator+=(const float16& rhs)
    {
        *this = *this + rhs;
        return *this;
    }

    float16 operator-=(const float16& rhs)
    {
        *this = *this - rhs;
        return *this;
    }

    float16 operator*=(float16 rhs)
    {
        *this = *this * rhs;
        return *this;
    }

    float16 operator/=(float16 rhs)
    {
        *this = *this / rhs;
        return *this;
    }

    std::array<float, 16> toArray() const
    {
        std::array<float, 16> arr;
        _mm512_store_ps(arr.data(), xs);
        return arr;
    }

    std::array<int, 16> toIntArray() const
    {
        __m512i intVector{ _mm512_cvttps_epi32(xs) };
        std::array<int, 16> arr;
        _mm512_storeu_si512((__m512i*)arr.data(), intVector);
        return arr;
    }

    uint16_t operator<(const float16& rhs) const { return _mm512_cmp_ps_mask(xs, rhs.xs, _CMP_LT_OQ); }
    uint16_t operator<=(const float16& rhs) const { return _mm512_cmp_ps_mask(xs, rhs.xs, _CMP_LE_OQ); }
    uint16_t operator>(const float16& rhs) const { return _mm512_cmp_ps_mask(xs, rhs.xs, _CMP_GT_OQ); }
    uint16_t operator>=(const float16& rhs) const { return _mm512_cmp_ps_mask(xs, rhs.xs, _CMP_GE_OQ); }
    uint16_t operator==(const float16& rhs) const { return _mm512_cmp_ps_mask(xs, rhs.xs, _CMP_EQ_OQ); }
};

inline
float16 sqrt(const float16& x)
{
    return float16(_mm512_sqrt_ps(x.xs));
}

inline 
float16 select(const float16& false_value, const float16& true_value, uint16_t mask)
{
    return _mm512_mask_blend_ps(mask, false_value.xs, true_value.xs);
}

inline
float16 move(const float16& dst, const float16& src, uint16_t mask)
{
    return _mm512_mask_mov_ps(dst.xs, mask, src.xs);
}

float16 dot(
    const float16& x0,
    const float16& y0,
    const float16& z0,
    const float16& x1,
    const float16& y1,
    const float16& z1
)
{
    const float16 x = x0 * x1;
    const float16 y = y0 * y1;
    const float16 z = z0 * z1;
    return x + y + z;
}

struct SimdSphere
{
    float16 radius;
    float16 x;
    float16 y;
    float16 z;
    float16 index;
    uint16_t active;

    SimdSphere() = default;

    SimdSphere(const float16& r, const float16& a, const float16& b, const float16& c, const float16& i, uint16_t act) :
        radius(r), x(a), y(b), z(c), index(i), active(act)
    {

    }
};

struct SimdRay
{
    float16 org_x;
    float16 org_y;
    float16 org_z;
    float16 dir_x;
    float16 dir_y;
    float16 dir_z;
    float16 distance_to_intersection;
    float16 hit_index;

    SimdRay(const utility::Ray& ray) :
        org_x(ray.org[0]),
        org_y(ray.org[1]),
        org_z(ray.org[2]),
        dir_x(ray.dir[0]),
        dir_y(ray.dir[1]),
        dir_z(ray.dir[2]),
        distance_to_intersection(ray.distance_to_intersection),
        hit_index(-float16::one())
    {

    }
};

utility::HitPoint merge(utility::Ray& ray, const SimdRay& simd_ray)
{
    utility::HitPoint hp;
    const auto distances = simd_ray.distance_to_intersection.toArray();
    const auto indexes = simd_ray.hit_index.toArray();

    float min_distance = 1e+50;
    int index = -1;
    for (int i = 0; i < 16; ++i)
    {
        if (indexes[i] >= 0 && min_distance > distances[i])
        {
            min_distance = distances[i];
            index = indexes[i];
        }
    }

    if (index == -1)
    {
        return {};
    }

    ray.distance_to_intersection = min_distance;
    hp.hit = true;
    hp.position = ray.intersectedPoint();
    hp.index = index;
    hp.distance = min_distance;
    return hp;
}

uint16_t checkIntersectionToSphere(
    const SimdSphere& sphere,
    SimdRay& ray,
    float max_distance = std::numeric_limits<float>::infinity(),
    uint16_t active_lane = 0xffff
)
{
    const float16 relative_position_x = sphere.x - ray.org_x;
    const float16 relative_position_y = sphere.y - ray.org_y;
    const float16 relative_position_z = sphere.z - ray.org_z;

    const float16 b = -dot(
        relative_position_x,
        relative_position_y,
        relative_position_z,
        ray.dir_x,
        ray.dir_y,
        ray.dir_z);
    const float16 c = dot(
        relative_position_x,
        relative_position_y,
        relative_position_z,
        relative_position_x,
        relative_position_y,
        relative_position_z
    ) - sphere.radius * sphere.radius;

    const float16 d = b * b - c;

    const uint16_t cmp0 = d >= float16::zero();
    active_lane &= cmp0;
    if (active_lane == 0)
    {
        return 0;
    }

    const float16 sqrt_d = sqrt(d);
    const uint16_t cmp1 = b > float16::zero();
    
    const float16 s = select(float16::one(), -float16::one(), cmp1);
    const float16 tmp = -b + s * sqrt_d;

    const uint16_t cmp2 = ~(tmp == float16::zero());
    active_lane &= cmp2;
    if (active_lane == 0)
    {
        return 0;
    }

    const float16 x1 = c / tmp;
    const float16 x2 = tmp;

    const float16 eps(EPS);
    const float16 original_distance = ray.distance_to_intersection;

    {
        const uint16_t x1_cmp0 = eps < x1;
        const uint16_t x1_cmp1 = (x1 < ray.distance_to_intersection) & (x1 < max_distance);
        const auto mask = active_lane & (x1_cmp0 & x1_cmp1);
        if (mask != 0)
        {
            ray.distance_to_intersection = move(ray.distance_to_intersection, x1, mask);
            ray.hit_index = move(ray.hit_index, sphere.index, mask);
        }
    }

    {
        const uint16_t x2_cmp0 = eps < x2;
        const uint16_t x2_cmp1 = (x2 < ray.distance_to_intersection) & (x2 < max_distance);
        const auto mask = active_lane & (x2_cmp0 & x2_cmp1);
        if (mask != 0)
        {
            ray.distance_to_intersection = move(ray.distance_to_intersection, x2, mask);
            ray.hit_index = move(ray.hit_index, sphere.index, mask);
        }
    }

    return original_distance > ray.distance_to_intersection;
}



uint16_t checkOcclusion(
    const SimdSphere& sphere,
    SimdRay& ray,
    uint16_t active_lane = 0xffff
)
{
    const float16 relative_position_x = sphere.x - ray.org_x;
    const float16 relative_position_y = sphere.y - ray.org_y;
    const float16 relative_position_z = sphere.z - ray.org_z;

    const float16 b = -dot(
        relative_position_x,
        relative_position_y,
        relative_position_z,
        ray.dir_x,
        ray.dir_y,
        ray.dir_z);
    const float16 c = dot(
        relative_position_x,
        relative_position_y,
        relative_position_z,
        relative_position_x,
        relative_position_y,
        relative_position_z
    ) - sphere.radius * sphere.radius;

    const float16 d = b * b - c;

    const uint16_t cmp0 = d >= float16::zero();
    active_lane &= cmp0;
    if (active_lane == 0)
    {
        return 0;
    }

    const float16 sqrt_d = sqrt(d);
    const uint16_t cmp1 = b > float16::zero();

    const float16 s = select(float16::one(), -float16::one(), cmp1);
    const float16 tmp = -b + s * sqrt_d;

    const uint16_t cmp2 = ~(tmp == float16::zero());
    active_lane &= cmp2;
    if (active_lane == 0)
    {
        return 0;
    }

    const float16 x1 = c / tmp;
    const float16 x2 = tmp;

    const float16 eps(EPS);
    const float16 original_distance = ray.distance_to_intersection;

    const uint16_t x1_cmp0 = eps < x1;
    const uint16_t x2_cmp0 = eps < x2;

    return active_lane & (x1_cmp0 | x2_cmp0);
}


}