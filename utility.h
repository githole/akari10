#pragma once

#include <algorithm>
#include <optional>
#include <vector>
#include <chrono>
#include <thread>
#include <mutex>
#include <functional>
#include <array>
#include <bitset>

#include "vec3.h"

#define PI 3.141592654f
#define EPS 1e-6f
#define PHI 1.618033988749895f

namespace utility
{

constexpr float kLarge = 1e+32f;
using Color = Float3;

template<typename T>
T clampValue(const T& x, const T& a, const T& b)
{
    if (x < a)
        return a;
    if (x > b)
        return b;
    return x;
}

struct Ray
{
    Float3 org;
    Float3 dir;
    float distance_to_intersection = 1e+32;

    void reset_distance()
    {
        distance_to_intersection = 1e+32;
    }

    Float3 intersectedPoint() const
    {
        return org + dir * distance_to_intersection;
    }
};


struct HitPoint
{
    bool hit = false;
    Float3 position;
    Float3 normal;
    int index = -1;
    float distance;
};


struct Image
{
    std::vector<float> body_;
    int width_{};
    int height_{};

    Image() = default;

    Image(int w, int h) : width_(w), height_(h)
    {
        body_.resize(w * h * 3);
    }

    bool isValid() const
    {
        return !body_.empty();
    }

    size_t clampedIndex(int x, int y) const
    {
        if (x <= 0)
            x = 0;
        if (x >= width_)
            x = width_ - 1;
        if (y <= 0)
            y = 0;
        if (y >= height_)
            y = height_ - 1;
        return ((size_t)x + (size_t)y * width_) * 3;
    }

    Color load(int x, int y) const
    {
        const auto index{ clampedIndex(x, y) };
        return {
            body_[index + 0],
            body_[index + 1],
            body_[index + 2],
        };
    }

    void store(int x, int y, const Color& color)
    {
        const auto index{ clampedIndex(x, y) };
        body_[index + 0] = color[0];
        body_[index + 1] = color[1];
        body_[index + 2] = color[2];
    }

    void accum(int x, int y, const Color& color)
    {
        const auto index{ clampedIndex(x, y) };
        body_[index + 0] += color[0];
        body_[index + 1] += color[1];
        body_[index + 2] += color[2];
    }
};

void set_thread_group(uint32_t thread_id);

bool writeHDRImage(const char* filename, const Image& image);

int writePNGImage(char const* filename, int w, int h, int comp, const void* data, int stride_in_bytes);

int writeJPEGImage(char const* filename, int w, int h, int comp, const void* data, int quality);

Image loadPNGIMage(const char* filename);

Image loadHDRImage(const char* filename);

inline float easeOutQuad(float x)
{
    return 1 - (1 - x) * (1 - x);
}

inline float easeInExpo(float x)
{
    return x == 0 ? 0 : pow(2, 10 * x - 10);
}

inline float easeOutQuint(float x)
{
    return 1 - pow(1 - x, 5);
}

inline float easeOutExpo(float x)
{
    return x == 1 ? 1 : 1 - pow(2, -10 * x);
}

inline float easeInOutCubic(float x)
{
    return x < 0.5 ? 4 * x * x * x : 1 - pow(-2 * x + 2, 3) / 2;
}

template<typename T>
struct Image3DT
{
    std::vector<T> body_;
    uint32_t X_, Y_, Z_;

    void load_from_file(const char* filename)
    {
        FILE* fp{ fopen(filename, "rb") };
        if (!fp)
        {
            return;
        }

        fread(&X_, sizeof(uint32_t), 1, fp);
        fread(&Y_, sizeof(uint32_t), 1, fp);
        fread(&Z_, sizeof(uint32_t), 1, fp);
        body_.resize(X_ * Y_ * Z_);
        fread(body_.data(), sizeof(T), body_.size(), fp);
        fclose(fp);
    }

    void init(uint32_t X, uint32_t Y, uint32_t Z)
    {
        body_.resize(X * Y * Z);
        X_ = X;
        Y_ = Y;
        Z_ = Z;
    }

    void setZero()
    {
        std::fill(body_.begin(), body_.end(), T(0.0f));
    }
#if 0
    T majorant() const
    {
        T maxv(-1);

        for (auto v : body_)
        {
            maxv = std::max(maxv, v);
        }

        return maxv;
    }
#endif
    size_t clampedIndex(int x, int y, int z) const
    {
        if (x <= 0)
            x = 0;
        if (x >= X_)
            x = X_ - 1;
        if (y <= 0)
            y = 0;
        if (y >= Y_)
            y = Y_ - 1;
        if (z <= 0)
            z = 0;
        if (z >= Z_)
            z = Z_ - 1;
        return (size_t)x + (size_t)y * X_ + (size_t)z * X_ * Y_;
    }

    void store_clamped(int x, int y, int z, T v)
    {
        const auto index{ clampedIndex(x, y, z) };
        body_[index] = v;
    }

    T load_clamped(int x, int y, int z) const
    {
        const auto index{ clampedIndex(x, y, z) };
        return body_[index];
    }

    T load(int x, int y, int z) const
    {
        if (x < 0 || y < 0 || z < 0 || x >= X_ || y >= Y_ || z >= Z_)
        {
            return {};
        }
        return load_clamped(x, y, z);
    }

    void local_minmax(int x, int y, int z, T& minv, T& maxv) const
    {
        minv = kLarge;
        maxv = -kLarge;

        const float vx1 = load(x + 1, y, z);
        const float vx2 = load(x - 1, y, z);
        const float vy1 = load(x, y + 1, z);
        const float vy2 = load(x, y - 1, z);
        const float vz1 = load(x, y, z + 1);
        const float vz2 = load(x, y, z - 1);
        const float Z = load(x, y, z);

        minv = std::min({ Z, vx1, vx2, vy1, vy2, vz1, vz2 });
        maxv = std::max({ Z, vx1, vx2, vy1, vy2, vz1, vz2 });
    }

    T load_trilinear(float u, float v, float w) const
    {
        if (u < 0 || v < 0 || w < 0 ||
            u >= 1 || v >= 1 || w >= 1)
        {
            return {};
        }

        const float fu = u * X_;
        const float fv = v * Y_;
        const float fw = w * Z_;

        const int iu = (int)fu;
        const float wu = fu - iu;

        const int iv = (int)fv;
        const float wv = fv - iv;

        const int iw = (int)fw;
        const float ww = fw - iw;

        T sum = {};
        for (int i = 0; i < 8; ++i)
        {
            const int u0 = i & 1;
            const int v0 = (i & 2) >> 1;
            const int w0 = (i & 4) >> 2;

            sum +=
                (u0 ? wu : (1 - wu)) *
                (v0 ? wv : (1 - wv)) *
                (w0 ? ww : (1 - ww)) *
                load(iu + u0, iv + v0, iw + w0);
        }
        return sum;
    }
};

using Image3D = Image3DT<float>;
using Color3D = Image3DT<utility::Color>;


inline
std::optional<HitPoint>
intersectToPlane(const Ray& ray, const Float3& planeOrigin, const Float3& planeNormal)
{
    float denom = dot(planeNormal, ray.dir);

    if (std::fabs(denom) < EPS)
    {
        return std::nullopt;
    }

    Float3 p0l0 = planeOrigin - ray.org;
    float t = dot(p0l0, planeNormal) / denom;

    if (t < 0)
    {
        return std::nullopt;
    }

    // 交差点の位置を計算
    Float3 intersectionPoint = ray.org + ray.dir * t;

    // ヒットポイントを作成
    HitPoint hitPoint;
    hitPoint.hit = true;
    hitPoint.position = intersectionPoint;
    hitPoint.normal = planeNormal; 
    hitPoint.distance = t;

    return hitPoint;
}

inline
std::optional<HitPoint>
intersectToAABB(const Ray& ray, const Float3& vmin, const Float3& vmax)
{
    const Float3 bounds[2] = {
        vmin, vmax
    };
    const Float3 center = (vmax + vmin) * 0.5f;

    float tmin, tmax, tymin, tymax, tzmin, tzmax;

    Float3 invdir(1.0f / ray.dir[0], 1.0f / ray.dir[1], 1.0f / ray.dir[2]);
    int sign[3];
    sign[0] = (invdir[0] < 0);
    sign[1] = (invdir[1] < 0);
    sign[2] = (invdir[2] < 0);

    tmin = (bounds[sign[0]][0] - ray.org[0]) * invdir[0];
    tmax = (bounds[1 - sign[0]][0] - ray.org[0]) * invdir[0];
    tymin = (bounds[sign[1]][1] - ray.org[1]) * invdir[1];
    tymax = (bounds[1 - sign[1]][1] - ray.org[1]) * invdir[1];

    if ((tmin > tymax) || (tymin > tmax))
        return {};

    int axis = 0;

    if (tymin > tmin)
    {
        axis = 1;
        tmin = tymin;
    }
    if (tymax < tmax)
        tmax = tymax;

    tzmin = (bounds[sign[2]][2] - ray.org[2]) * invdir[2];
    tzmax = (bounds[1 - sign[2]][2] - ray.org[2]) * invdir[2];

    if ((tmin > tzmax) || (tzmin > tmax))
        return {};

    if (tzmin > tmin)
    {
        axis = 2;
        tmin = tzmin;
    }
    if (tzmax < tmax)
        tmax = tzmax;

    HitPoint hp;
    float distance = tmin > 0 ? tmin : tmax;
    hp.position = ray.org + distance * ray.dir;

    Float3 normal(0.0f, 0.0f, 0.0f);
    normal[axis] = 1.0f;
    if (center[axis] > hp.position[axis])
    {
        normal *= -1.0f;
    }
    hp.normal = normal;

    hp.hit = true;
    hp.index = -1;
    hp.distance = distance;

    return hp;
}

template<typename T>
T repeat(T x, T border)
{
    if (x < 0)
    {
        x = std::fmod(x, border) + border;
    }

    if (border <= x)
    {
        return std::fmod(x, border);
    }

    return x;
}

struct AABB
{
    Float3 bounds[2];
    Float3 center;
    Float3 edge;

    AABB() {}

    int N;

    Float3 worldToUV(const Float3& pos) const
    {
        return (pos - bounds[0]) / edge;
    }


    void computeIndex(const Float3& pos, int& ix, int& iy, int& iz)
    {
        const auto f = worldToUV(pos) * (float)N;

        ix = std::floor(f[0]);
        iy = std::floor(f[1]);
        iz = std::floor(f[2]);
    }


    AABB(const AABB& another)
    {
        bounds[0] = another.bounds[0];
        bounds[1] = another.bounds[1];
        center = another.center;
        edge = another.edge;
    }

    AABB& operator=(const AABB& another)
    {
        this->bounds[0] = another.bounds[0];
        this->bounds[1] = another.bounds[1];
        this->center = another.center;
        this->edge = another.edge;
        return *this;
    }

    AABB(const Float3& vmin, const Float3& vmax)
    {
        bounds[0] = vmin;
        bounds[1] = vmax;
        center = (bounds[0] + bounds[1]) * 0.5f;
        edge = vmax - vmin;
    }

    std::optional<HitPoint> intersect_(const Ray& ray) const
    {
        return intersectToAABB(ray, bounds[0], bounds[1]);
    }

    bool inside(const Float3& p) const
    {
        return
            bounds[0][0] <= p[0] &&
            bounds[0][1] <= p[1] &&
            bounds[0][2] <= p[2] &&

            p[0] <= bounds[1][0] &&
            p[1] <= bounds[1][1] &&
            p[2] <= bounds[1][2];
    }
};


#if 0
struct Hitpoint
{
    float distance{ kLarge };
    Float3 position; // world space
    Float3 normal; // world space
};

struct AABB
{
    Float3 bounds[2];
    Float3 center;

    AABB(const Float3& vmin, const Float3& vmax)
    {
        bounds[0] = vmin;
        bounds[1] = vmax;
        center = (bounds[0] + bounds[1]) * 0.5f;
    }

    AABB()
    {
        for (int i = 0; i < 3; ++i)
        {
            bounds[0][i] = kLarge;
            bounds[1][i] = -kLarge;
        }
    }

    void merge(const AABB& aabb)
    {
        for (int i = 0; i < 3; ++i)
        {
            bounds[0][i] = std::min(bounds[0][i], aabb.bounds[0][i]);
            bounds[1][i] = std::max(bounds[1][i], aabb.bounds[1][i]);
        }
    }

    std::optional<Hitpoint> intersect(const Ray& ray) const
    {
        auto tmphp = intersect_(ray);

        if (tmphp && tmphp->distance >= 0)
        {
            return tmphp;
        }

        return {};
    }

    std::optional<Hitpoint> intersect_(const Ray& ray) const
    {
        float tmin, tmax, tymin, tymax, tzmin, tzmax;


        Float3 invdir(1.0f / ray.dir[0], 1.0f / ray.dir[1], 1.0f / ray.dir[2]);
        int sign[3];
        sign[0] = (invdir[0] < 0);
        sign[1] = (invdir[1] < 0);
        sign[2] = (invdir[2] < 0);

        tmin = (bounds[sign[0]][0] - ray.org[0]) * invdir[0];
        tmax = (bounds[1 - sign[0]][0] - ray.org[0]) * invdir[0];
        tymin = (bounds[sign[1]][1] - ray.org[1]) * invdir[1];
        tymax = (bounds[1 - sign[1]][1] - ray.org[1]) * invdir[1];

        if ((tmin > tymax) || (tymin > tmax))
            return {};

        int axis = 0;

        if (tymin > tmin)
        {
            axis = 1;
            tmin = tymin;
        }
        if (tymax < tmax)
            tmax = tymax;

        tzmin = (bounds[sign[2]][2] - ray.org[2]) * invdir[2];
        tzmax = (bounds[1 - sign[2]][2] - ray.org[2]) * invdir[2];

        if ((tmin > tzmax) || (tzmin > tmax))
            return {};

        if (tzmin > tmin)
        {
            axis = 2;
            tmin = tzmin;
        }
        if (tzmax < tmax)
            tmax = tzmax;

        Hitpoint hitpoint;
        hitpoint.distance = tmin > 0 ? tmin : tmax;
        hitpoint.position = ray.org + hitpoint.distance * ray.dir;

        Float3 normal(0.0f, 0.0f, 0.0f);
        normal[axis] = 1.0f;
        if (center[axis] > hitpoint.position[axis])
        {
            normal *= -1.0f;
        }
        hitpoint.normal = normal;

        return hitpoint;
    }
};
#endif


namespace random {

    inline uint32_t rotr(uint32_t x, int shift) {
        return (x >> shift) | (x << (32 - shift));
    }

    inline uint64_t rotr(uint64_t x, int shift) {
        return (x >> shift) | (x << (64 - shift));
    }

    struct splitmix64 {
        uint64_t x;

        splitmix64(uint64_t a = 0) : x(a) {}

        uint64_t next() {
            uint64_t z = (x += 0x9e3779b97f4a7c15);
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
            z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
            return z ^ (z >> 31);
        }
    };

    // PCG(64/32)
    // http://www.pcg-random.org/download.html
    // initial_inc from official library
    struct PCG_64_32 {
        uint64_t state;
        uint64_t inc;

        PCG_64_32(uint64_t initial_state = 0x853c49e6748fea9bULL,
            uint64_t initial_inc = 0xda3e39cb94b95bdbULL)
            : state(initial_state), inc(initial_inc) {}

        void set_seed(uint64_t seed) {
            splitmix64 s(seed);
            state = s.next();
        }

        using return_type = uint32_t;
        return_type next() {
            auto oldstate = state;
            state = oldstate * 6364136223846793005ULL + (inc | 1);
            uint32_t xorshifted = uint32_t(((oldstate >> 18u) ^ oldstate) >> 27u);
            uint32_t rot = oldstate >> 59u;

            return rotr(xorshifted, rot);
        }

        // [0, 1)
        float next01() {
            return (float)(((double)next()) /
                ((double)std::numeric_limits<uint32_t>::max() + 1));
        }

        float next(float minV, float maxV)
        {
            return next01() * (maxV - minV) + minV;
        }
    };

} // namespace random

inline Float3 sample_uniform_sphere_surface(float u, float v) {
    const float tz = u * 2 - 1;
    const float phi = v * PI * 2;
    const float k = sqrt(1.0 - tz * tz);
    const float tx = k * cos(phi);
    const float ty = k * sin(phi);
    return Float3(tx, ty, tz);
}

template <typename Vec3>
inline void createOrthoNormalBasis(const Vec3& normal, Vec3* tangent, Vec3* binormal) {
    if (abs(normal[0]) > abs(normal[1]))
    {
        (*tangent) = cross(Vec3(0, 1, 0), normal);
        (*tangent) = normalize(*tangent);
    }
    else
    {
        (*tangent) = cross(Vec3(1, 0, 0), normal);
        (*tangent) = normalize(*tangent);
    }
    (*binormal) = cross(normal, *tangent);
    (*binormal) = normalize(*binormal);
}

inline float remap(float x, float a, float b)
{
    return x * (b - a) + a;
}

struct Timer
{
    const char* label;
    std::chrono::steady_clock::time_point start_time;

    Timer(const char* l) : label(l) 
    {
        start_time = std::chrono::high_resolution_clock::now();
    }

    ~Timer()
    {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        std::cout << "* " << label << ": " << (duration / 1000.0f) << " [sec]" << std::endl;

    }
};


class TaskExecutor
{
    std::thread thread_;
    std::atomic<bool> end_;
    std::mutex mutex_;

    std::vector <std::function<void(void)>> tasks_;

public:
    void appendTask(const std::function<void(void)>& task)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        tasks_.push_back(task);
    }

    void worker()
    {
        while (true)
        {
            std::function<void(void)> next_task;

            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!tasks_.empty())
                {
                    next_task = *(tasks_.end() - 1);
                    tasks_.pop_back();
                }
            }

            if (next_task)
            {
                next_task();
                continue;
            }

            // emptyかつendなら終了
            if (end_.load())
            {
                return;
            }

            // emptyだったらちょい待つ
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    TaskExecutor()
    {
        thread_ = std::thread(&TaskExecutor::worker, this);
        end_.store(false);
    }

    ~TaskExecutor()
    {
        end_.store(true);
        thread_.join();
    }
};

inline bool solveQuadratic(float b, float c, float& x1, float& x2)
{
    const float d = b * b - c;
    if (d < 0)
    {
        return false;
    }
    float sqrt_d = std::sqrt(d);
    float tmp = (b > 0.0f) ? (-b - sqrt_d) : (-b + sqrt_d);
    if (tmp == 0.0f)
    {
        return false;
    }
    x1 = c / tmp;
    x2 = tmp;
    return true;
}

struct Sphere
{
    float radius;
    Float3 position;
    int metaID = 0;

    Sphere() {}

    Sphere(const Sphere& s)
    {
        radius = s.radius;
        position = s.position;
        metaID = s.metaID;
    }

    Sphere(float radius, const Float3& position)
        : radius(radius), position(position)
    {
    }

    bool checkIntersectionAndUpdateRay(Ray& ray) const
    {
        const Float3 relative_position = position - ray.org;

        const float b = dot(relative_position, ray.dir);
        const float c = dot(relative_position, relative_position) - radius * radius;
        float x1, x2;

        if (solveQuadratic(-b, c, x1, x2))
        {
            bool updated = false;
            if (EPS < x1 && x1 < ray.distance_to_intersection)
            {
                ray.distance_to_intersection = x1;
                updated = true;
            }
            if (EPS < x2 && x2 < ray.distance_to_intersection)
            {
                ray.distance_to_intersection = x2;
                updated = true;
            }
            return updated;
        }
        return false;
    }

    Float3 computeNormal(const Float3& surface_position) const
    {
        return normalize(surface_position - position);
    }
};

inline
void printBitPattern(uint16_t value)
{
    std::bitset<16> bits(value);
    std::cout << bits << std::endl;
}

template<typename T>
void printArray(const T& arr)
{
    for (auto& v : arr)
    {
        std::cout << v << ", ";
    }
    std::cout << std::endl;
}

// 原点(0, 0, 0)を仮定
template<typename CallBack>
void dda(
    float x0, float y0, float z0,
    float dx, float dy, float dz,
    float cell_size_x, float cell_size_y, float cell_size_z,
    int N,
    CallBack&& callback
)
{
    const float l = sqrt(dx * dx + dy * dy + dz * dz);
    dx /= l;
    dy /= l;
    dz /= l;

    const float small_dx = (cell_size_x * 0.0001) * dx;
    const float small_dy = (cell_size_y * 0.0001) * dy;
    const float small_dz = (cell_size_z * 0.0001) * dz;

    x0 += small_dx;
    y0 += small_dy;
    z0 += small_dz;

    const int add_ix = dx >= 0 ? 1 : 0;
    const int add_iy = dy >= 0 ? 1 : 0;
    const int add_iz = dz >= 0 ? 1 : 0;

    for (int I = 0; I < 128; ++I)
    {
        if (x0 == 0 || y0 == 0 || z0 == 0)
        {
            break;
        }

        const int ix = std::floor(x0 / cell_size_x);
        const int iy = std::floor(y0 / cell_size_y);
        const int iz = std::floor(z0 / cell_size_z);

        // printf("[%d, %d, %d]", ix, iy, iz);

        if (ix >= N || iy >= N || iz >= N || ix < 0 || iy < 0 || iz < 0)
        {
            break;
        }

        if (callback(ix, iy, iz))
        {
            return;
        }

        const float left_x = -x0 + (ix + add_ix) * cell_size_x;
        const float tx = left_x / dx;
        const float left_y = -y0 + (iy + add_iy) * cell_size_y;
        const float ty = left_y / dy;
        const float left_z = -z0 + (iz + add_iz) * cell_size_z;
        const float tz = left_z / dz;

        if (tx < ty && tx < tz)
        {
            x0 = (ix + add_ix) * cell_size_x;
            x0 += small_dx;
            y0 += tx * dy;
            z0 += tx * dz;
        }
        else if (ty < tx && ty < tz)
        {
            x0 += ty * dx;
            y0 = (iy + add_iy) * cell_size_y;
            y0 += small_dy;
            z0 += ty * dz;
        }
        else
        {
            x0 += tz * dx;
            y0 += tz * dy;
            z0 = (iz + add_iz) * cell_size_z;
            z0 += small_dz;
        }
    }
}

inline
bool isAABBOverlappingSphere(const Float3& vmin, const Float3& vmax, const Sphere& sphere)
{
    float closestX = std::max(vmin[0], std::min(sphere.position[0], vmax[0]));
    float closestY = std::max(vmin[1], std::min(sphere.position[1], vmax[1]));
    float closestZ = std::max(vmin[2], std::min(sphere.position[2], vmax[2]));

    float distanceSquared = 
        (closestX - sphere.position[0]) * (closestX - sphere.position[0]) +
        (closestY - sphere.position[1]) * (closestY - sphere.position[1]) +
        (closestZ - sphere.position[2]) * (closestZ - sphere.position[2]);

    // printf("[len: %f]", sqrt(distanceSquared))

    return distanceSquared <= (sphere.radius * sphere.radius);
}

inline Float3 sampleCosWeightedHemisphere(float randomValue0, float randomValue1)
{
    const float r0 = 2 * PI *randomValue0;
    const float r1 = std::sqrt(randomValue1);
    const Float3 tmp(std::cos(r0) * r1, std::sin(r0) * r1, std::sqrt(1 - r1));
    return normalize(tmp);
}

inline
float frac(float x)
{
    return x - (int)x;
}


struct ThreadDispacher
{
    std::vector<std::function<void(void)>> tasks;
    std::atomic<int> currentTaskIndex = 0;

    std::atomic<bool> end;

    std::vector<std::thread> threads;

    void worker(int thread_id)
    {
        set_thread_group(thread_id);

        while (!end.load())
        {
            const int taskIndex = currentTaskIndex.fetch_add(1);
            if (taskIndex >= tasks.size())
            {
                return;
            }
            tasks[taskIndex]();
        }
    }

    void append(const std::function<void(void)>& f)
    {
        tasks.push_back(f);
    }

    void start(int N)
    {
        end.store(false);
        for (int i = 0; i < N; ++i)
        {
            threads.push_back(std::thread(&ThreadDispacher::worker, this, i));
        }
    }

    void wait()
    {
        for (auto& t : threads)
        {
            t.join();
        }
    }

    ~ThreadDispacher()
    {
    }
};

}