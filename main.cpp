#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <random>
#include <algorithm>

#include "utility.h"
#include "renderer.h"
#include "simd.h"
#include "perlin.h"
#include "pbd.h"

using Float3 = utility::Float3;

using Random = utility::random::PCG_64_32;

struct BaseParameter
{
    int width{ 1280 };
    int height{ 720 };
    int super_sample_count{ 5 };

    float movie_time_sec{ 10.0f };
    int max_frame{ 300 }; 
} g_param;

void update(float dt)
{

}

std::vector<uint8_t> tonemap(float screen_scale, const utility::Image& image)
{
    const int comp = 3;
    std::vector<uint8_t> ldr_image(image.width_ * image.height_ * comp);

#pragma omp parallel for schedule(dynamic, 1)
    for (int iy = 0; iy < image.height_; ++iy)
    {
        for (int ix = 0; ix < image.width_; ++ix)
        {
            const auto col = image.load(ix, iy);

            const uint8_t r{ (uint8_t)utility::clampValue(screen_scale * pow(col[0], 1 / 2.4f) * 255, 0.0f, 255.0f) };
            const uint8_t g{ (uint8_t)utility::clampValue(screen_scale * pow(col[1], 1 / 2.4f) * 255, 0.0f, 255.0f) };
            const uint8_t b{ (uint8_t)utility::clampValue(screen_scale * pow(col[2], 1 / 2.4f) * 255, 0.0f, 255.0f) };

            const auto idx{ (ix + iy * image.width_) * comp };

            ldr_image[idx + 0] = r;
            ldr_image[idx + 1] = g;
            ldr_image[idx + 2] = b;
        }
    }

    return ldr_image;
}

bool g_debug = false;

constexpr int GridSize = 16;



struct NoiseVec3
{
    std::vector<Float3> noise_field_;
    int N_;

    void setup(int seed, int N)
    {
        N_ = N;
        perlin::PerlinNoise x(seed);
        perlin::PerlinNoise y(seed * 63 + 17);
        perlin::PerlinNoise z(seed * 65536 + 9);

        noise_field_.resize(N * N * N);

        const float s = 0.125f * 0.25f; // è¨Ç≥Ç¢Ç∆í·é¸îg

#pragma omp parallel for schedule(dynamic, 1)
        for (int iz = 0; iz < N; ++iz)
        {
            for (int iy = 0; iy < N; ++iy)
            {
                for (int ix = 0; ix < N; ++ix)
                {
                    const float fx = (ix + 0.5f) / N * s;
                    const float fy = (iy + 0.5f) / N * s;
                    const float fz = (iz + 0.5f) / N + s;

                    const float nx = x.noise(fx, fy, fz);
                    const float ny = y.noise(fx, fy, fz);
                    const float nz = z.noise(fx, fy, fz);

                    noise_field_[ix + iy * N + iz * N * N] = 2.0f * Float3(nx, ny, nz) - Float3(1.0f);
                }
            }
        }
    }

    float get(int comp, int ix, int iy, int iz) const
    {
        if (ix < 0)
        {
            ix += N_;
        }
        else if (N_ <= ix)
        {
            ix -= N_;
        }

        if (iy < 0)
        {
            iy += N_;
        }
        else if (N_ <= iy)
        {
            iy -= N_;
        }

        if (iz < 0)
        {
            iz += N_;
        }
        else if (N_ <= iz)
        {
            iz -= N_;
        }

        return noise_field_[ix + iy * N_ + iz * N_ * N_][comp];
    }

    float deriv(int comp, int axis, int ix, int iy, int iz) const
    {
        int offset[3] = {};
        offset[axis] = 1;
        const float v0 = get(comp, ix + offset[0], iy + offset[1], iz + offset[2]);
        const float v1 = get(comp, ix - offset[0], iy - offset[1], iz - offset[2]);
        const float h = 1.0f / N_;
        return (v0 - v1) / (2 * h);
    }

    void dump(const char* n)
    {

        utility::Image image(N_, N_);

        for (int iy = 0; iy < image.width_; ++iy)
        {
            for (int ix = 0; ix < image.width_; ++ix)
            {
                image.store(ix, iy, noise_field_[ix + iy * N_]);
            }
        }


        auto ldr_image = tonemap(1, image);

        char buf[256];
        snprintf(buf, sizeof(buf), n);;
        utility::writeJPEGImage(buf, N_, N_, 3, ldr_image.data(), 100);

    }
};




NoiseVec3 g_3d_noise;
std::vector<utility::Sphere> g_spheres, g_spheres2;

std::vector<utility::Sphere> g_spheres_cut2;

struct VelField
{
    std::vector<Float3> vel_field;

    template<typename AABB>
    Float3 get_vel(const AABB& aabb, const Float3& pos) const
    {
        Float3 uv = aabb.worldToUV(pos);
        const int N = g_3d_noise.N_;
        uv *= N;

        int ix = uv[0];
        int iy = uv[1];
        int iz = uv[2];

        ix = utility::repeat(ix, N);
        iy = utility::repeat(iy, N);
        iz = utility::repeat(iz, N);

        return vel_field[ix + iy * N + iz * N * N];
    }

    void setupVelField(float scale)
    {
        utility::Timer _("setupVelField");

        const int N = g_3d_noise.N_;
        vel_field.resize(N * N * N);

#pragma omp parallel for schedule(dynamic, 1)
        for (int iz = 0; iz < N; ++iz)
        {
            for (int iy = 0; iy < N; ++iy)
            {
                for (int ix = 0; ix < N; ++ix)
                {
                    const float x = g_3d_noise.deriv(2, 1, ix, iy, iz) - g_3d_noise.deriv(1, 2, ix, iy, iz);
                    const float y = g_3d_noise.deriv(0, 2, ix, iy, iz) - g_3d_noise.deriv(2, 0, ix, iy, iz);
                    const float z = g_3d_noise.deriv(1, 0, ix, iy, iz) - g_3d_noise.deriv(0, 1, ix, iy, iz);

                    vel_field[ix + iy * N + iz * N * N] = scale * Float3(x, y, z);
                }
            }
        }
    }
};

VelField g_velfield;

struct PBDWorld
{
    std::vector<pbd::Particle> ps;
    std::vector<pbd::DistanceConstraint> cs;
    std::vector<pbd::CollisionPair> collisions;

    void simulate(float dt)
    {
        utility::Timer _("  simulate");
        pbd::simulate(ps, cs, collisions, dt);
    }
};

PBDWorld g_pbd_world;

void setup_global(const utility::Image& moji, int N)
{
    utility::Timer _("setup_global");

    g_3d_noise.setup(42, 256);


    float tm = 0;
    {
        volatile utility::Timer _("  initialize");
        // g_spheres.resize(N);

#if 0
#pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < N; ++i)
        {
            Random r(i);
            r.next();

            float x = r.next(-0.2, 0.2);
            float y = r.next(-0.2, 0.2);
            float z = r.next(-0.2, 0.2);

            float t = tm * 5 + r.next(-10, 10);

            float px = r.next(-2, 2);
            float py = r.next(-2, 2);
            float pz = r.next(-2, 2);

            float ox = r.next(-5, 5);
            float oy = r.next(-5, 5);
            float oz = r.next(-5, 5);

            x = x + 0.01f * cos(t * px + ox);
            y = y + 0.015f * sin(t * py + oy);
            z = z + 0.02f * cos(t * pz + oz);

            float o = 0.005f * sin(t * 5.0 + r.next01());

            g_spheres[i] = utility::Sphere(0.01f + o, Float3(x, y, z));
            //spheres[i] = utility::Sphere(0.02, Float3(x, y, z));
        }
#endif

        int S = powf(N, 1.0f / 3.0f);

        g_spheres.resize(S * S * S + N);

#pragma omp parallel for schedule(dynamic, 1)
        for (int iz = 0; iz < S; ++iz)
        {
            for (int iy = 0; iy < S; ++iy)
            {
                for (int ix = 0; ix < S; ++ix)
                {
                    int i = ix + iy * S + iz * S * S;

                    Random r(i);
                    r.next();


                    float x = (ix + 0.5f) / S * 0.4f - 0.2f;
                    float y = (iy + 0.5f) / S * 0.4f - 0.2f;
                    float z = (iz + 0.5f) / S * 0.4f - 0.2f;


                    /*
                    float x = r.next(-0.2, 0.2);
                    float y = r.next(-0.2, 0.2);
                    float z = r.next(-0.2, 0.2);
                    */

                    float t = tm * 5 + r.next(-10, 10);

                    float px = r.next(-2, 2);
                    float py = r.next(-2, 2);
                    float pz = r.next(-2, 2);

                    float ox = r.next(-5, 5);
                    float oy = r.next(-5, 5);
                    float oz = r.next(-5, 5);

                    /*
                    x = x + 0.01f * cos(t * px + ox);
                    y = y + 0.015f * sin(t * py + oy);
                    z = z + 0.02f * cos(t * pz + oz);
                    */

                    // float o = 0.005f * sin(t * 5.0 + r.next01());


                    float radius = 0.01f;

                    g_spheres[i] = utility::Sphere(radius, Float3(x, y, z));

                    if (ix == 0 || ix == S - 1 ||
                        iy == 0 || iy == S - 1 ||
                        iz == 0 || iz == S - 1)
                    {
                        g_spheres[i].metaID = 1;
                    }

                    int mx = (int)((ix + 0.5f) / S * 64) % 64;
                    int my = (int)((1 - (iy + 0.5f) / S) * 64) % 64;

                    const float v = moji.load(mx, my)[0];
                    if (v <= 254 && iz == S - 1)
                    {

                        g_spheres[i].metaID = 3;

                        g_spheres[i].radius *= (1 - (v / 254.0f));
                    }


                    /*
                    if (length(Float3(x, y, z)) <= 0.1f)
                    {
                        g_spheres[i].metaID = 1;
                    }
                    */
                }
            }
        }

#pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < N; ++i)
        {
            Random r(i);
            r.next();
            
            float x = r.next(-0.175, 0.175);
            float y = r.next(-0.175, 0.175);
            float z = r.next(-0.175, 0.175);

            float t = tm * 5 + r.next(-10, 10);

            float px = r.next(-2, 2);
            float py = r.next(-2, 2);
            float pz = r.next(-2, 2);

            float ox = r.next(-5, 5);
            float oy = r.next(-5, 5);
            float oz = r.next(-5, 5);

            x = x + 0.01f * cos(t * px + ox);
            y = y + 0.015f * sin(t * py + oy);
            z = z + 0.02f * cos(t * pz + oz);

            float o = 0.005f * sin(t * 5.0 + r.next01());

            float radius = 0.005f + o;

            g_spheres[(S * S * S) + i] = utility::Sphere(radius, Float3(x, y, z));
            // g_spheres[(S * S * S) + i].metaID = 2;
        }

        std::minstd_rand engine(0);
        std::shuffle(g_spheres.begin(), g_spheres.end(), engine);
        g_spheres2 = g_spheres;
    }


    g_velfield.setupVelField(0.001f);


    // Cut2
    {
        volatile utility::Timer _("  initialize 2");

        /*
        const int N2 = 512;

        int S2 = powf(N2, 1.0f / 3.0f);
        */
//        g_spheres_cut2.resize(S2 * S2 * S2);


        for (int C = 0; C < 3; ++C)
        {
            for (int iy = 0; iy < moji.height_; ++iy)
            {
                for (int ix = 0; ix < moji.width_; ++ix)
                {
                    const auto v = moji.load(ix, iy);

                    if (v[0] < 255.0f)
                    {
                        const float fx = (ix + 0.5f) / moji.width_;
                        const float fy = 1 - (iy + 0.5f) / moji.height_;

                        const float x = fx * 0.6 - 0.3f;
                        const float y = fy * 0.6 - 0.3f + 0.3f + (C * 2.0f);

                        for (int iz = 0; iz < 4; ++iz)
                        {
                            const float fz = (iz + 0.5f) / 16;
                            const float z = fz * 0.2 - 0.1f;

                            float radius = (1.0f - v[0] / 255.0f) * 0.01f;
                            auto s = utility::Sphere(radius, Float3(x, y, z));
                            s.metaID = C;
                            g_spheres_cut2.push_back(s);
                        }
                    }
                }
            }
        }



/*
#pragma omp parallel for schedule(dynamic, 1)
        for (int iz = 0; iz < S2; ++iz)
        {
            for (int iy = 0; iy < S2; ++iy)
            {
                for (int ix = 0; ix < S2; ++ix)
                {
                    int i = ix + iy * S2 + iz * S2 * S2;

                    Random r(i);
                    r.next();


                    float x = (ix + 0.5f) / S2 * 0.4f - 0.2f;
                    float y = (iy + 0.5f) / S2 * 0.4f - 0.2f;
                    float z = (iz + 0.5f) / S2 * 0.4f - 0.2f;


                    float radius = 0.03f;

                    g_spheres_cut2 [i] = utility::Sphere(radius, Float3(x, y, z));
                }
            }
        }
*/

        printf("Spheres: %d\n", g_spheres_cut2.size());


        // çSë©ÇçÏÇÈ

        for (int i = 0; i < g_spheres_cut2.size(); ++i)
        {
            pbd::Particle p;
            p.vel = 0;
            p.pos = g_spheres_cut2[i].position;
            p.mass = 1.0f;
            p.sphereIndex = i;
            p.radius = g_spheres_cut2[i].radius;
            p.metaID = g_spheres_cut2[i].metaID;
            g_pbd_world.ps.push_back(p);
        }

        for (int i = 0; i < g_spheres_cut2.size(); ++i)
        {
            for (int j = i + 1; j < g_spheres_cut2.size(); ++j)
            {
                auto& s0 = g_spheres_cut2[i];
                auto& s1 = g_spheres_cut2[j];

                if (s0.metaID == s1.metaID)
                {

                    const float l = length(s0.position - s1.position);
                    if (l <= s0.radius * 4)
                    {
                        pbd::DistanceConstraint c;
                        c.p1 = i;
                        c.p2 = j;
                        c.restLength = l;
                        g_pbd_world.cs.push_back(c);
                    }
                }
            }
        }

        for (auto& c : g_pbd_world.cs)
        {
            g_pbd_world.ps[c.p1].cs.push_back(&c);
            g_pbd_world.ps[c.p2].cs.push_back(&c);
        }

        printf("Constraints: %d\n", g_pbd_world.cs.size());


#if 0
        g_spheres_cut2.resize(N);

#pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < N; ++i)
        {
            Random r(i);
            r.next();

            float x = r.next(-0.2, 0.2);
            float y = r.next(-0.2, 0.2);
            float z = r.next(-0.2, 0.2);

            float t = tm * 5 + r.next(-10, 10);

            float px = r.next(-2, 2);
            float py = r.next(-2, 2);
            float pz = r.next(-2, 2);

            float ox = r.next(-5, 5);
            float oy = r.next(-5, 5);
            float oz = r.next(-5, 5);

            x = x + 0.01f * cos(t * px + ox);
            y = y + 0.015f * sin(t * py + oy);
            z = z + 0.02f * cos(t * pz + oz);

            float o = 0.005f * sin(t * 5.0 + r.next01());

            g_spheres_cut2[i] = utility::Sphere(0.01f + o, Float3(x, y, z));
        }
#endif
    }
}

void final_setup(const utility::Image& moji)
{

    // Cut2
    {
        volatile utility::Timer _("  initialize 3");

        int lastSize = g_spheres_cut2.size();

        for (int C = 0; C < 3; ++C)
        {
            for (int iy = 0; iy < moji.height_; ++iy)
            {
                for (int ix = 0; ix < moji.width_; ++ix)
                {
                    const auto v = moji.load(ix, iy);

                    if (v[0] < 255.0f)
                    {
                        const float fx = (ix + 0.5f) / moji.width_;
                        const float fy = 1 - (iy + 0.5f) / moji.height_;

                        const float x = fx * 0.6 - 0.3f;
                        const float y = fy * 0.6 - 0.3f + 0.3f + 0.5f + C * 1.0f;

                        for (int iz = 0; iz < 4; ++iz)
                        {
                            const float fz = (iz + 0.5f) / 16;
                            const float z = fz * 0.2 - 0.1f;

                            float radius = (1.0f - v[0] / 255.0f) * 0.01f;
                            auto s = utility::Sphere(radius, Float3(x, y, z));
                            s.metaID = C + 3;
                            g_spheres_cut2.push_back(s);
                        }
                    }
                }
            }
        }


        printf("Spheres: %d\n", g_spheres_cut2.size());


        // çSë©ÇçÏÇÈ

        for (int i = lastSize; i < g_spheres_cut2.size(); ++i)
        {
            pbd::Particle p;
            p.vel = 0;
            p.pos = g_spheres_cut2[i].position;
            p.mass = 1.0f;
            p.sphereIndex = i;
            p.radius = g_spheres_cut2[i].radius;
            p.metaID = g_spheres_cut2[i].metaID;
            p.vel = Float3(0, -1.0f, 0);
            g_pbd_world.ps.push_back(p);
        }

        Random rng;

        for (int i = lastSize; i < g_spheres_cut2.size(); ++i)
        {
            for (int j = i + 1; j < g_spheres_cut2.size(); ++j)
            {
                auto& s0 = g_spheres_cut2[i];
                auto& s1 = g_spheres_cut2[j];

                if (s0.metaID == s1.metaID)
                {
                    if (s0.metaID == 3)
                    {
                        continue;
                    }

                    const float l = length(s0.position - s1.position);
                    if (l <= s0.radius * 3)
                    {
                        pbd::DistanceConstraint c;
                        c.p1 = i;
                        c.p2 = j;
                        c.restLength = l;
                        g_pbd_world.cs.push_back(c);
                    }
                }
            }
        }

        for (auto& c : g_pbd_world.cs)
        {
            g_pbd_world.ps[c.p1].cs.push_back(&c);
            g_pbd_world.ps[c.p2].cs.push_back(&c);
        }

        printf("Constraints: %d\n", g_pbd_world.cs.size());
    }
}



struct Scene
{
    //std::vector<utility::Sphere> spheres;

    std::vector<simd::SimdSphere> simd_spheres;

    utility::AABB aabb;

    auto compute_cell_aabb(int cx, int cy, int cz) const
    {
        const Float3 vmin = aabb.bounds[0];
        const Float3 cell_edge = aabb.edge / Float3(GridSize);
        const Float3 cell_vmin = vmin + cell_edge * Float3(cx, cy, cz);
        const Float3 cell_vmax = cell_vmin + cell_edge;
        return std::make_pair(cell_vmin, cell_vmax);
    }

    using GridType = std::vector<std::vector<utility::Sphere*>>;
    using IndexGridType = std::vector<std::vector<int>>;

    GridType grid;
    IndexGridType index_grid;
    std::vector<std::vector<simd::SimdSphere>> simd_grid;

    Scene()
    {
        grid.resize(GridSize * GridSize * GridSize);
        index_grid.resize(GridSize * GridSize * GridSize);
        simd_grid.resize(GridSize * GridSize * GridSize);
    }


    std::vector<Float3> color_table;
    std::vector<Float3> color_table2;

    int getIndex(int ix, int iy, int iz) const
    {
        int index = ix + iy * GridSize + iz * GridSize * GridSize;

        if (ix < 0 || iy < 0 || iz < 0 ||
            GridSize <= ix || GridSize <= iy || GridSize <= iz)
        {
            return -1;
        }

        return index;
    }


    void setup_cut2(std::vector<utility::Sphere>& spheres, int N, float tm, float dt)
    {
        utility::Timer _("setup");
        aabb = utility::AABB(
            Float3(-1.25, -1, -1.25),
            Float3(1.25, 1, 1.25)
        );
        aabb.N = GridSize;

        color_table.push_back(Float3(0.85f, 0.2f, 0.2f));

        color_table2.push_back(Float3(0.0f));
        color_table2.push_back(Float3(0.0f));
        color_table2.push_back(Float3(0.7f));
        color_table2.push_back(Float3(0.95f));
        color_table2.push_back(Float3(0.0f));
        color_table2.push_back(Float3(0.0f));
        color_table2.push_back(Float3(0.0f));



        g_pbd_world.simulate(dt * 0.3f);

        // feedback
        {
            volatile utility::Timer _("  feedback");
#pragma omp parallel for schedule(dynamic, 1)
            for (int i = 0; i < g_pbd_world.ps.size(); ++i)
            {
                auto& sphere = spheres[i];
                auto& particle = g_pbd_world.ps[i];
                sphere.position = particle.pos;

                if (sphere.position[1] < -0.5f)
                {
                    printf("*");
                }
            }
        }



        setup_postprocess(1, spheres);
    }

    void setup_postprocess(int phase, std::vector<utility::Sphere>& spheres)
    {
        // uniform grid
        {
            volatile utility::Timer _("  uniform grid");
            const Float3 vmin = aabb.bounds[0];
            const Float3 cell_edge = aabb.edge / Float3(GridSize);

            struct Item
            {
                int gridIndex;
                int sphereIndex;

                Item(int a, int b) : gridIndex(a), sphereIndex(b) {}
            };
            int max_threads = omp_get_max_threads();


            for (int i = 0; i < spheres.size(); ++i)
            {
                // int thread_id = omp_get_thread_num();

                int ix, iy, iz;
                auto& sphere = spheres[i];
                aabb.computeIndex(sphere.position, ix, iy, iz);

                const int gridIndex = getIndex(ix, iy, iz);
                if (gridIndex >= 0)
                {
                    // tmpBuf[thread_id].emplace_back(gridIndex, i);

                    grid[gridIndex].push_back(&sphere);
                    index_grid[gridIndex].push_back(i);
                }

                for (int ox = -1; ox <= 1; ++ox)
                {
                    for (int oy = -1; oy <= 1; ++oy)
                    {
                        for (int oz = -1; oz <= 1; ++oz)
                        {
                            int cx = ix + ox;
                            int cy = iy + oy;
                            int cz = iz + oz;

                            if ((ox == 0 && oy == 0 && oz == 0) ||
                                cx < 0 || GridSize <= cx ||
                                cy < 0 || GridSize <= cy ||
                                cz < 0 || GridSize <= cz)
                            {
                                continue;
                            }

                            const auto [cell_vmin, cell_vmax] = compute_cell_aabb(cx, cy, cz);


                            if (utility::isAABBOverlappingSphere(cell_vmin, cell_vmax, sphere))
                            {
                                const int gridIndex2 = getIndex(cx, cy, cz);
                                if (gridIndex2 >= 0)
                                {
                                    grid[gridIndex2].push_back(&sphere);
                                    index_grid[gridIndex2].push_back(i);
                                    // printf("{%d,%d,%d, (%d, %d, %d)}", ix, iy, iz, cx, cy, cz);
                                }
                            }
                        }
                    }
                }
            }
        }

#if 0
        if (phase == 1)
        {
            utility::Timer _("  make collision pair");
            g_pbd_world.collisions.clear();

            for (int i = 0; i < spheres.size(); ++i)
            {
                int ix, iy, iz;
                auto& sphere = spheres[i];
                aabb.computeIndex(sphere.position, ix, iy, iz);
                const int gridIndex = getIndex(ix, iy, iz);

                if (gridIndex < 0)
                {
                    continue;
                }

                for (int j = 0; j < grid[gridIndex].size(); ++j)
                {
                    auto& p = *grid[gridIndex][j];

                    if (sphere.metaID != p.metaID)
                    {
                        const Float3 delta = p.position - sphere.position;
                        const auto l = length(delta);
                        if (l < sphere.radius + p.radius)
                        {
                            pbd::CollisionPair cp;
                            cp.p1 = i;
                            cp.p2 = index_grid[gridIndex][j];
                            cp.r1 = sphere.radius;
                            cp.r2 = p.radius;

                            cp.delta = delta;
                            cp.dist = l;

                            g_pbd_world.collisions.push_back(cp);
                        }
                    }
                }
            }
        }
#endif

        // simdify
        {
            utility::Timer _("  simdify");
#pragma omp parallel for schedule(dynamic, 1)
            for (int cell_index = 0; cell_index < GridSize * GridSize * GridSize; ++cell_index)
            {
                auto& cell = grid[cell_index];
                auto& index_cell = index_grid[cell_index];

                for (int i = 0; i < cell.size(); i += 16)
                {
                    std::array<float, 16> radius;
                    std::array<float, 16> x;
                    std::array<float, 16> y;
                    std::array<float, 16> z;
                    std::array<float, 16> si;

                    uint16_t active = 0;

                    for (int j = 0; j < 16; ++j)
                    {
                        if (i + j >= cell.size())
                        {
                            break;
                        }

                        active |= (1 << j);

                        auto& s = *cell[i + j];
                        radius[j] = s.radius;
                        x[j] = s.position[0];
                        y[j] = s.position[1];
                        z[j] = s.position[2];
                        si[j] = index_cell[i + j];
                    }

                    simd::SimdSphere s{
                        simd::float16(radius),
                            simd::float16(x),
                            simd::float16(y),
                            simd::float16(z),
                            simd::float16(si),
                            active
                    };

                    simd_grid[cell_index].push_back(s);
                }
            }
        }
#if 0
        for (int i = 0; i < spheres.size(); i += 16)
        {
            std::array<float, 16> radius;
            std::array<float, 16> x;
            std::array<float, 16> y;
            std::array<float, 16> z;
            std::array<float, 16> si;

            uint16_t active = 0;

            for (int j = 0; j < 16; ++j)
            {
                if (i + j >= spheres.size())
                {
                    break;
                }

                active |= (1 << j);

                auto& s = spheres[i + j];
                radius[j] = s.radius;
                x[j] = s.position[0];
                y[j] = s.position[1];
                z[j] = s.position[2];
                si[j] = i + j;
            }

            simd::SimdSphere s{
                simd::float16(radius),
                    simd::float16(x),
                    simd::float16(y),
                    simd::float16(z),
                    simd::float16(si),
                    active
            };

            simd_spheres.push_back(s);
        }
#endif
    }

    void setup(std::vector<utility::Sphere>& spheres, int N, float tm, float dt, float update_count)
    {
        utility::Timer _("setup");

        aabb = utility::AABB(
            Float3(-1, -1, -1),
            Float3(1, 1, 1)
        );
        aabb.N = GridSize;

        for (int i = 0; i < 10; ++i)
        {
            color_table.push_back(Float3(0.9f));
            color_table.push_back(Float3(0.9f));
            color_table.push_back(Float3(0.9f));
            color_table.push_back(Float3(0.9f));
            color_table.push_back(Float3(0.9f));
            color_table.push_back(Float3(0.9f));
            color_table.push_back(Float3(0.9f));
            color_table.push_back(Float3(0.9f));
            //color_table.push_back(Float3(0.9f, 0.3f, 0.2f));
            //color_table.push_back(Float3(0.9f, 0.3f, 0.2f));
            //color_table.push_back(Float3(0.9f, 0.3f, 0.2f));
            color_table.push_back(Float3(0.9f, 0.3f, 0.2f));
            // color_table.push_back(Float3(-0.5f, -0.9f, -0.5f));
        }

        color_table2.push_back(Float3(0.0f));
        color_table2.push_back(Float3(0.0f));
        color_table2.push_back(Float3(0.7f));
        color_table2.push_back(Float3(-10.0f, -10.0f, -10.0f));
        color_table2.push_back(Float3(0.0f));
        color_table2.push_back(Float3(0.0f));
        color_table2.push_back(Float3(0.0f));

        {
            utility::Timer _("update spheres");

#pragma omp parallel for schedule(dynamic, 1)
            for (int i = 0; i < spheres.size(); ++i)
            {
                auto& sphere = spheres[i];
                const auto vel = g_velfield.get_vel(aabb, sphere.position);
                float vels =    pow(sphere.radius / 0.01f, -0.5f);
                sphere.position += vels * vel * (update_count *  dt);
            }
        }

        setup_postprocess(0, spheres);
    }

    bool checkOcclusionToGridAABB(utility::Ray& ray) const
    {
        if (!aabb.inside(ray.org))
        {
            auto hp_aabb = aabb.intersect_(ray);

            if (!hp_aabb)
            {
                return {};
            }

            ray.org = hp_aabb->position + 0.001f * ray.dir;
            ray.reset_distance();
        }

        Float3 dda_pos;
        dda_pos = ray.org - aabb.bounds[0];

        float cell_size_x = aabb.edge[0] / GridSize;
        float cell_size_y = aabb.edge[1] / GridSize;
        float cell_size_z = aabb.edge[2] / GridSize;

        const Float3 vmin = aabb.bounds[0];
        const Float3 cell_edge = aabb.edge / Float3(GridSize);

        bool occluded = false;
        simd::SimdRay simd_ray(ray);
        utility::dda(
            dda_pos[0], dda_pos[1], dda_pos[2],
            ray.dir[0], ray.dir[1], ray.dir[2],
            cell_size_x, cell_size_y, cell_size_z,
            GridSize,
            [&](int ix, int iy, int iz)
            {
                const int index = getIndex(ix, iy, iz);
                if (index >= 0)
                {
                    for (auto& simd_sphere : simd_grid[index])
                    {
                        if (simd::checkOcclusion(simd_sphere, simd_ray, simd_sphere.active) != 0)
                        {
                            occluded = true;
                            return true;
                        }
                    }
                }

                return false;
            }
        );
        return occluded;
    }



    utility::HitPoint intersectToGridAABB(const std::vector<utility::Sphere>& spheres, utility::Ray& ray) const
    {
        if (!aabb.inside(ray.org))
        {
            auto hp_aabb = aabb.intersect_(ray);

            if (!hp_aabb)
            {
                return {};
            }

            ray.org = hp_aabb->position + 0.001f * ray.dir;
            ray.reset_distance();
        }
#if 1
#if 1
        Float3 dda_pos;
        dda_pos = ray.org - aabb.bounds[0];

        float cell_size_x = aabb.edge[0] / GridSize;
        float cell_size_y = aabb.edge[1] / GridSize;
        float cell_size_z = aabb.edge[2] / GridSize;

        const Float3 vmin = aabb.bounds[0];
        const Float3 cell_edge = aabb.edge / Float3(GridSize);

        simd::SimdRay simd_ray(ray);
        utility::dda(
            dda_pos[0], dda_pos[1], dda_pos[2],
            ray.dir[0], ray.dir[1], ray.dir[2],
            cell_size_x, cell_size_y, cell_size_z,
            GridSize,
            [&](int ix, int iy, int iz)
            {
                const int index = getIndex(ix, iy, iz);
                if (index >= 0)
                {
                    const auto [cell_vmin, cell_vmax] = compute_cell_aabb(ix, iy, iz);

                    auto current_ray(ray);
                    auto hp = utility::intersectToAABB(current_ray, cell_vmin, cell_vmax);

                    float max_distance = std::numeric_limits<float>::infinity();

                    if (hp)
                    {
                        current_ray.org = hp->position + current_ray.dir * 0.00001f;
                        auto hp2 = utility::intersectToAABB(current_ray, cell_vmin, cell_vmax);

                        if (hp2)
                        {
                            max_distance = hp->distance + hp2->distance;
                        }
                    }

                    uint16_t hitMask = 0;
                    for (auto& simd_sphere : simd_grid[index])
                    {
                        hitMask |= simd::checkIntersectionToSphere(simd_sphere, simd_ray, max_distance, simd_sphere.active);
                    }

                    if (hitMask != 0)
                    {
                        return true;
                    }

                }

                return false;
            }
        );

#else

        simd::SimdRay simd_ray(ray);

        for (auto& simd_sphere : simd_spheres)
        {
            simd::checkIntersectionToSphere(simd_sphere, simd_ray);
        }
#endif

        auto hp = simd::merge(ray, simd_ray);

        if (hp.hit)
        {
            hp.normal = spheres[hp.index].computeNormal(hp.position);
        }

        return hp;
#else
        int hit_index = -1;
        for (int i = 0; i < spheres.size(); ++i)
        {
            if (spheres[i].checkIntersectionAndUpdateRay(ray))
            {
                hit_index = i;
            }
        }

        if (hit_index == -1)
        {
            return {};
        }

        utility::HitPoint hp;
        hp.hit = true;
        hp.index = hit_index;
        hp.position = ray.intersectedPoint();
        hp.normal = spheres[hit_index].computeNormal(hp.position);
        hp.distance = ray.distance_to_intersection;
        return hp;
#endif
    }


    utility::HitPoint intersect(int phase, const std::vector<utility::Sphere>& spheres, utility::Ray& ray) const
    {
        utility::Ray primaryRay(ray);

        auto grid_aabb_hp = intersectToGridAABB(spheres, primaryRay);

        if (grid_aabb_hp.hit)
        {
            return grid_aabb_hp;
        }


        if (phase == 0)
        {

            // îwåiÅiè∞ÅjÇ∆ÇÃåç∑îªíË

            {
                auto hp = utility::intersectToPlane(ray, Float3(0, -0.8f, 0), Float3(0, 1.0f, 0));
                if (hp)
                {
                    hp->index = -2;
                    return *hp;
                }
            }
        }
        else if (phase == 1)
        {
            {
                auto hp = utility::intersectToPlane(ray, Float3(0, -0.35f, 0), Float3(0, 1.0f, 0));
                if (hp)
                {
                    hp->index = -2;
                    return *hp;
                }
            }

            /*
            {
                auto hp = utility::intersectToPlane(ray, Float3(0, 0, -1.0f), Float3(0, 0, 1.0f));
                if (hp)
                {
                    hp->index = -3;
                    return *hp;
                }
            }
            */
        }


        return {};
    }

};

struct ImageSet
{
    std::vector<utility::Image> images;
    std::vector<utility::Image> objImages;
    utility::Image aaImage;
};

struct Context
{
    Scene scene;
    utility::Image* bn;
    utility::Image* moji;
    std::array<ImageSet, 3>* imageSets;
    int frame_number;

    std::vector<utility::Sphere>* spheres;
    int phase = 0;
};

struct Pixel
{
    Float3 L;
    int objectID;
};

Pixel radiance(Context& ctx, Random& rng, const utility::Ray& primaryRay)
{
    Float3 background(1.2f);
    //const Float3 background(0.2f);

    if (ctx.phase == 0)
    {
        background *= 1.2f;
    }

    if (ctx.phase == 1)
    {
        background *= 1.5f;
    }


#if 0

    // AABB
    {
        auto hp = ctx.scene.aabb.intersect_(primaryRay);
        if (hp)
        {
            int ix, iy, iz;
            ctx.scene.aabb.computeIndex(hp->position, ix, iy, iz);

            Random rng(ix + iy * GridSize + iz * GridSize * GridSize);

            Pixel p;
            
            p.L = Float3(rng.next01(), rng.next01(), rng.next01());
            p.objectID = rng.next();

            return p;
        }
    }
#endif 

#if 1
    utility::Ray primary_ray(primaryRay);

    Float3 throughput(1.0f);

    utility::Ray current_ray(primary_ray);

    Float3 L(0.0f);
    Pixel p;
    p.objectID = -2;

    for (int bounce = 0; bounce < 5; ++bounce)
    {
        current_ray.reset_distance();

        auto hp = ctx.scene.intersect(ctx.phase, *ctx.spheres, current_ray);

        if (!hp.hit)
        {
            L += throughput * background;
            break;
        }

        // return (hp.normal + Float3(1.0f)) * 0.5f;

        Float3 normal = hp.normal;
        Float3 tangent, binormal;
        utility::createOrthoNormalBasis<Float3>(normal, &tangent, &binormal);

        const float u0 = rng.next01();
        const float u1 = rng.next01();
        Float3 localDir = utility::sampleCosWeightedHemisphere(u0, u1);
        Float3 worldDir = localDir[2] * normal + localDir[1] * binormal + localDir[0] * tangent;

        current_ray.org = hp.position + normal * 0.00001f;
        current_ray.dir = worldDir;

        if (bounce == 0)
        {
            p.objectID = hp.index;
        }

        if (hp.index != -1)
        {
            Float3 color;
            if (ctx.phase == 0)
            {
                if (hp.index >= 0)
                {
                    if ((*ctx.spheres)[hp.index].metaID == 1)
                    {
                        hp.index = 0;
                    }

                    color = ctx.scene.color_table[hp.index % ctx.scene.color_table.size()];


                    if ((*ctx.spheres)[hp.index].metaID == 2)
                    {
                        color = Float3(0.8f);
                    }
                    if ((*ctx.spheres)[hp.index].metaID == 3)
                    {
                        color = Float3(0.85f, 0.1f, 0.1f);
                    }
                }
                else
                {
                    color = ctx.scene.color_table2[(-hp.index) % ctx.scene.color_table2.size()];
                }
            }
            else if (ctx.phase == 1)
            {
                if (hp.index >= 0)
                {

                    color = ctx.scene.color_table[hp.index % ctx.scene.color_table.size()];


                    if ((*ctx.spheres)[hp.index].metaID == 1)
                    {
                        color = Float3(0.2f, 0.2f, 0.9f);
                    }
                    else if ((*ctx.spheres)[hp.index].metaID == 2)
                    {
                        color = Float3(0.2f, 0.9f, 0.2f);
                    }
                    else if ((*ctx.spheres)[hp.index].metaID == 3)
                    {
                        color = Float3(0.9f, 0.2f, 0.0f);
                    }
                    else if ((*ctx.spheres)[hp.index].metaID == 4)
                    {
                        color = Float3(0.7f, 0.7f, 0.2f);
                    }
                    else if ((*ctx.spheres)[hp.index].metaID == 5)
                    {
                        color = Float3(0.2f, 0.3f, 0.9f);
                    }
                }
                else
                {
                    color = ctx.scene.color_table2[(-hp.index) % ctx.scene.color_table2.size()];
                }
            }


#if 0
            if (hp.index == -2) // è∞
            {
                utility::Ray shadowRay;


                Float3 o;

                o[0] = rng.next(-0.15f, 0.15f);
                o[1] = rng.next(-0.15f, 0.15f);
                o[2] = rng.next(-0.15f, 0.15f);

                shadowRay.dir = normalize(o-current_ray.org); // å¥ì_Ç…å¸Ç©Ç§
                shadowRay.org = current_ray.org + 0.01f * shadowRay.dir;
                shadowRay.reset_distance();
                if (!ctx.scene.checkOcclusionToGridAABB(shadowRay))
                {
                    L += throughput * Float3(10, 10, 10);
                }
            }
#endif


            if (color[0] >= 0)
            {
                throughput *= color;
            }
            else
            {
                L += throughput * (-color); // emission
            }
        }
    }


    p.L = L;

    return p;
#endif
#if 0
    utility::Ray ray(primaryRay);

    auto hp = ctx.scene.intersect(ray);

    if (!hp.hit)
    {
        return 0; // Background
    }

    Float3 normal = hp.normal;
    Float3 tangent, binormal;
    utility::createOrthoNormalBasis<Float3>(normal, &tangent, &binormal);

    const int N = 16;

    float ao = 0;

    for (int i = 0; i < N; ++i)
    {
        
        const float u0 = rng.next01();
        const float u1 = rng.next01();
        

        /*
        float u0 = utility::frac((float)i / PHI);
        float u1 = utility::frac((float)i / N);
        */

        Float3 localDir = utility::sampleCosWeightedHemisphere(u0, u1);

        Float3 worldDir = localDir[2] * normal + localDir[1] * binormal + localDir[0] * tangent;

        utility::Ray aoRay;
        aoRay.org = hp.position + normal * 0.001f;
        aoRay.dir = worldDir;
        
        const bool occluded = ctx.scene.checkOcclusionInsideAABB(aoRay);

        ao += (float)occluded / N;
    }

    return 1 - ao;

//    return dot(hp.normal, Float3(0.7, 0.5, 0.8));

#endif
}

void render(Context& ctx, utility::TaskExecutor& executor, const char* dir, int frame_number, int width, int height)
{
    utility::Timer _("render");
    utility::Image image(width, height);

    const auto aspect = (float)g_param.width / g_param.height;
    renderer::Camera camera;

    if (frame_number <= 105)
    {

        camera.org = Float3(3, 3, 7.0f);
        camera.dir = normalize(Float3(0, 0, 0) - camera.org);


        camera.org += -camera.dir * (frame_number / 72.0f);

        camera.up = Float3(0, 1, 0);
        camera.distance_to_film = 1.5f + (frame_number / 72.0f) * 0.3f;
        camera.film_height = 0.5f;
        camera.film_width = camera.film_height * aspect;
    }
    else if (frame_number <= 300)
    {
        camera.org = Float3(0, 0.3, 3.0f);
        camera.dir = normalize(Float3(0, -0.2f, 0) - camera.org);


        camera.org += -camera.dir * (frame_number / 72.0f);

        camera.up = Float3(0, 1, 0);
        camera.distance_to_film = 1.5f + (frame_number / 72.0f) * 0.3f;
        camera.film_height = 0.5f;
        camera.film_width = camera.film_height * aspect;
    }

    const int total_sample = g_param.super_sample_count;

    const int dither[8][8] = {
        { 0, 32,  8, 40,  2, 34, 10, 42 },
        { 48, 16, 56, 24, 50, 18, 58, 26 },
        { 12, 44,  4, 36, 14, 46,  6, 38 },
        { 60, 28, 52, 20, 62, 30, 54, 22 },
        { 3,  35, 11, 43,  1, 33,  9, 41 },
        { 51, 19, 59, 27, 49, 17, 57, 25 },
        { 15, 47,  7, 39, 13, 45,  5, 37 },
        { 63, 31, 55, 23, 61, 29, 53, 21 }
    };

    //Image objectImage(width, height);

    const int current = frame_number % ctx.imageSets->size();

    auto& current_images = (*ctx.imageSets)[current].images;
    auto& current_objImages = (*ctx.imageSets)[current].objImages;

    int rx = 0;
    int ry = 0;

    Random rng(frame_number);

    rx = rng.next(0, 64);
    ry = rng.next(0, 64);

    auto renderBlock = [&](int bx, int by, int ex, int ey) {
        for (int iy = by; iy < ey; ++iy)
        {
            for (int ix = bx; ix < ex; ++ix)
            {
                Random rng;
                //            rng.set_seed(ix + iy * width);
                //            rng.set_seed(dither[ix%8][iy%8]);

                int val = ctx.bn->load((ix + rx) % ctx.bn->width_, (iy + ry) % ctx.bn->height_)[0];


                //image.store(ix, iy, Float3(val / 255.0f));
                //continue;

                rng.set_seed(val);

                rng.next();

                for (int ss = 0; ss < total_sample; ++ss)
                {
                    /*
                    int sx = total_sample / ss_count;
                    int sy = total_sample % ss_count;
                    */

                    float u0 = utility::frac((float)ss / PHI);
                    float u1 = utility::frac((float)ss / total_sample);

                    /*
                    const float u = -(ix + (0.5f) / ss_count) / width * 2 + 1;
                    const float v = (iy + (0.5f) / ss_count) / height * 2 - 1;
                    */
                    const float u = -(ix + u0) / width * 2 + 1;
                    const float v = (iy + u1) / height * 2 - 1;

                    const auto ray = generateCameraRay(camera, u, v);
                    const auto p = radiance(ctx, rng, ray);
                    // image.accum(ix, iy, c * (1.0f / total_sample));

                    current_images[ss].store(ix, iy, p.L * (1.0f / total_sample));
                    current_objImages[ss].store(ix, iy, p.objectID);
                }
            }
        }
    };

    utility::ThreadDispacher dispatcher;
    const int blockWidth = 4;
    const int blockHeight = 4;

    for (int iy = 0; iy < height; iy += blockHeight)
    {
        for (int ix = 0; ix < width; ix += blockWidth)
        {
            auto renderBlock2 = [ix, iy, blockWidth, blockHeight, width, height, renderBlock]()
            {
                renderBlock(ix, iy,
                    std::min<int>(ix + blockWidth, width - 1),
                    std::min<int>(iy + blockHeight, height - 1));
            };
            dispatcher.append(renderBlock2);
        }
    }

    dispatcher.start(omp_get_max_threads());

    dispatcher.wait();


    // 3x3 ÉuÉâÅ[

//    utility::Image aaImage = images[0];


    auto& aaImage = (*ctx.imageSets)[current].aaImage;
    utility::Image finalImage(width, height);

#if 0
    utility::Image simpleImage(width, height);
#pragma omp parallel for schedule(dynamic, 1)
    for (int iy = 0; iy < height; ++iy)
    {
        for (int ix = 0; ix < width; ++ix)
        {
            Float3 simpleSum(0.0f);
            for (int s = 0; s < total_sample; ++s)
            {
                const int current = frame_number % ctx.imageSets->size();
                auto& current_images = (*ctx.imageSets)[current].images;
                simpleSum += current_images[s].load(ix, iy);
            }
            simpleImage.store(ix, iy, simpleSum);
        }
    }
#endif

    auto blurBlock = [&](int bx, int by, int ex, int ey) {
        for (int iy = by; iy < ey; ++iy)
        {
            for (int ix = bx; ix < ex; ++ix)
            {
                Float3 sum(0.0f);

#if 1
                for (int s = 0; s < total_sample; ++s)
                {
                    const int current = frame_number % ctx.imageSets->size();
                    auto& current_images = (*ctx.imageSets)[current].images;
                    auto& current_objImages = (*ctx.imageSets)[current].objImages;
                    int object_id = current_objImages[s].load(ix, iy)[0];

                    const float weight[5][5] = {
                        {1.0 / 273,  4.0 / 273,  7.0 / 273,  4.0 / 273,  1.0 / 273},
                        {4.0 / 273, 16.0 / 273, 26.0 / 273, 16.0 / 273,  4.0 / 273},
                        {7.0 / 273, 26.0 / 273, 41.0 / 273, 26.0 / 273,  7.0 / 273},
                        {4.0 / 273, 16.0 / 273, 26.0 / 273, 16.0 / 273,  4.0 / 273},
                        {1.0 / 273,  4.0 / 273,  7.0 / 273,  4.0 / 273,  1.0 / 273}
                    };

                    Float3 tmpsum(0.0f);
                    float weight_sum = 0;

                    for (int t = 0; t < ctx.imageSets->size(); ++t)
                    {
                        const int target = (current + (ctx.imageSets->size() - t)) % ctx.imageSets->size();
                        auto& target_images = (*ctx.imageSets)[target].images;
                        auto& target_objImages = (*ctx.imageSets)[target].objImages;
                        //const float temporal_weight = 1.0f / (abs(current - target) + 1.0f);

                        const float temporal_weight = 1;
                        for (int oy = -2; oy <= 2; ++oy)
                        {
                            for (int ox = -2; ox <= 2; ++ox)
                            {
                                int another_id = target_objImages[s].load(ix + ox, iy + oy)[0];

                                if (object_id == another_id)
                                {
                                    float w = weight[oy + 2][ox + 2] * temporal_weight;

                                    w = w * w;

                                    tmpsum += w * target_images[s].load(ix + ox, iy + oy);
                                    weight_sum += w;
                                }
                            }
                        }
                    }

                    tmpsum /= weight_sum;

                    sum += tmpsum;
                }

#else
                const int current = frame_number % ctx.imageSets->size();
                auto& current_images = (*ctx.imageSets)[current].images;
                auto& current_objImages = (*ctx.imageSets)[current].objImages;

                for (int s = 0; s < total_sample; ++s)
                {
                    sum += current_images[s].load(ix, iy);
                }


#endif

                aaImage.store(ix, iy, sum);
            }
        }
    };


    {
        utility::Timer _("  blur");
        utility::ThreadDispacher dispatcher2;
        const int blockWidth = 8;
        const int blockHeight = 8;

        for (int iy = 2; iy < height; iy += blockHeight)
        {
            for (int ix = 2; ix < width; ix += blockWidth)
            {
                auto blurBlock2 = [ix, iy, blockWidth, blockHeight, width, height, blurBlock]()
                {
                    blurBlock(ix, iy,
                        std::min<int>(ix + blockWidth, width - 2),
                        std::min<int>(iy + blockHeight, height - 2));
                };
                dispatcher2.append(blurBlock2);
            }
        }

        dispatcher2.start(omp_get_max_threads());

        dispatcher2.wait();


        auto& src = aaImage;

        for (int iy = 0; iy < height; ++iy)
        {
            int sx, sy;
            sx = 2;
            sy = utility::clampValue(iy, 2, height - 3);
            aaImage.store(0, iy, src.load(sx, sy));
            aaImage.store(1, iy, src.load(sx, sy));

            sx = width - 3;
            sy = utility::clampValue(iy, 2, height - 3);
            aaImage.store(width - 2, iy, src.load(sx, sy));
            aaImage.store(width - 1, iy, src.load(sx, sy));
        }
        for (int ix = 0; ix < width; ++ix)
        {
            int sx, sy;
            sx = utility::clampValue(ix, 2, width - 3);
            sy = 2;
            aaImage.store(ix, 0, src.load(sx, sy));
            aaImage.store(ix, 1, src.load(sx, sy));

            sx = utility::clampValue(ix, 2, width - 3);
            sy = height-3;
            aaImage.store(ix, height - 2, src.load(sx, sy));
            aaImage.store(ix, height - 1, src.load(sx, sy));
        }

        if (ctx.phase == 0)
        {

#pragma omp parallel for schedule(dynamic, 1)
            for (int iy = 0; iy < height; ++iy)
            {
                for (int ix = 0; ix < width; ++ix)
                {
                    Float3 sum = 0.0f;
                    for (int i = 0; i < 3; ++i)
                    {
                        sum += (*ctx.imageSets)[i].aaImage.load(ix, iy) / 3.0f;
                    }
                    finalImage.store(ix, iy, sum);
                }
            }
        }

    }


#if 0
    utility::Image aaImage(image.width_, image.height_);

    auto blurBlock = [&](int bx, int by, int ex, int ey) {
        for (int iy = by; iy < ey; ++iy)
        {
            for (int ix = bx; ix < ex; ++ix)
            {
                float weight_sum = 0;

                const float weight[3][3] = {
                    {1.0f / 16.0f, 1.0f / 8.0f, 1.0f / 16.0f},
                    {1.0f / 8.0f, 1.0f / 4.0f, 1.0f / 8.0f},
                    {1.0f / 16.0f, 1.0f / 8.0f, 1.0f / 16.0f},
                };

                Float3 sum(0.0f);
                for (int oy = -1; oy <= 1; ++oy)
                {
                    for (int ox = -1; ox <= 1; ++ox)
                    {
                        const float w = weight[oy + 1][ox + 1];

                        sum += w * image.load(ix + ox, iy + oy);
                    }
                }

                aaImage.store(ix, iy, sum);
            }
        }
    };

    {
        utility::Timer _("  blur");
        utility::ThreadDispacher dispatcher2;
        const int blockWidth = 8;
        const int blockHeight = 8;

        for (int iy = 0; iy < height; ++iy)
        {
            aaImage.store(0, iy, image.load(0, iy));
            aaImage.store(width-1, iy, image.load(width-1, iy));
        }
        for (int ix = 0; ix < width; ++ix)
        {
            aaImage.store(ix, 0, image.load(ix, 0));
            aaImage.store(ix, height-1, image.load(ix, height-1));
        }

        for (int iy = 1; iy < height; iy += blockHeight)
        {
            for (int ix = 1; ix < width; ix += blockWidth)
            {
                auto blurBlock2 = [ix, iy, blockWidth, blockHeight, width, height, blurBlock]()
                {
                    blurBlock(ix, iy,
                        std::min<int>(ix + blockWidth, width - 2),
                        std::min<int>(iy + blockHeight, height - 2));
                };
                dispatcher2.append(blurBlock2);
            }
        }

        dispatcher2.start(omp_get_max_threads());

        dispatcher2.wait();
    }
#endif

#if 0
#pragma omp parallel for schedule(dynamic, 1)
    for (int iy = 0; iy < height; ++iy)
    {
        for (int ix = 0; ix < width; ++ix)
        {
            Random rng;
//            rng.set_seed(ix + iy * width);
//            rng.set_seed(dither[ix%8][iy%8]);

            int val = ctx.bn->load(ix % ctx.bn->width_, iy % ctx.bn->height_)[0];


            //image.store(ix, iy, Float3(val / 255.0f));
            //continue;

            rng.set_seed(val);

            rng.next();

            for (int sx = 0; sx < ss_count; ++sx)
            {
                for (int sy = 0; sy < ss_count; ++sy)
                {
                    /*
                    const float u = -(ix + (0.5f) / ss_count) / width * 2 + 1;
                    const float v = (iy + (0.5f) / ss_count) / height * 2 - 1;
                    */
                    const float u = -(ix + (sx + 0.5f) / ss_count) / width * 2 + 1;
                    const float v = (iy + (sy + 0.5f) / ss_count) / height * 2 - 1;

                    const auto ray = generateCameraRay(camera, u, v);

                    const Float3 c = radiance(ctx, rng, ray);
                    image.accum(ix, iy, c * (1.0f / (ss_count * ss_count)));
                }
            }
        }
    }
#endif


    float screen_scale = 1;


    if (280 <= frame_number && frame_number <= 299)
    {
        screen_scale *= (1.0f - ((frame_number - 280) / 19.0f));
    }

    if (0 <= frame_number && frame_number <= 5)
    {
        screen_scale *= (frame_number) / 5.0f;
    }

    auto ldr_image = tonemap(screen_scale, ctx.phase == 0 ?  finalImage : aaImage);
#if 0
    auto ldr_image = tonemap(screen_scale, simpleImage);
#endif

    char buf[256];
    snprintf(buf, sizeof(buf), "%s/%03d.jpg", dir, frame_number);
    executor.appendTask([=]() {
        utility::writeJPEGImage(buf, width, height, 3, ldr_image.data(), 100);
    });
}

#if 0
void test()
{
    {
        simd::float16 a(std::array<float, 16>{
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        });
        simd::float16 x(std::array<float, 16>{
            15, 14, 12, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0
        });

        simd::float16 b = simd::float16::one();

        //a += b;

        utility::printArray(a.toArray());

        auto c = a > simd::float16::zero();

        utility::printBitPattern(c);
    }
    {

        uint16_t mask = 0;

        simd::float16 a(-1.0f);
        simd::float16 x(1.0f);

        auto c = simd::select(a, x, mask);

        utility::printArray(c.toArray());
    }

    {
        simd::float16 x(7.0f);
        simd::float16 y = simd::float16::one();
        const uint16_t mask = 0b1011011111001101;

        auto c = simd::move(x, y, mask);

        utility::printArray(c.toArray());
    }
}
#endif

#if 0
void test2()
{
    utility::Image image(512, 512);


    float cell_size = 1.0f;

    const int N = 512;

    utility::random::PCG_64_32 rng(42);

    for (int line = 0; line < 1; ++line)
    {
        float x0 = rng.next(0, N - 1);
        float y0 = rng.next(0, N - 1);
        float z0 = rng.next(0, N - 1);
        float dx = rng.next(-1, 1);
        float dy = rng.next(-1, 1);
        float dz = rng.next(-1, 1);

        const float l = sqrt(dx * dx + dy * dy + dz * dz);
        dx /= l;
        dy /= l;
        dz /= l;

        const int add_ix = dx >= 0 ? 1 : -1;
        const int add_iy = dy >= 0 ? 1 : -1;
        const int add_iz = dz >= 0 ? 1 : -1;

        while (true)
        {
            const int ix = std::floor(x0 / cell_size);
            const int iy = std::floor(y0 / cell_size);
            const int iz = std::floor(z0 / cell_size);

            printf("[%d, %d, %d]", ix, iy, iz);

            if (ix >= N || iy >= N || iz >= N || ix < 0 || iy < 0 || iz < 0)
            {
                break;
            }

            image.store(ix, iy, utility::Color(1, 1, 1));

            const float left_x = -x0 + (ix + add_ix) * cell_size;
            const float tx = left_x / dx;
            const float left_y = -y0 + (iy + add_iy) * cell_size;
            const float ty = left_y / dy;
            const float left_z = -z0 + (iz + add_iz) * cell_size;
            const float tz = left_z / dz;

            if (tx < ty && tx < tz)
            {
                x0 = (ix + add_ix) * cell_size;
                y0 += tx * dy;
                z0 += tz * dz;
            }
            else if (ty < tx && ty < tz)
            {
                x0 += ty * dx;
                y0 = (iy + add_iy) * cell_size;
                z0 += ty * dz;
            }
            else
            {
                x0 += tz * dx;
                y0 += tz * dy;
                z0 = (iz + add_iz) * cell_size;
            }
        }
    }


    auto ldr_image = tonemap(1.0f, image);

    char buf[256];
    snprintf(buf, sizeof(buf), "./out/test.jpg");
    utility::writeJPEGImage(buf, 512, 512, 3, ldr_image.data(), 100);
}
#endif

#if 0
void test2()
{
    utility::Image image(512, 512);


    float cell_size = 1.0f;

    const int N = 512;

    float x0 = 50.2f;
    float y0 = 10.5f;
    float dx = -1.0f;
    float dy = 0.7f;

    /*
    float x0 = 0.0f;
    float y0 = 0.1f;
    float dx = 0.3f;
    float dy = 0.7f;
    */

    const float l = sqrt(dx * dx + dy * dy);
    dx /= l;
    dy /= l;

    const int left_ix = dx >= 0 ? 1 : 0;
    const int left_iy = dy >= 0 ? 1 : 0;

    const int add_ix = dx >= 0 ? 1 : 0;
    const int add_iy = dy >= 0 ? 1 : 0;

    while (true)
    {
        const int ix = std::floor(x0 / cell_size);
        const int iy = std::floor(y0 / cell_size);

        printf("[%d, %d, (%f, %f)]", ix, iy, x0, y0);

        if (ix >= N || iy >= N || ix < 0 || iy < 0)
        {
            break;
        }

        image.store(ix, iy, utility::Color(1, 1, 1));

        const float left_x = -x0 + (ix + left_ix) * cell_size;
        const float tx = left_x / dx;
        const float left_y = -y0 + (iy + left_iy) * cell_size;
        const float ty = left_y / dy;

        if (ty < tx)
        {
            x0 += ty * dx;
            y0 = (iy + add_iy) * cell_size;
            y0 += 0.0001 * dy;
        }
        else
        {
            x0 = (ix + add_ix) * cell_size;
            x0 += 0.0001 * dx;
            y0 += tx * dy;
        }
    }
    /*
    utility::random::PCG_64_32 rng;

    for (int line = 0; line < 1; ++line)
    {
        float x0 = rng.next(0, N-1);
        float y0 = rng.next(0, N-1);
        float dx = rng.next(-1, 1);
        float dy = rng.next(-1, 1);

        const float l = sqrt(dx * dx + dy * dy);
        dx /= l;
        dy /= l;

        const int add_ix = dx >= 0 ? 1 : -1;
        const int add_iy = dy >= 0 ? 1 : -1;

        while (true)
        {
            const int ix = std::floor(x0 / cell_size);
            const int iy = std::floor(y0 / cell_size);

            // printf("[%d, %d, (%f, %f)]", ix, iy, x0, y0);

            if (ix >= N || iy >= N || ix < 0 || iy < 0)
            {
                break;
            }

            image.store(ix, iy, utility::Color(1, 1, 1));

            const float left_x = -x0 + (ix + add_ix) * cell_size;
            const float tx = left_x / dx;
            const float left_y = -y0 + (iy + add_iy) * cell_size;
            const float ty = left_y / dy;

            if (ty < tx)
            {
                x0 += ty * dx;
                y0 = (iy + add_iy) * cell_size;
            }
            else
            {
                x0 = (ix + add_ix) * cell_size;
                y0 += tx * dy;
            }
        }
    }
*/


    auto ldr_image = tonemap(1.0f, image);

    char buf[256];
    snprintf(buf, sizeof(buf), "./out/test.jpg");
    utility::writeJPEGImage(buf, 512, 512, 3, ldr_image.data(), 100);
}
#endif


void test3()
{
    utility::Image image(512, 512);

    perlin::PerlinNoise n(53);


    float s = 1 / 256.0f;

    for (int iy = 0; iy < image.width_; ++iy)
    {
        for (int ix = 0; ix < image.width_; ++ix)
        {
            const float x = n.noise(ix*s, iy*s, 0);
            image.store(ix, iy, Float3(x));
        }
    }


    auto ldr_image = tonemap(1.0f, image);

    char buf[256];
    snprintf(buf, sizeof(buf), "./out/test.jpg");
    utility::writeJPEGImage(buf, 512, 512, 3, ldr_image.data(), 100);

}

int main(int argc, char** argv)
{
    int t = omp_get_max_threads();
    printf("Threads: %d\n", t);


    /*
    test3();
    return 0;
*/

    
    
    utility::Timer _("main");

    if (argc <= 1)
    {
        printf("Error\n");
        return -1;
    }
    const char* dir = argv[1];
    const int start_frame = argc >= 4 ? atoi(argv[3]) : 0;
    const int frame_limit = argc >= 3 ? atoi(argv[2]) : g_param.max_frame;
    const float dt = g_param.movie_time_sec / g_param.max_frame;
    const int width = g_param.width;
    const int height = g_param.height;

    printf("dt: %f\n", dt);

    const int sphere_count = 4096 * 64;
//    const int sphere_count = 4096;

    utility::TaskExecutor executor;

    auto moji = utility::loadPNGIMage("./moji.png");
    setup_global(moji, sphere_count);

    // g_3d_noise.dump("./out/noise.jpg");


    auto bn = utility::loadPNGIMage("./bn.png");

    std::array<ImageSet, 3> imageSets;

    const int total_sample = g_param.super_sample_count;
    for (auto& imageSet : imageSets)
    {
        imageSet.images.resize(total_sample);
        imageSet.objImages.resize(total_sample);
        imageSet.aaImage = utility::Image(width, height);

        for (int i = 0; i < total_sample; ++i)
        {
            imageSet.images[i] = utility::Image(width, height);
            imageSet.objImages[i] = utility::Image(width, height);
        }
    }
    

    for (int frame_number = start_frame; frame_number < frame_limit; ++frame_number)
    {
        {
            utility::Timer _("frame");

            printf("Frame: %d\n", frame_number);

            Context ctx;
            ctx.bn = &bn;
            ctx.moji = &moji;
            ctx.imageSets = &imageSets;
            ctx.frame_number = frame_number;


            if (frame_number <= 105)
            {
                ctx.phase = 0;
                ctx.spheres = &g_spheres;

                float update_count = 0;
                if (frame_number < 10)
                {
                    update_count = 0;
                }
                else if (frame_number < 45)
                {
                    update_count = 10;
                }
                else
                {
                    update_count = 0.25;
                }

                ctx.scene.setup(g_spheres, sphere_count, dt * frame_number, dt, update_count);
                update(dt);
                render(ctx, executor, dir, frame_number, width, height);
            }
            else if (frame_number <= 300)
            {
                if (frame_number == 210)
                {
                    final_setup(moji);
                }

                ctx.phase = 1;
                ctx.spheres = &g_spheres_cut2;

                ctx.scene.setup_cut2(g_spheres_cut2, sphere_count, dt * frame_number, dt);
                update(dt);
                render(ctx, executor, dir, frame_number, width, height);
            }
        }
    }
    return 0;
}