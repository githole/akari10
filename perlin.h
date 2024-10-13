#pragma once

#include <vector>
#include "utility.h"

namespace perlin
{

inline
float fade(float t)
{
    return t * t * t * (t * (t * 6 - 15) + 10);
}

inline
float grad(int hash, float x, float y, float z)
{
    int h = hash & 15;
    float u = h < 8 ? x : y;
    float v = h < 4 ? y : h == 12 || h == 14 ? x : z;
    return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
}

inline
float lerp(float t, float a, float b) {
    return a + t * (b - a);
}

class PerlinNoise {
public:
    PerlinNoise(unsigned int seed = 2024)
    {
        p.resize(512);
        for (int i = 0; i < 256; ++i)
        {
            p[i] = i;
        }

        utility::random::PCG_64_32 rng(seed);
        for (int i = 255; i > 0; --i)
        {
            int swapIndex = rng.next() % (i + 1);
            std::swap(p[i], p[swapIndex]);
        }

        for (int i = 0; i < 256; ++i)
        {
            p[256 + i] = p[i];
        }
    }

    float noise(float x, float y, float z) const
    {
        x *= 255;
        y *= 255;
        z *= 255;

        int X = static_cast<int>(std::floor(x)) & 255;
        int Y = static_cast<int>(std::floor(y)) & 255;
        int Z = static_cast<int>(std::floor(z)) & 255;

        x -= std::floor(x);
        y -= std::floor(y);
        z -= std::floor(z);

        float u = fade(x);
        float v = fade(y);
        float w = fade(z);

        int A = p[X] + Y;
        int AA = p[A] + Z;
        int AB = p[A + 1] + Z;
        int B = p[X + 1] + Y;
        int BA = p[B] + Z;
        int BB = p[B + 1] + Z;

        float res = lerp(w,
            lerp(v,
                lerp(u, grad(p[AA], x, y, z), grad(p[BA], x - 1, y, z)),
                lerp(u, grad(p[AB], x, y - 1, z), grad(p[BB], x - 1, y - 1, z))),
            lerp(v,
                lerp(u, grad(p[AA + 1], x, y, z - 1), grad(p[BA + 1], x - 1, y, z - 1)),
                lerp(u, grad(p[AB + 1], x, y - 1, z - 1), grad(p[BB + 1], x - 1, y - 1, z - 1))));

        return (res + 1.0f) / 2.0f;
    }

private:
    std::vector<int> p;
};

}