#pragma once

#include "utility.h"

namespace renderer
{

struct Camera
{
    utility::Float3 org;
    utility::Float3 dir;
    utility::Float3 up;

    // [meter]
    float distance_to_film;
    float film_width;
    float film_height;
};


// ‚±‚Ìu, v‚Í[-1, 1]
utility::Ray
generateCameraRay(const Camera& camera, float u, float v)
{
    const auto side = normalize(cross(camera.up, camera.dir));
    const auto up = normalize(cross(side, camera.dir));

    const auto p_on_film =
        camera.org + camera.distance_to_film * camera.dir +
        side * u * camera.film_width / 2.0f +
        up * v * camera.film_height / 2.0f;

    const auto dir = normalize(p_on_film - camera.org);

    return { camera.org, dir };
}




}