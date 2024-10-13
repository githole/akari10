#pragma once


#include "utility.h"
#include "vec3.h"

namespace pbd
{
struct DistanceConstraint;

struct Particle
{
    utility::Float3 pos, prevPos, vel;
    float mass;
    int sphereIndex;
    float radius;
    int metaID;

    std::vector<DistanceConstraint*> cs;
};

struct CollisionPair
{
    Particle* p1;
    Particle *p2;
    float r1, r2;

    utility::Float3 delta;
    float dist;
};

struct DistanceConstraint 
{
    int p1, p2;
    float restLength;

    utility:: Float3 cp1, cp2;

    void solve(std::vector<Particle>& particles) const
    {
        auto delta = particles[p2].pos - particles[p1].pos;
        auto dist = length(delta);
        if (dist > 0.0) 
        {
            /*
            float invMass1 = (particles[p1].mass > 0.0f) ? 1.0f / particles[p1].mass : 0.0f;
            float invMass2 = (particles[p2].mass > 0.0f) ? 1.0f / particles[p2].mass : 0.0f;
            float invMassSum = invMass1 + invMass2;
            */

            constexpr float invMass1 = 1;
            constexpr float invMass2 = 1;
            constexpr float invMassSum = invMass1 + invMass2;

            auto correction = (dist - restLength) * (delta / dist);
            
            particles[p1].pos += (invMass1 / invMassSum) * correction;
            particles[p2].pos -= (invMass2 / invMassSum) * correction;
            
            /*
            cp1 = (invMass1 / invMassSum) * correction;
            cp2= -(invMass2 / invMassSum) * correction;
            */
        }
    }
};


void handleCollisionBetweenParticles(const CollisionPair& p, Particle& p1, Particle& p2, float restitution, float friction, float radius1, float radius2)
{
    utility::Float3 delta = p.delta;
    float dist = p.dist;
    float penetration = (radius1 + radius2) - dist;

    // 衝突が発生している場合
    if (penetration > 0.0f && dist > 0.0f)
    {
        utility::Float3 normal = delta / dist;

        float invMass1 = (p1.mass > 0.0f) ? 1.0f / p1.mass : 0.0f;
        float invMass2 = (p2.mass > 0.0f) ? 1.0f / p2.mass : 0.0f;
        float invMassSum = invMass1 + invMass2;

        utility::Float3 correction = normal * (penetration / invMassSum);
        p1.pos -= correction * invMass1;
        p2.pos += correction * invMass2;

        utility::Float3 relativeVelocity = p2.vel - p1.vel;
        float velAlongNormal = dot(relativeVelocity, normal);

        if (velAlongNormal < 0.0f)
        {
            float j = -(1.0f + restitution) * velAlongNormal;
            j /= invMassSum;

            utility::Float3 impulse = normal * j;
            p1.vel -= impulse * invMass1;
            p2.vel += impulse * invMass2;

            utility::Float3 tangent = relativeVelocity - (normal * velAlongNormal);
            if (length(tangent) > 0.0f)
            {
                tangent = normalize(tangent);
                float jt = -dot(relativeVelocity, tangent);
                jt /= invMassSum;

                utility::Float3 frictionImpulse;
                if (fabs(jt) < j * friction)
                {
                    frictionImpulse = tangent * jt;
                }
                else
                {
                    frictionImpulse = tangent * (-j * friction);
                }

                p1.vel -= frictionImpulse * invMass1;
                p2.vel += frictionImpulse * invMass2;
            }
        }
    }
}

void applyExternalForces(std::vector<Particle>& particles, float dt)
{
    const utility::Float3 g(0.0, -9.81f, 0.0);
//#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < particles.size(); ++i)
    {
        auto& particle = particles[i];
        if (particle.mass > 0.0)
        {
            particle.vel += g * dt;
            particle.pos += particle.vel * dt;
        }
    }
}

void solveConstraints(std::vector<DistanceConstraint>& constraints, std::vector<Particle>& particles, int iteration)
{
#if 0
    for (int i = 0; i < iteration; ++i)
    {
//#pragma omp parallel for schedule(dynamic, 1)
        for (int p = 0; p < constraints.size(); ++p)
        {
            constraints[p].solve(particles);
        }

//#pragma omp parallel for schedule(dynamic, 1)
        for (int p = 0; p < particles.size(); ++p)
        {
            for (auto& c : particles[p].cs)
            {
                if (c->p1 == p)
                {
                    particles[p].pos += c->p1;
                }
                else 
                {
                    particles[p].pos += c->p2;
                }
            }
        }
    }
#endif

    for (int i = 0; i < iteration; ++i) 
    {
        for (auto& constraint : constraints)
        {
            constraint.solve(particles);
        }
    }
}

void updateVelocities(std::vector<Particle>& particles, float dt) 
{
//#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < particles.size(); ++i)
    {
        auto& particle = particles[i];
        if (particle.mass > 0.0)
        {
            particle.vel = (particle.pos - particle.prevPos) / dt;
        }
    }
}


void handleCollisionWithPlane(Particle& particle, double restitution, double friction)
{
    const float py = -0.35f;
    if (particle.pos[1] - particle.radius < py)
    {
        particle.pos[1] = py + particle.radius;
        particle.vel[1] *= -restitution;
        particle.vel[0] *= friction;
        particle.vel[2] *= friction;
    }
}

inline const int GridSize = 16;

inline
utility::Float3 worldToUV(const utility::Float3& pos)
{
    return (pos - utility::Float3(-1.0f)) * 0.5f;
}


inline
void computeIndex(const utility::Float3& pos, int& ix, int& iy, int& iz)
{
    const auto f = worldToUV(pos) * (float)GridSize;

    ix = std::floor(f[0]);
    iy = std::floor(f[1]);
    iz = std::floor(f[2]);
}

inline
int getIndex(int ix, int iy, int iz)
{
    int index = ix + iy * GridSize + iz * GridSize * GridSize;

    if (ix < 0 || iy < 0 || iz < 0 ||
        GridSize <= ix || GridSize <= iy || GridSize <= iz)
    {
        return -1;
    }

    return index;
}

void simulate(
    std::vector<Particle>& particles,
    std::vector<DistanceConstraint>& constraints, 
    std::vector<CollisionPair>& collisions, 
    float dt)
{
    const int iteration = 5;

    const double restitution = 0.9;
    const double friction = 0.9;


    static std::vector<std::vector<Particle*>> grid(GridSize * GridSize * GridSize);

    // substep的な
    for (int i = 0; i < iteration; ++i)
    {
        for (auto& particle : particles)
        {
            particle.prevPos = particle.pos;
        }

        applyExternalForces(particles, dt / iteration);
        {
            // utility::Timer _("    solveConstraints");
            solveConstraints(constraints, particles, 7);
        }
        updateVelocities(particles, dt / iteration);




        // 粒子同士のコリジョン
        {
            // utility::Timer _("    collisions");

            collisions.clear();

            for (auto& v : grid)
            {
                v.clear();
            }

            for (int i = 0; i < particles.size(); ++i)
            {
                int ix, iy, iz;
                computeIndex(particles[i].pos, ix, iy, iz);
                int index = getIndex(ix, iy, iz);
                if (index >= 0)
                {
                    grid[index].push_back(&particles[i]);
                }
            }

            for (int i = 0; i < particles.size(); ++i)
            {
                int ix, iy, iz;
                computeIndex(particles[i].pos, ix, iy, iz);
                auto& p1 = particles[i];
                for (int ox = -1; ox <= 1; ++ox)
                {
                    for (int oy = -1; oy <= 1; ++oy)
                    {
                        for (int oz = -1; oz <= 1; ++oz)
                        {
                            int cx = ix + ox;
                            int cy = iy + oy;
                            int cz = iz + oz;
                            int index = getIndex(cx, cy, cz);
                            if (index < 0)
                            {
                                continue;
                            }

                            auto& g = grid[index];

                            for (auto* p2 : g)
                            {
                                if (p1.metaID == p2->metaID)
                                {
                                    continue;
                                }

                                const float l2 = length2(p1.pos - p2->pos);
                                if (l2 < (p1.radius + p2->radius) * (p1.radius + p2->radius))
                                {
                                    CollisionPair col;
                                    col.p1 = &p1;
                                    col.p2 = p2;
                                    col.r1 = p1.radius;
                                    col.r2 = p2->radius;
                                    col.delta = p2->pos - p1.pos;
                                    col.dist = sqrt(l2);
                                    collisions.push_back(col);
                                }
                            }
                        }
                    }
                }
            }

            for (auto& col : collisions)
            {
                handleCollisionBetweenParticles(
                    col,
                    *col.p1, *col.p2,
                    restitution, friction,
                    col.r1, col.r2
                );
            }
        }

        //#pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < particles.size(); ++i)
        {
            handleCollisionWithPlane(particles[i], restitution, friction);
        }

#if 0
        if (i == iteration - 1)
        {
            for (auto& col : collisions)
            {
                handleCollisionBetweenParticles(
                    col,
                    particles[col.p1], particles[col.p2],
                    restitution, friction,
                    col.r1, col.r2
                );
            }
        }
#endif
    }

}

}