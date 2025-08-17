using Unity.Entities;
using Unity.Mathematics;
using UnityEngine;

namespace Boids
{
    class BoidAuthoring : MonoBehaviour
    {
        class Baker : Baker<BoidAuthoring>
        {
            public override void Bake(BoidAuthoring authoring)
            {
                Entity entity = GetEntity(TransformUsageFlags.Dynamic);
                AddComponent(entity, new Boid
                {
                    Velocity = new float3(0, 0, 0)
                });
            }
        }
    }

    public struct Boid : IComponentData
    {
        public float3 Velocity;
    }
}
