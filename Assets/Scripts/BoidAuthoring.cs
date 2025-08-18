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
                Entity entity = GetEntity(TransformUsageFlags.Renderable | TransformUsageFlags.WorldSpace);
                AddComponent(entity, new Boid
                {

                });
            }
        }
    }

    public struct Boid : IComponentData
    {
    }
}
