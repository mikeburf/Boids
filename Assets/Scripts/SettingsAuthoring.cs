using Unity.Entities;
using Unity.Entities.Serialization;
using Unity.Mathematics;
using UnityEngine;

namespace Boids
{
    class SettingsAuthoring : MonoBehaviour
    {
        public BoidAuthoring Prefab;
        [Min(0)] public int Count = 1000;
        public float SpawnBounds;

        [Header("Config")]
        public float MaxSpeed;
        public float CellSize;
        public float CohesionWeight = 0.01f;
        public float AlignmentWeight = 0.125f;
        public float SeperationWeight = 1f;
        public float SeperationDistance = 1f;
        public float TargetSeekWeight = 1f;

        class Baker : Baker<SettingsAuthoring>
        {
            public override void Bake(SettingsAuthoring authoring)
            {
                Entity prefab = GetEntity(authoring.Prefab, TransformUsageFlags.Dynamic);

                Entity entity = GetEntity(TransformUsageFlags.None);
                AddComponent(entity, new Settings
                {
                    Prefab = prefab,
                    TargetPosition = authoring.transform.position,
                    SqrMaxSpeed = authoring.MaxSpeed * authoring.MaxSpeed,
                    InverseCellSize = 1 / authoring.CellSize,

                    CohesionWeight = authoring.CohesionWeight,
                    AlignmentWeight = authoring.AlignmentWeight,
                    SeperationWeight = authoring.SeperationWeight,
                    SeperationDistance = authoring.SeperationDistance,
                    TargetSeekWeight = authoring.TargetSeekWeight
                });
                AddComponent(entity, new SpawnOnce
                {
                    Count = authoring.Count,
                    SpawnMin = (float3)authoring.transform.position - math.float3(authoring.SpawnBounds),
                    SpawnMax = (float3)authoring.transform.position + math.float3(authoring.SpawnBounds)
                });
            }
        }
    }

    public struct Settings : IComponentData
    {
        public Entity Prefab;
        public float3 TargetPosition;
        public float SqrMaxSpeed;
        public float InverseCellSize;

        public float CohesionWeight;
        public float AlignmentWeight;
        public float SeperationWeight;
        public float SeperationDistance;
        public float TargetSeekWeight;
    }

    public struct SpawnOnce : IComponentData
    {
        public int Count;
        public float3 SpawnMin;
        public float3 SpawnMax;
    }
}
