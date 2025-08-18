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
        public float Damping;
        public float CellSize;
        [Min(0)] public int DetectionCellSize; // the edge size of the AABB used to determine neighbors

        [Header("Weights")]
        public float CohesionWeight = 0.01f;
        public float AlignmentWeight = 0.125f;
        public float SeperationWeight = 1f;
        public float SeperationDistance = 1f;
        public float TargetSeekWeight = 1f;
        public float TargetSeekDistance = 10f;

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
                    Damping = authoring.Damping,
                    CellSize = authoring.CellSize,
                    DetectionCellSize = authoring.DetectionCellSize,

                    CohesionWeight = authoring.CohesionWeight,
                    AlignmentWeight = authoring.AlignmentWeight,
                    SeperationWeight = authoring.SeperationWeight,
                    SeperationDistance = authoring.SeperationDistance,
                    TargetSeekWeight = authoring.TargetSeekWeight,
                    TargetSeekDistance = authoring.TargetSeekDistance
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
        public float Damping;
        public float CellSize;
        public int DetectionCellSize;

        public float CohesionWeight;
        public float AlignmentWeight;
        public float SeperationWeight;
        public float SeperationDistance;
        public float TargetSeekWeight;
        public float TargetSeekDistance;
    }

    public struct SpawnOnce : IComponentData
    {
        public int Count;
        public float3 SpawnMin;
        public float3 SpawnMax;
    }
}
