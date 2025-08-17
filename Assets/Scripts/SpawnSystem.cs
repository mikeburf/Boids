using Unity.Burst;
using Unity.Entities;
using Unity.Collections;
using Unity.Transforms;
using Unity.Mathematics;

namespace Boids
{
    partial struct SpawnSystem : ISystem
    {
        EntityQuery spawnerQuery;

        [BurstCompile]
        public void OnCreate(ref SystemState state)
        {
            state.RequireForUpdate<Settings>();

            spawnerQuery = new EntityQueryBuilder(Allocator.Temp)
                .WithAll<SpawnOnce>()
                .Build(ref state);
        }

        [BurstCompile]
        public void OnUpdate(ref SystemState state)
        {
            Settings setup = SystemAPI.GetSingleton<Settings>();

            if (!spawnerQuery.IsEmpty)
            {
                foreach (var spawnOnce in SystemAPI.Query<RefRO<SpawnOnce>>())
                {
                    NativeArray<Entity> newBoids = state.EntityManager.Instantiate(setup.Prefab, spawnOnce.ValueRO.Count, Allocator.Temp);
                    Random r = new Random(1); //TODO: Implement a better seed

                    foreach (Entity newBoid in newBoids)
                    {
                        LocalTransform t = state.EntityManager.GetComponentData<LocalTransform>(newBoid);
                        t.Position = r.NextFloat3(spawnOnce.ValueRO.SpawnMin, spawnOnce.ValueRO.SpawnMax);
                        state.EntityManager.SetComponentData(newBoid, t);
                    }
                }
                state.EntityManager.RemoveComponent<SpawnOnce>(spawnerQuery);
            }

        }

        [BurstCompile]
        public void OnDestroy(ref SystemState state)
        {

        }
    }
}
