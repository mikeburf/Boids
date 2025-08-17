using Unity.Burst;
using Unity.Collections;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Transforms;
using UnityEngine.LightTransport;

namespace Boids
{
    [UpdateInGroup(typeof(FixedStepSimulationSystemGroup))]
    partial struct BoidSystem : ISystem
    {
        EntityQuery boidQuery;

        [BurstCompile]
        public void OnCreate(ref SystemState state)
        {
            state.RequireForUpdate<Settings>();

            boidQuery = new EntityQueryBuilder(Allocator.Temp)
                .WithAll<Boid, LocalTransform>()
                .Build(ref state);
        }

        [BurstCompile]
        public void OnUpdate(ref SystemState state)
        {
            Settings settings = SystemAPI.GetSingleton<Settings>();

            int boidCount = boidQuery.CalculateEntityCount();
            var CellHash = new NativeParallelMultiHashMap<int, int>(boidCount, state.WorldUnmanaged.UpdateAllocator.ToAllocator);
            var CellIndices = CollectionHelper.CreateNativeArray<int, RewindableAllocator>(boidCount, ref state.WorldUnmanaged.UpdateAllocator);
            var Positions = CollectionHelper.CreateNativeArray<float3, RewindableAllocator>(boidCount, ref state.WorldUnmanaged.UpdateAllocator);
            var Velocities = CollectionHelper.CreateNativeArray<float3, RewindableAllocator>(boidCount, ref state.WorldUnmanaged.UpdateAllocator);
            var AvgPositions = CollectionHelper.CreateNativeArray<float3, RewindableAllocator>(boidCount, ref state.WorldUnmanaged.UpdateAllocator);
            var AvgVelocities = CollectionHelper.CreateNativeArray<float3, RewindableAllocator>(boidCount, ref state.WorldUnmanaged.UpdateAllocator);
            var SeperationVectors = CollectionHelper.CreateNativeArray<float3, RewindableAllocator>(boidCount, ref state.WorldUnmanaged.UpdateAllocator);

            var hashJobHandle = new HashJob
            {
                CellHash = CellHash.AsParallelWriter(),
                CellIndices = CellIndices,
                InverseCellSize = settings.InverseCellSize,
            }.ScheduleParallel(boidQuery, state.Dependency);

            var dataJobHandle = new DataJob
            {
                Positions = Positions,
                Velocities = Velocities,
            }.ScheduleParallel(boidQuery, state.Dependency);

            var initHandle = JobHandle.CombineDependencies(hashJobHandle, dataJobHandle);

            var avgJobHandle = new AvgJob
            {
                CellHash = CellHash,
                CellIndices = CellIndices,
                Positions = Positions,
                Velocities = Velocities,
                AvgPosition = AvgPositions,
                AvgVelocity = AvgVelocities
            }.ScheduleParallel(boidCount, 32, initHandle);

            var seperationJobHandle = new SeperationJob
            {
                CellHash = CellHash,
                CellIndices = CellIndices,
                Positions = Positions,
                SeperationVectors = SeperationVectors,
                SqrSeperationDistance = settings.SeperationDistance * settings.SeperationDistance,
            }.ScheduleParallel(boidCount, 32, initHandle);

            var calcJobHandle = JobHandle.CombineDependencies(avgJobHandle, seperationJobHandle);

            var boidUpdateJobHandle = new BoidUpdateJob
            {
                AvgPositions = AvgPositions,
                AvgVelocities = AvgVelocities,
                SeperationVectors = SeperationVectors,
                settings = settings,
                DeltaTime = SystemAPI.Time.DeltaTime
            }.ScheduleParallel(boidQuery, calcJobHandle);

            state.Dependency = boidUpdateJobHandle;
        }

        [BurstCompile]
        public void OnDestroy(ref SystemState state)
        {

        }
    }

    // using a hashmap for now as an alternative a proper space-partitioning structure.
    // has the limitation at the moment of essentially ignoring everything outside the current cell
    [BurstCompile]
    public partial struct HashJob : IJobEntity
    {
        [WriteOnly] public NativeParallelMultiHashMap<int, int>.ParallelWriter CellHash;
        [WriteOnly] public NativeArray<int> CellIndices;

        public float InverseCellSize;

        [BurstCompile]
        public void Execute([EntityIndexInQuery] int indexInQuery, in Boid boid, in LocalTransform transform)
        {

            int hash = (int)math.hash(math.int3(math.floor(transform.Position * InverseCellSize)));

            CellIndices[indexInQuery] = hash;

            CellHash.Add(hash, indexInQuery);
        }
    }

    [BurstCompile]
    public partial struct DataJob : IJobEntity
    {
        [WriteOnly]
        public NativeArray<float3> Positions;
        [WriteOnly]
        public NativeArray<float3> Velocities;

        [BurstCompile]
        public void Execute([EntityIndexInQuery] int entityIndexInQuery, in Boid boid, in LocalTransform transform)
        {
            Positions[entityIndexInQuery] = transform.Position;
            Velocities[entityIndexInQuery] = boid.Velocity;
        }
    }

    public struct AvgJob : IJobFor
    {
        [ReadOnly] public NativeArray<int> CellIndices;
        [ReadOnly] public NativeParallelMultiHashMap<int, int> CellHash;

        [ReadOnly] public NativeArray<float3> Positions;
        [ReadOnly] public NativeArray<float3> Velocities;

        [WriteOnly] public NativeArray<float3> AvgPosition;
        [WriteOnly] public NativeArray<float3> AvgVelocity;

        public void Execute(int index)
        {
            var myCell = CellHash.GetValuesForKey(CellIndices[index]);


            float3 v = float3.zero;
            float3 p = float3.zero;
            int count = 0;
            while (myCell.MoveNext())
            {
                int i = myCell.Current;
                if (i == index) continue; // Skip self in average calculation
                v += Positions[i];
                p += Velocities[i];
                count++;
            }

            if (count <= 0)
            {
                AvgPosition[index] = 0;
                AvgVelocity[index] = 0;
            }
            else
            {
                v /= count;
                p /= count;

                AvgPosition[index] = p;
                AvgVelocity[index] = v;
            }
        }
    }

    public struct SeperationJob : IJobFor
    {
        [ReadOnly] public NativeArray<int> CellIndices;
        [ReadOnly] public NativeParallelMultiHashMap<int, int> CellHash;

        [ReadOnly] public NativeArray<float3> Positions;
        [WriteOnly] public NativeArray<float3> SeperationVectors;

        public float SqrSeperationDistance;
        public void Execute(int index)
        {
            var myCell = CellHash.GetValuesForKey(CellIndices[index]);
            float3 pos = Positions[index];

            float3 seperation = float3.zero;

            while (myCell.MoveNext())
            {
                int i = myCell.Current;
                if (i == index) continue;
                if (math.distancesq(pos, Positions[i]) < SqrSeperationDistance)
                    seperation += pos - Positions[i];
            }
            SeperationVectors[index] = seperation;
        }
    }

    [BurstCompile]
    public partial struct BoidUpdateJob : IJobEntity
    {
        [ReadOnly]
        public NativeArray<float3> AvgPositions;
        [ReadOnly]
        public NativeArray<float3> AvgVelocities;
        [ReadOnly]
        public NativeArray<float3> SeperationVectors;

        public Settings settings;

        public float DeltaTime;

        [BurstCompile]
        public void Execute([EntityIndexInQuery] int indexInQuery, ref Boid boid, ref LocalTransform transform)
        {
            var s = settings;

            float3 avgPos = AvgPositions[indexInQuery];
            float3 avgVel = AvgVelocities[indexInQuery];

            float3 cohesion = (avgPos - transform.Position) * s.CohesionWeight;
            float3 alignment = (avgVel - boid.Velocity) * s.AlignmentWeight;
            float3 seperation = SeperationVectors[indexInQuery] * s.SeperationWeight;
            float3 targetVec = (s.TargetPosition - avgPos) * s.TargetSeekWeight;

            boid.Velocity += (cohesion + alignment + seperation + targetVec) * DeltaTime;

            float sqrMag = math.lengthsq(boid.Velocity);

            if (sqrMag > s.SqrMaxSpeed)
            {
                boid.Velocity *= math.sqrt(s.SqrMaxSpeed / sqrMag);
            }

            transform.Position += boid.Velocity * DeltaTime;
        }
    }
}
