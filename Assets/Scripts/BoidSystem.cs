using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Collections;
using Unity.Entities;
using Unity.Entities.UniversalDelegates;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Transforms;
using UnityEditor.ShaderGraph.Internal;
using UnityEngine;
using UnityEngine.LightTransport;
using UnityEngine.Rendering;
using static UnityEditor.PlayerSettings;
using static UnityEngine.Rendering.HDROutputUtils;

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

            var Positions = CollectionHelper.CreateNativeArray<float3, RewindableAllocator>(boidCount, ref state.WorldUnmanaged.UpdateAllocator);
            var Velocities = CollectionHelper.CreateNativeArray<float3, RewindableAllocator>(boidCount, ref state.WorldUnmanaged.UpdateAllocator);
            var VelocityChanges = CollectionHelper.CreateNativeArray<float3, RewindableAllocator>(boidCount, ref state.WorldUnmanaged.UpdateAllocator);

            int offsetCount = 2 * settings.DetectionCellSize + 1;
            offsetCount *= offsetCount * offsetCount;
            offsetCount -= 1;

            var SearchOffsets = CollectionHelper.CreateNativeArray<float3, RewindableAllocator>(offsetCount, ref state.WorldUnmanaged.UpdateAllocator);
            int size = 0;
            for (int dx = -settings.DetectionCellSize; dx <= settings.DetectionCellSize; dx++)
                for (int dy = -settings.DetectionCellSize; dy <= settings.DetectionCellSize; dy++)
                    for (int dz = -settings.DetectionCellSize; dz <= settings.DetectionCellSize; dz++)
                    {
                        if (dx == 0 && dy == 0 && dz == 0) continue;
                        float3 offset = settings.CellSize * math.float3(dx, dy, dz);
                        SearchOffsets[size++] = offset;
                    }

            

            var CellHash = new NativeParallelMultiHashMap<int, int>(boidCount * (offsetCount + 1), state.WorldUnmanaged.UpdateAllocator.ToAllocator);
            var BoidHashes = CollectionHelper.CreateNativeArray<int, RewindableAllocator>(boidCount, ref state.WorldUnmanaged.UpdateAllocator);


            var hashJobHandle = new HashJob
            {
                CellHash = CellHash.AsParallelWriter(),
                BoidHashes = BoidHashes,
                InverseCellSize = 1 / settings.CellSize,
                SearchOffsets = SearchOffsets,
            }.ScheduleParallel(boidQuery, state.Dependency);

            var copyDataJobHandle = new CopyDataJob
            {
                Positions = Positions,
                Velocities = Velocities,
            }.ScheduleParallel(boidQuery, state.Dependency);

            var initHandle = JobHandle.CombineDependencies(hashJobHandle, copyDataJobHandle);

            var steerJobHandle = new SteerJob
            {
                BoidHashes = CellHash,
                CellIndices = BoidHashes,
                Positions = Positions,
                Velocities = Velocities,
                DeltaVs = VelocityChanges,
                Settings = settings,
            }.ScheduleParallel(boidCount, 32, initHandle);

            var boidUpdateJobHandle = new BoidUpdateJob
            {
                DeltaVs = VelocityChanges,
                Damping = settings.Damping,
                DeltaTime = SystemAPI.Time.DeltaTime
            }.ScheduleParallel(boidQuery, steerJobHandle);

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
        [WriteOnly] public NativeArray<int> BoidHashes;

        [ReadOnly] public NativeArray<float3> SearchOffsets;

        public float InverseCellSize;

        [BurstCompile]
        public void Execute([EntityIndexInQuery] int indexInQuery, in Boid boid, in LocalTransform transform)
        {

            int hash = GetHash(transform.Position, InverseCellSize);
            CellHash.Add(indexInQuery, hash);
            BoidHashes[indexInQuery] = hash;

            for (int i = 0; i < SearchOffsets.Length; i++)
            {
                CellHash.Add(GetHash(transform.Position + SearchOffsets[i], InverseCellSize), indexInQuery);
            }
        }

        [BurstCompile]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static int GetHash(in float3 pos, float inverseCellSize)
        {
            return (int)math.hash(math.int3(math.floor(pos * inverseCellSize)));
        }
    }

    [BurstCompile]
    public partial struct CopyDataJob : IJobEntity
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

    [BurstCompile]
    public struct SteerJob : IJobFor
    {
        [ReadOnly] public NativeArray<int> CellIndices;
        [ReadOnly] public NativeParallelMultiHashMap<int, int> BoidHashes;

        [ReadOnly] public NativeArray<float3> Positions;
        [ReadOnly] public NativeArray<float3> Velocities;

        [WriteOnly] public NativeArray<float3> DeltaVs;

        public Settings Settings;

        [BurstCompile]
        public void Execute(int index)
        {
            var myCell = BoidHashes.GetValuesForKey(CellIndices[index]);

            float3 averageV = float3.zero;
            float3 averagePos = float3.zero;
            float3 seperation = float3.zero;
            int neighbors = 0;

            float sqrSeperationDistance = Settings.SeperationDistance * Settings.SeperationDistance;
            float sqrDetectSize = (Settings.DetectionCellSize * Settings.CellSize);
            sqrDetectSize *= sqrDetectSize;

            while (myCell.MoveNext())
            {
                int i = myCell.Current;
                if (i == index) continue; // Skip self

                if (math.distancesq(Positions[i], Positions[index]) > sqrDetectSize) continue;

                averagePos += Positions[i];
                averageV += Velocities[i];


                if (math.distancesq(Positions[index], Positions[i]) < sqrSeperationDistance)
                    seperation += Positions[index] - Positions[i];

                neighbors++;
            }

            float3 velocityChange = float3.zero;

            if (neighbors > 0)
            {
                averageV /= neighbors;
                averagePos /= neighbors;

                velocityChange += (averageV - Velocities[index]) * Settings.AlignmentWeight;
                velocityChange += (averagePos - Positions[index]) * Settings.CohesionWeight;

                if (Settings.SeperationDistance > 0)
                    velocityChange += seperation * Settings.SeperationWeight / Settings.SeperationDistance;
            }

            float sqrSeekDistance = Settings.TargetSeekDistance * Settings.TargetSeekDistance;

            if (math.distancesq(Positions[index], Settings.TargetPosition) > sqrSeekDistance)
            {
                velocityChange += (Settings.TargetPosition - Positions[index]) * Settings.TargetSeekWeight;
            }

            DeltaVs[index] = velocityChange;    
        }

    }

    [BurstCompile]
    public partial struct BoidUpdateJob : IJobEntity
    {
        [ReadOnly]
        public NativeArray<float3> DeltaVs;

        public float Damping;

        public float DeltaTime;

        [BurstCompile]
        public void Execute([EntityIndexInQuery] int indexInQuery, ref Boid boid, ref LocalTransform transform)
        {
            boid.Velocity += DeltaVs[indexInQuery] * DeltaTime;

            boid.Velocity *= (1 - Damping * DeltaTime);


            /*
            float sqrSpeed = math.lengthsq(boid.Velocity);

            if (sqrSpeed > Damping)
            {
                boid.Velocity *= math.sqrt(Damping / sqrSpeed);
            }
            */

            transform.Position += boid.Velocity * DeltaTime;
        }
    }
}
