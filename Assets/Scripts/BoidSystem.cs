using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Collections;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Transforms;

namespace Boids
{
    [UpdateInGroup(typeof(FixedStepSimulationSystemGroup))]
    partial struct BoidSystem : ISystem
    {
        EntityQuery boidQuery;

        NativeArray<float3> searchOffsets;
        bool builtSearchOffsets;

        [BurstCompile]
        public void OnCreate(ref SystemState state)
        {
            state.RequireForUpdate<Settings>();

            boidQuery = new EntityQueryBuilder(Allocator.Temp)
                .WithAll<Boid, LocalToWorld>()
                .Build(ref state);

        }

        [BurstCompile]
        public void OnUpdate(ref SystemState state)
        {
            Settings settings = SystemAPI.GetSingleton<Settings>();

            int boidCount = boidQuery.CalculateEntityCount();

            if (!builtSearchOffsets)
            {
                BuildSearchOffsets(in settings, out searchOffsets);
                builtSearchOffsets = true;
            }

            var CellHash = new NativeParallelMultiHashMap<int, int>(boidCount * (searchOffsets.Length + 1), state.WorldUnmanaged.UpdateAllocator.ToAllocator);
            var BoidHashes = CollectionHelper.CreateNativeArray<int, RewindableAllocator>(boidCount, ref state.WorldUnmanaged.UpdateAllocator);

            var Positions = CollectionHelper.CreateNativeArray<float3, RewindableAllocator>(boidCount, ref state.WorldUnmanaged.UpdateAllocator);
            var Headings = CollectionHelper.CreateNativeArray<float3, RewindableAllocator>(boidCount, ref state.WorldUnmanaged.UpdateAllocator);
            var NewHeadings = CollectionHelper.CreateNativeArray<float3, RewindableAllocator>(boidCount, ref state.WorldUnmanaged.UpdateAllocator);


            var hashJobHandle = new HashJob
            {
                CellHash = CellHash.AsParallelWriter(),
                BoidHashes = BoidHashes,
                InverseCellSize = 1 / settings.CellSize,
                SearchOffsets = searchOffsets,
            }.ScheduleParallel(boidQuery, state.Dependency);

            var copyDataJobHandle = new CopyDataJob
            {
                Positions = Positions,
                Headings = Headings,
            }.ScheduleParallel(boidQuery, state.Dependency);

            var initHandle = JobHandle.CombineDependencies(hashJobHandle, copyDataJobHandle);

            var steerJobHandle = new SteerJob
            {
                BoidHashes = CellHash,
                CellIndices = BoidHashes,
                Positions = Positions,
                Headings = Headings,
                NewHeadings = NewHeadings,
                Settings = settings,
            }.ScheduleParallel(boidCount, 32, initHandle);

            var boidUpdateJobHandle = new BoidUpdateJob
            {
                NewHeadings = NewHeadings,
                TurnSpeed = settings.turnSpeed,
                MoveSpeed = settings.MoveSpeed,
                DeltaTime = SystemAPI.Time.DeltaTime
            }.ScheduleParallel(boidQuery, steerJobHandle);

            state.Dependency = boidUpdateJobHandle;
        }

        [BurstCompile]
        public static void BuildSearchOffsets(in Settings settings, out NativeArray<float3> searchOffsets)
        {
            int offsetCount = 2 * settings.DetectionCellSize + 1;
            offsetCount *= offsetCount * offsetCount;
            offsetCount -= 1;

            searchOffsets = new NativeArray<float3>(offsetCount, Allocator.Persistent);

            int i = 0;
            for (int x = -settings.DetectionCellSize; x <= settings.DetectionCellSize; x++)
                for (int y = -settings.DetectionCellSize; y <= settings.DetectionCellSize; y++)
                    for (int z = -settings.DetectionCellSize; z <= settings.DetectionCellSize; z++)
                    {
                        if (x == 0 && y == 0 && z == 0) continue;
                        float3 offset = settings.CellSize * math.float3(x, y, z);
                        searchOffsets[i++] = offset;
                    }
            //var SearchOffsets = CollectionHelper.CreateNativeArray<float3, RewindableAllocator>(offsetCount, ref state.WorldUnmanaged.UpdateAllocator);
        }

        [BurstCompile]
        public void OnDestroy(ref SystemState state)
        {
            searchOffsets.Dispose();
        }
    }


    [BurstCompile]
    public partial struct HashJob : IJobEntity
    {
        [WriteOnly] public NativeParallelMultiHashMap<int, int>.ParallelWriter CellHash;
        [WriteOnly] public NativeArray<int> BoidHashes;

        [ReadOnly] public NativeArray<float3> SearchOffsets;

        public float InverseCellSize;

        [BurstCompile]
        public void Execute([EntityIndexInQuery] int indexInQuery, in Boid boid, in LocalToWorld transform)
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
        public NativeArray<float3> Headings;

        [BurstCompile]
        public void Execute([EntityIndexInQuery] int entityIndexInQuery, in Boid boid, in LocalToWorld transform)
        {
            Positions[entityIndexInQuery] = transform.Position;
            Headings[entityIndexInQuery] = transform.Forward;
        }
    }

    [BurstCompile]
    public struct SteerJob : IJobFor
    {
        [ReadOnly] public NativeArray<int> CellIndices;
        [ReadOnly] public NativeParallelMultiHashMap<int, int> BoidHashes;

        [ReadOnly] public NativeArray<float3> Positions;
        [ReadOnly] public NativeArray<float3> Headings;

        [WriteOnly] public NativeArray<float3> NewHeadings;

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
                averageV += Headings[i];


                if (math.distancesq(Positions[index], Positions[i]) < sqrSeperationDistance)
                    seperation += Positions[index] - Positions[i];

                neighbors++;
            }

            float3 newHeadings = float3.zero;

            if (neighbors > 0)
            {
                averageV /= neighbors;
                averagePos /= neighbors;

                newHeadings += (averageV - Headings[index]) * Settings.AlignmentWeight;
                newHeadings += (averagePos - Positions[index]) * Settings.CohesionWeight;

                if (Settings.SeperationDistance > 0)
                    newHeadings += seperation * Settings.SeperationWeight / Settings.SeperationDistance;
            }

            float sqrSeekDistance = Settings.TargetSeekDistance * Settings.TargetSeekDistance;

            if (math.distancesq(Positions[index], Settings.TargetPosition) > sqrSeekDistance)
            {
                newHeadings += (Settings.TargetPosition - Positions[index]) * Settings.TargetSeekWeight;
            }

            NewHeadings[index] = newHeadings;    
        }

    }

    [BurstCompile]
    public partial struct BoidUpdateJob : IJobEntity
    {
        [ReadOnly]
        public NativeArray<float3> NewHeadings;

        public float TurnSpeed;
        public float MoveSpeed;
        public float DeltaTime;

        [BurstCompile]
        public void Execute([EntityIndexInQuery] int indexInQuery, in Boid boid, ref LocalToWorld transform)
        {
            float3 target = math.normalizesafe(NewHeadings[indexInQuery]) * TurnSpeed;
            float3 newHeading = math.normalizesafe(transform.Forward + (target - transform.Forward) * DeltaTime);

            transform = new LocalToWorld
            {
                Value = float4x4.TRS(

                        new float3(transform.Position + (newHeading * MoveSpeed * DeltaTime)),
                        quaternion.LookRotationSafe(newHeading, math.up()),
                        transform.Value.Scale())
            };
        }
    }
}
