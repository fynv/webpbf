const workgroup_size = 64;

const shader_code = `
@group(0) @binding(0)
var<storage, read> bPos : array<vec4f>;

@group(0) @binding(1)
var<storage, read> bVel : array<vec4f>;

@group(0) @binding(2)
var<storage, read> bCellPrefixSum : array<u32>;

@group(0) @binding(3)
var<storage, read> bGridParticleHash : array<u32>;

@group(0) @binding(4)
var<storage, read> bGridParticleIndexInCell : array<u32>;

@group(0) @binding(5)
var<storage, read_write> bSortedPos : array<vec4f>;

@group(0) @binding(6)
var<storage, read_write> bSortedVel : array<vec4f>;

@group(0) @binding(7)
var<storage, read_write> bGridParticleIndex : array<u32>;

@compute @workgroup_size(${workgroup_size},1,1)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>)
{
    let idx = GlobalInvocationID.x;
    if (idx >= arrayLength(&bGridParticleHash)) 
    {
        return;
    }

    let hash = bGridParticleHash[idx];
    let id_in_cell = bGridParticleIndexInCell[idx];
    let offset = select(0, bCellPrefixSum[hash-1], hash>0);
    let new_idx = offset + id_in_cell;

    bSortedPos[new_idx] = bPos[idx];
    bSortedVel[new_idx] = bVel[idx];
    bGridParticleIndex[new_idx] = idx;
}
`;

function GetPipeline()
{
    if (!("scatter" in engine_ctx.cache.pipelines))
    {
        let shaderModule = engine_ctx.device.createShaderModule({ code: shader_code });
        let bindGroupLayouts = [engine_ctx.cache.bindGroupLayouts.scatter];
        const pipelineLayoutDesc = { bindGroupLayouts };
        let layout = engine_ctx.device.createPipelineLayout(pipelineLayoutDesc);

        engine_ctx.cache.pipelines.scatter = engine_ctx.device.createComputePipeline({
            layout,
            compute: {
                module: shaderModule,
                entryPoint: 'main',
            },
        });
    }

    return engine_ctx.cache.pipelines.scatter;
}

export function Scatter(commandEncoder, psystem)
{
    let pipeline = GetPipeline();

    let num_particles= psystem.numParticles;
    let num_groups = Math.floor((num_particles + workgroup_size - 1)/workgroup_size);
    let bind_group = psystem.bind_group_scatter;

    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bind_group);
    passEncoder.dispatchWorkgroups(num_groups, 1,1); 
    passEncoder.end();
}


