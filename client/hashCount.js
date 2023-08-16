const workgroup_size = 64;

const shader_code = `

struct Params
{
    globalMin: vec4f,
    globalMax: vec4f,
    gridMin: vec4f,
    gridMax: vec4f,
    gridDiv: vec4i,
    numParticles: u32,
    particleRadius: f32,
    numGridCells: u32,
    sizeGridBuf: u32,
    h: f32,
    particleMass: f32,
    time_step: f32,
    gas_const: f32,
    pg: f32,
    gravity: f32,
    pt: f32,
    pmin_sur_grad: f32
};

@group(0) @binding(0)
var<uniform> uParams : Params;

const worldOrigin = vec3(-1.0, -1.0, -1.0);
const particleRadius = 1.0/ 64.0;
const cellSize = vec3(particleRadius * 2.0);
const gridSize = vec3i(64);

@group(0) @binding(1)
var<storage, read> bPos : array<vec4f>;

@group(0) @binding(2)
var<storage, read_write> bGridParticleHash : array<u32>;

@group(0) @binding(3)
var<storage, read_write> bGridParticleIndexInCell : array<u32>;

@group(0) @binding(4)
var<storage, read_write> dCellCount : array<atomic<u32>>;

fn calcGridPos(p: vec3f) -> vec3i
{
    let cellSize = (uParams.gridMax.xyz - uParams.gridMin.xyz)/vec3f(uParams.gridDiv.xyz);
    return min(vec3i((p - uParams.gridMin.xyz)/cellSize), uParams.gridDiv.xyz - 1); 
}

fn calcGridHash(gridPos: vec3i) -> u32
{    
    let gp = (gridPos + uParams.gridDiv.xyz) %  uParams.gridDiv.xyz;
    return u32(gp.x + (gp.y +  gp.z * uParams.gridDiv.y)* uParams.gridDiv.x);
}


@compute @workgroup_size(${workgroup_size},1,1)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>)
{
    let idx = GlobalInvocationID.x;
    if (idx >= uParams.numParticles) 
    {
        return;
    }

    let pos = bPos[idx].xyz;
    let gridPos = calcGridPos(pos);
    let hash = calcGridHash(gridPos);

    bGridParticleHash[idx] = hash;
    bGridParticleIndexInCell[idx] = atomicAdd(&dCellCount[hash], 1);
}
`;

function GetPipeline()
{
    if (!("hash_count" in engine_ctx.cache.pipelines))
    {
        let shaderModule = engine_ctx.device.createShaderModule({ code: shader_code });
        let bindGroupLayouts = [engine_ctx.cache.bindGroupLayouts.hash_count];
        const pipelineLayoutDesc = { bindGroupLayouts };
        let layout = engine_ctx.device.createPipelineLayout(pipelineLayoutDesc);

        engine_ctx.cache.pipelines.hash_count = engine_ctx.device.createComputePipeline({
            layout,
            compute: {
                module: shaderModule,
                entryPoint: 'main',
            },
        });
    }

    return engine_ctx.cache.pipelines.hash_count;
}


export function HashCount(commandEncoder, psystem)
{
    let pipeline = GetPipeline();

    let num_particles= psystem.numParticles;
    let num_groups = Math.floor((num_particles + workgroup_size - 1)/workgroup_size);
    let bind_group = psystem.bind_group_hash_count;

    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bind_group);
    passEncoder.dispatchWorkgroups(num_groups, 1,1); 
    passEncoder.end();
}

