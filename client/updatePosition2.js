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

@group(0) @binding(1)
var<storage, read_write> bPos : array<vec4f>;

@group(0) @binding(2)
var<storage, read_write> bVel : array<vec4f>;

@group(0) @binding(3)
var<storage, read> bSortedPos : array<vec4f>;

@group(0) @binding(4)
var<storage, read> bSortedVel : array<vec4f>;

@group(0) @binding(5)
var<storage, read> bGridParticleIndex : array<u32>;

@group(0) @binding(6)
var<storage, read> bDepth : array<f32>;

@group(0) @binding(7)
var<storage, read> bNorm : array<u32>;

const tolerance = 0.001;

@compute @workgroup_size(${workgroup_size},1,1)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>)
{
    let idx = GlobalInvocationID.x;
    if (idx >= arrayLength(&bSortedPos)) 
    {
        return;
    }

    var pos = bSortedPos[idx].xyz;
    var vel = bSortedVel[idx].xyz;

    var world_move = vel* uParams.time_step;
    let dis = length(world_move);
    let dir = world_move/dis;

    if (dis>0.0)
    {
        let hit_dis = bDepth[idx];
        if (hit_dis < dis + tolerance)
        {          
            
            //world_move = vec3(0.0);
            world_move = max(hit_dis - tolerance, 0.0) * dir;

            //vel = vec3(0.0);
            let u32norm = bNorm[idx];
            let u8norm = vec3(u32norm & 0xffu, (u32norm>>8)& 0xffu, (u32norm>>16)& 0xffu);
            let norm = normalize((vec3f(u8norm)+0.5)/128.0 - 1.0);
            vel -= 1.94* dot(vel, norm) * norm;
        }
    }

    pos += world_move;

    if (pos.y < uParams.globalMin.y)
    {
        pos.y = uParams.globalMin.y;
        vel.y *= -0.94;
    }

    if (pos.x > uParams.globalMax.x)
    {
        pos.x = uParams.globalMax.x;
        vel.x *= -0.94;
    }

    if (pos.x < uParams.globalMin.x)
    {
        pos.x = uParams.globalMin.x;
        vel.x *= -0.94;
    }

    if (pos.y > uParams.globalMax.y)
    {
        pos.y = uParams.globalMax.y;
        vel.y *= -0.94;
    }   
   
    if (pos.z > uParams.globalMax.z)
    {
        pos.z = uParams.globalMax.z;
        vel.z *= -0.94;
    }

    if (pos.z < uParams.globalMin.z)
    {
        pos.z = uParams.globalMin.z;
        vel.z *= -0.94;
    }

    let originalIndex = bGridParticleIndex[idx]; 
    bPos[originalIndex] = vec4(pos, 1.0);
    bVel[originalIndex] = vec4(vel, 0.0);
}
`;

function GetPipeline()
{
    if (!("update_position2" in engine_ctx.cache.pipelines))
    {
        let shaderModule = engine_ctx.device.createShaderModule({ code: shader_code });
        let bindGroupLayouts = [engine_ctx.cache.bindGroupLayouts.update_position2];
        const pipelineLayoutDesc = { bindGroupLayouts };
        let layout = engine_ctx.device.createPipelineLayout(pipelineLayoutDesc);

        engine_ctx.cache.pipelines.update_position2 = engine_ctx.device.createComputePipeline({
            layout,
            compute: {
                module: shaderModule,
                entryPoint: 'main',
            },
        });
    }
    return engine_ctx.cache.pipelines.update_position2;
}

export function UpdatePosition2(commandEncoder, psystem)
{
    let pipeline = GetPipeline();

    let num_particles= psystem.numParticles;
    let num_groups =  Math.floor((num_particles + workgroup_size - 1)/workgroup_size);
    let bind_group = psystem.bind_group_update_position;

    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bind_group);
    passEncoder.dispatchWorkgroups(num_groups, 1,1); 
    passEncoder.end();
}
