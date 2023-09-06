const shader_code = `
var<private> g_id_io : u32;
var<private> g_origin: vec3f;
var<private> g_dir: vec3f;
var<private> g_tmin: f32;
var<private> g_tmax: f32;

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
var<storage, read> bSortedPos : array<vec4f>;

@group(0) @binding(2)
var<storage, read> bSortedVel: array<vec4f>;

@group(0) @binding(3)
var<storage, read_write> bDepth : array<f32>;

@group(0) @binding(4)
var<storage, read_write> bNorm : array<u32>;

fn render()
{
    let delta = -g_origin;
    let proj = dot(delta, g_dir);
    if (proj > 0.0)
    {
        let center_dis2 = dot(delta, delta) - proj*proj;
        if (center_dis2<1.0)
        {
            let back_move = sqrt(1.0 - center_dis2);
            if (proj > back_move)
            {
                let hit_dis = proj - back_move;
                if (hit_dis < g_tmax)
                {
                    bDepth[g_id_io] = hit_dis;
                    let world_norm = normalize(g_origin + hit_dis * g_dir);
                    let u8norm = vec3u((world_norm + 1.0) * 0.5 * 255.0);
                    bNorm[g_id_io] = u8norm.x + (u8norm.y << 8) + (u8norm.z << 16);
                }
            }
        }
    }
}

const tolerance = 0.05;

@compute @workgroup_size(64,1,1)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>)
{
    let idx = GlobalInvocationID.x;
    if (idx >= arrayLength(&bDepth)) 
    {
        return;
    }

    let world_origin = bSortedPos[idx].xyz;
    let world_move = bSortedVel[idx].xyz * uParams.time_step;
    let dis = length(world_move);
    if (dis==0.0)
    {
        return;
    }
    let world_dir = world_move/dis;

    g_id_io = idx;
    g_origin = world_origin;
    g_dir = world_dir;
    g_tmin = 0.0;
    g_tmax = dis + tolerance;

    render();
}
`;



function GetPipeline()
{
    if (!("raycast_collide_ball" in engine_ctx.cache.pipelines))
    {
        let shaderModule = engine_ctx.device.createShaderModule({ code: shader_code });
        let bindGroupLayouts = [engine_ctx.cache.bindGroupLayouts.raycast1];
        const pipelineLayoutDesc = { bindGroupLayouts };
        let layout = engine_ctx.device.createPipelineLayout(pipelineLayoutDesc);
        
        engine_ctx.cache.pipelines.raycast_collide_ball = engine_ctx.device.createComputePipeline({
            layout,
            compute: {
                module: shaderModule,
                entryPoint: 'main',
            },
        });
    }
    return engine_ctx.cache.pipelines.raycast_collide_ball;
}



export function RaycastCollideBall(commandEncoder, target)
{
    let pipeline = GetPipeline();

    let num_particles= target.psystem.numParticles;
    let num_groups =  Math.floor((num_particles +63)/64);    

    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);   
    passEncoder.setBindGroup(0, target.bind_group_raycast);
    passEncoder.dispatchWorkgroups(num_groups, 1,1); 
    passEncoder.end();
}


