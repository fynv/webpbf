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
var<storage, read> bSortedPos : array<vec4f>;

@group(0) @binding(2)
var<storage, read> bCellPrefixSum : array<u32>;

@group(0) @binding(3)
var<storage, read_write> bSortedDensity : array<f32>;

const PI = 3.14159265359;

var<private> H2 : f32;
var<private> KDensityConst: f32;

fn init_consts()
{
    H2 = uParams.h * uParams.h;
    let h3 = H2 * uParams.h;
    KDensityConst = 315.0 / (64.0 * PI * h3 * h3 * h3);    
}

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


fn KernelDensity(distSq: f32) -> f32
{   
	if(distSq > H2)
    {
        return 0.0;
    }    
    let f = H2 - distSq;    
	return KDensityConst * f * f * f;
}

@compute @workgroup_size(${workgroup_size},1,1)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>)
{
    let idx = GlobalInvocationID.x;
    if (idx >= arrayLength(&bSortedPos)) 
    {
        return;
    }

    init_consts();

    var pos = bSortedPos[idx].xyz;
    let gridPos = calcGridPos(pos);

    var d = 0.0;
    for (var z=-1; z<=1; z++)
    {
        for (var y=-1; y<=1; y++)
        {
            for (var x=-1; x<=1; x++)
            {
                let neighbourPos = gridPos + vec3(x,y,z);

                if (neighbourPos.x<0 || neighbourPos.x >=  uParams.gridDiv.x 
                    || neighbourPos.y<0 || neighbourPos.y >=  uParams.gridDiv.y
                    || neighbourPos.z<0 || neighbourPos.z >=  uParams.gridDiv.z)
                {
                    continue;
                }

                let gridHash = calcGridHash(neighbourPos);
                let startIndex = select(0, bCellPrefixSum[gridHash-1], gridHash>0);
                let endIndex =  bCellPrefixSum[gridHash];

                for (var j = startIndex; j< endIndex; j++)
                {
                    let pos2 = bSortedPos[j].xyz;
                    let delta_pos = pos2 - pos;
                    let dis2 = dot(delta_pos, delta_pos);
                    d += KernelDensity(dis2);
                }                
            }
        }
    }

    d*= uParams.particleMass;

    bSortedDensity[idx] = d;
}
`;


function GetPipeline()
{
    if (!("update_density" in engine_ctx.cache.pipelines))
    {
        let shaderModule = engine_ctx.device.createShaderModule({ code: shader_code });
        let bindGroupLayouts = [engine_ctx.cache.bindGroupLayouts.update_density];
        const pipelineLayoutDesc = { bindGroupLayouts };
        let layout = engine_ctx.device.createPipelineLayout(pipelineLayoutDesc);

        engine_ctx.cache.pipelines.update_density = engine_ctx.device.createComputePipeline({
            layout,
            compute: {
                module: shaderModule,
                entryPoint: 'main',
            },
        });
    }
    return engine_ctx.cache.pipelines.update_density;
}

export function UpdateDensity(commandEncoder, psystem)
{
    let pipeline = GetPipeline();

    let num_particles= psystem.numParticles;
    let num_groups =  Math.floor((num_particles + workgroup_size - 1)/workgroup_size);
    let bind_group = psystem.bind_group_update_density;

    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bind_group);
    passEncoder.dispatchWorkgroups(num_groups, 1,1); 
    passEncoder.end();
}

