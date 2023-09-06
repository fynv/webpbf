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
var<storage, read> bSortedDensity : array<f32>;

@group(0) @binding(6)
var<storage, read> bGridParticleIndex : array<u32>;

@group(0) @binding(7)
var<storage, read> bCellPrefixSum : array<u32>;

@group(1) @binding(0)
var uTexDepth: texture_depth_2d;

@group(1) @binding(1)
var uSampler: sampler;

const PI = 3.14159265359;
const PREST_D = 1000.0;

var<private> H2 : f32;
var<private> CSelf: f32;
var<private> KPressureConst: f32;
var<private> KViscosityConst: f32;
var<private> KSurfaceTensionConst: f32;

fn init_consts()
{
    H2 = uParams.h * uParams.h;
    let h3 = H2 * uParams.h;        
    KPressureConst = -45.0 / (PI*h3*h3);
    KViscosityConst =  45.0 / (PI*h3*h3);
    KSurfaceTensionConst = - 945.0 / (32.0 * PI *h3*h3*h3);
    CSelf = -945.0 * 3.0 / (32.0*PI*H2*h3);
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


fn KernelPressureGrad(dist: f32) -> f32
{
	if (dist > uParams.h)
    {
		return 0;
    }
    
	if(dist!=0)
    {
        return KPressureConst / dist * (uParams.h-dist)*(uParams.h-dist);
    }	
    else
    {
		return KPressureConst / 100000.0 * (uParams.h-dist)*(uParams.h-dist);
    }
}

fn KernelViscosityLaplacian(dist: f32) -> f32
{
    if (dist > uParams.h)
    {
        return 0;
    }
    
	return KViscosityConst * (uParams.h - dist);
}

struct Tension
{
    lap: f32,
    grad: f32
};


fn KernelSurfaceTension(distSq: f32) -> Tension
{
    var ret : Tension;

	if(distSq > H2)
    {
        ret.grad = 0.0;
        ret.lap  = 0.0;
        return ret;
	}
    
	let r2mh2 = distSq - H2;
	let tmp = KSurfaceTensionConst * r2mh2;
	ret.grad = tmp * r2mh2;
	ret.lap = tmp * (7.0*distSq - 3.0*H2);
    return ret;
}

fn fetch_pos(xz: vec2f) -> vec3f
{
    let uv = vec2(xz - uParams.globalMin.xz)/(uParams.globalMax.xz - uParams.globalMin.xz);
    let depth = textureSampleLevel(uTexDepth, uSampler, uv, 0);
    let h = depth *(uParams.globalMax.y - uParams.globalMin.y) + uParams.globalMin.y;
    return vec3(xz.x, h, xz.y);
}

fn MinDiff(P: vec3f, Pr: vec3f, Pl: vec3f) -> vec3f
{
    let V1 = Pr - P;
    let V2 = P - Pl;
    return select(V2, V1, dot(V1,V1) < dot(V2,V2));
}

fn ReconstructNormal(xz: vec2f, P: vec3f) -> vec3f
{
    let r = uParams.particleRadius;
    var Pr = fetch_pos(xz + vec2(r, 0.0));
    var Pl = fetch_pos(xz + vec2(-r, 0.0));
    var Pt = fetch_pos(xz + vec2(0.0, -r));
    var Pb = fetch_pos(xz + vec2(0.0, r));
    return normalize(cross(MinDiff(P, Pr, Pl), MinDiff(P, Pt, Pb)));
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
    var vel = bSortedVel[idx].xyz;

    let d = bSortedDensity[idx];
    let v = 1.0 / d;
    let p = max(0.0, uParams.gas_const * (d - PREST_D));

    var c_lap = CSelf * v;
    var n = vec3(0.0);

    let gridPos = calcGridPos(pos);

    var fpressure = vec3(0.0);
    var fviscosity = vec3(0.0);
    
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
                var startIndex = 0u;
                if (gridHash>0)
                {
                    startIndex = bCellPrefixSum[gridHash-1];
                }
                let endIndex =  bCellPrefixSum[gridHash];

                for (var j = startIndex; j< endIndex; j++)
                {
                    let pos2 = bSortedPos[j].xyz;
                    let vel2 = bSortedVel[j].xyz;
                    let d2 = bSortedDensity[j];
                    let v2 = 1.0 / d2;
                    let p2 =  max(0.0, uParams.gas_const * (d2 - PREST_D));
                    
                    let delta_pos = pos - pos2;
                    let dist = length(delta_pos);
                    let grad = (p + p2)/2.0 * KernelPressureGrad(dist);
                    fpressure -= delta_pos * grad * v2;

                    let dv = vel2 - vel; 
                    fviscosity += dv * v2 * KernelViscosityLaplacian(dist);

                    let tension = KernelSurfaceTension(dist*dist);
                    n += delta_pos * (tension.grad * v2);
                    c_lap += tension.lap * v2;
                }
            }
        }
    }
    
    fviscosity *= uParams.pg * uParams.particleMass;		
    fpressure *= uParams.particleMass;

    var acc = (fviscosity + fpressure) * v;
    n *= uParams.particleMass;
    c_lap *= uParams.particleMass;

    let nl = length(n);
    if (nl > uParams.pmin_sur_grad)
    {
        let fsur = n * (uParams.pt * c_lap / nl);
        acc -= fsur * v;
    }
    acc += uParams.gravity * vec3(0.0, -1.0, 0.0);

    vel += acc * uParams.time_step;
    let pos_next = pos + vel* uParams.time_step;

    let ref_next = fetch_pos(pos_next.xz);
    if (pos_next.y < ref_next.y)
    {
        let norm = ReconstructNormal(pos_next.xz, ref_next);
        let proj = dot(vel, norm) * norm;        
        vel += -1.94 * proj;
    }
    else
    {
        pos = pos_next;
    }

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
    if (!("update_position" in engine_ctx.cache.pipelines))
    {
        let shaderModule = engine_ctx.device.createShaderModule({ code: shader_code });
        let bindGroupLayouts = [engine_ctx.cache.bindGroupLayouts.update_position, engine_ctx.cache.bindGroupLayouts.height_field];
        const pipelineLayoutDesc = { bindGroupLayouts };
        let layout = engine_ctx.device.createPipelineLayout(pipelineLayoutDesc);

        engine_ctx.cache.pipelines.update_position = engine_ctx.device.createComputePipeline({
            layout,
            compute: {
                module: shaderModule,
                entryPoint: 'main',
            },
        });
    }
    return engine_ctx.cache.pipelines.update_position;
}

export function UpdatePosition(commandEncoder, psystem, height_target)
{
    let pipeline = GetPipeline();

    let num_particles= psystem.numParticles;
    let num_groups =  Math.floor((num_particles + workgroup_size - 1)/workgroup_size);    

    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, psystem.bind_group_update_position);
    passEncoder.setBindGroup(1, height_target.bind_group);
    passEncoder.dispatchWorkgroups(num_groups, 1,1); 
    passEncoder.end();
}

