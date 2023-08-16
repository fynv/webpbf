const workgroup_size = 64;
const workgroup_size_2x = workgroup_size*2;

const shader_code1 = `
@group(0) @binding(0)
var<storage, read> bPosIn : array<vec4f>;

@group(0) @binding(1)
var<storage, read_write> bPosOut : array<vec4f>;

var<workgroup> s_buf : array<vec4f, ${workgroup_size_2x}>;

@compute @workgroup_size(${workgroup_size},1,1)
fn main(
    @builtin(local_invocation_id) LocalInvocationID : vec3<u32>,
    @builtin(workgroup_id) WorkgroupID : vec3<u32>)
{
    let threadIdx = LocalInvocationID.x;
    let blockIdx = WorkgroupID.x;    
    let count = arrayLength(&bPosIn);

    var i = threadIdx + blockIdx*${workgroup_size_2x};
    if (i<count)
    {
        s_buf[threadIdx] = bPosIn[i];
    }

    i = threadIdx + ${workgroup_size} + blockIdx*${workgroup_size_2x};
    if (i<count)
    {
        s_buf[threadIdx + ${workgroup_size}] = bPosIn[i];
    }

    workgroupBarrier();

    var stride = ${workgroup_size}u;
    while(stride>0)
    {
        if (threadIdx<stride)
        {
            i = threadIdx + stride  + blockIdx*${workgroup_size_2x};
            if (i<count)
            {
                s_buf[threadIdx] = min(s_buf[threadIdx], s_buf[threadIdx + stride]);
            }
        }
        stride = stride >> 1;
        workgroupBarrier();        
    }

    if (threadIdx==0)
    {
        bPosOut[blockIdx] = s_buf[0];
    }
}
`;


const shader_code2 = `
@group(0) @binding(0)
var<storage, read> bPosIn : array<vec4f>;

@group(0) @binding(1)
var<storage, read_write> bPosOut : array<vec4f>;

var<workgroup> s_buf : array<vec4f, ${workgroup_size_2x}>;

@compute @workgroup_size(${workgroup_size},1,1)
fn main(
    @builtin(local_invocation_id) LocalInvocationID : vec3<u32>,
    @builtin(workgroup_id) WorkgroupID : vec3<u32>)
{
    let threadIdx = LocalInvocationID.x;
    let blockIdx = WorkgroupID.x;    
    let count = arrayLength(&bPosIn);

    var i = threadIdx + blockIdx*${workgroup_size_2x};
    if (i<count)
    {
        s_buf[threadIdx] = bPosIn[i];
    }

    i = threadIdx + ${workgroup_size} + blockIdx*${workgroup_size_2x};
    if (i<count)
    {
        s_buf[threadIdx + ${workgroup_size}] = bPosIn[i];
    }

    workgroupBarrier();

    var stride = ${workgroup_size}u;
    while(stride>0)
    {
        if (threadIdx<stride)
        {
            i = threadIdx + stride  + blockIdx*${workgroup_size_2x};
            if (i<count)
            {
                s_buf[threadIdx] = max(s_buf[threadIdx], s_buf[threadIdx + stride]);
            }
        }
        stride = stride >> 1;
        workgroupBarrier();        
    }

    if (threadIdx==0)
    {
        bPosOut[blockIdx] = s_buf[0];
    }
}
`;

function GetPipeline1()
{
    if (!("particle_reduction_1" in engine_ctx.cache.pipelines))
    {
        let shaderModule = engine_ctx.device.createShaderModule({ code: shader_code1 });
        let bindGroupLayouts = [engine_ctx.cache.bindGroupLayouts.particle_reduction];
        const pipelineLayoutDesc = { bindGroupLayouts };
        let layout = engine_ctx.device.createPipelineLayout(pipelineLayoutDesc);
        
        engine_ctx.cache.pipelines.particle_reduction_1 = engine_ctx.device.createComputePipeline({
            layout,
            compute: {
                module: shaderModule,
                entryPoint: 'main',
            },
        });
    }
    return engine_ctx.cache.pipelines.particle_reduction_1;
}

function GetPipeline2()
{
    if (!("particle_reduction_2" in engine_ctx.cache.pipelines))
    {
        let shaderModule = engine_ctx.device.createShaderModule({ code: shader_code2 });
        let bindGroupLayouts = [engine_ctx.cache.bindGroupLayouts.particle_reduction];
        const pipelineLayoutDesc = { bindGroupLayouts };
        let layout = engine_ctx.device.createPipelineLayout(pipelineLayoutDesc);
        
        engine_ctx.cache.pipelines.particle_reduction_2 = engine_ctx.device.createComputePipeline({
            layout,
            compute: {
                module: shaderModule,
                entryPoint: 'main',
            },
        });
    }
    return engine_ctx.cache.pipelines.particle_reduction_2;
}



export function ParticleReduction(commandEncoder, psystem)
{
    {
        let pipeline = GetPipeline1();
        const passEncoder = commandEncoder.beginComputePass();
        let count = psystem.numParticles;  
        for (let i=0; i<psystem.dPos.length-1; i++)
        {
            count = Math.floor((count + workgroup_size_2x - 1)/workgroup_size_2x);                    
            passEncoder.setPipeline(pipeline);
            passEncoder.setBindGroup(0, psystem.bind_group_particle_reduction[i]);
            passEncoder.dispatchWorkgroups(count, 1,1); 
        }
        passEncoder.end();
    }

    commandEncoder.copyBufferToBuffer(psystem.dPos[psystem.dPos.length-1], 0, psystem.dConstant, 8*4, 4 * 4);

    {
        let pipeline = GetPipeline2();
        const passEncoder = commandEncoder.beginComputePass();
        let count = psystem.numParticles;  
        for (let i=0; i<psystem.dPos.length-1; i++)
        {
            count = Math.floor((count + workgroup_size_2x - 1)/workgroup_size_2x);           
            passEncoder.setPipeline(pipeline);
            passEncoder.setBindGroup(0, psystem.bind_group_particle_reduction[i]);
            passEncoder.dispatchWorkgroups(count, 1,1); 
        }
        passEncoder.end();
    }

    commandEncoder.copyBufferToBuffer(psystem.dPos[psystem.dPos.length-1], 0, psystem.dConstant, 12*4, 4 * 4);

}


