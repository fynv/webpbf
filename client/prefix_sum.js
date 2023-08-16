const workgroup_size = 64;
const workgroup_size_2x = workgroup_size*2;

function condition(cond, a, b="")
{
    return cond? a: b;
}

function get_shader1(has_group_buf)
{
    return  `
@group(0) @binding(0)
var<storage, read_write> bData : array<u32>;    

${condition(has_group_buf,`
@group(0) @binding(1)
var<storage, read_write> bGroup : array<u32>;
`)}

var<workgroup> s_buf : array<u32, ${workgroup_size_2x}>;

@compute @workgroup_size(${workgroup_size},1,1)
fn main(
    @builtin(local_invocation_id) LocalInvocationID : vec3<u32>,
    @builtin(workgroup_id) WorkgroupID : vec3<u32>)
{
    let threadIdx = LocalInvocationID.x;
    let blockIdx = WorkgroupID.x;    
    let count = arrayLength(&bData);

    var i = threadIdx + blockIdx*${workgroup_size_2x};
    if (i<count)
    {
        s_buf[threadIdx] = bData[i];
    }

    i = threadIdx + ${workgroup_size} + blockIdx*${workgroup_size_2x};
    if (i<count)
    {
        s_buf[threadIdx + ${workgroup_size}] = bData[i];
    }

    workgroupBarrier();

    var half_size_group = 1u;
    var size_group = 2u;

    while(half_size_group <= ${workgroup_size})
    {
        let gid = threadIdx/half_size_group;
        let tid = gid*size_group + half_size_group + threadIdx % half_size_group;
        i = tid + blockIdx*${workgroup_size_2x};
        if (i<count)
        {
            s_buf[tid] = s_buf[gid*size_group + half_size_group -1] + s_buf[tid];
        }
        half_size_group = half_size_group << 1;
        size_group = size_group << 1;
        workgroupBarrier();
    }

    i = threadIdx + blockIdx*${workgroup_size_2x};
    if (i<count)
    {
        bData[i] = s_buf[threadIdx];
    }
    
    i = threadIdx + ${workgroup_size} + blockIdx*${workgroup_size_2x};
    if (i<count)
    {
        bData[i] = s_buf[threadIdx + ${workgroup_size}];
    }


${condition(has_group_buf,`        
    let count_group = arrayLength(&bGroup);
    if (threadIdx == 0 && blockIdx<count_group)
    {        
        bGroup[blockIdx] = s_buf[${workgroup_size_2x} - 1];
    }
`)}
}
`;
}

function GetPipeline1(has_group_buf)
{
    if (!("prefix_sum_1" in engine_ctx.cache.pipelines))
    {
        engine_ctx.cache.pipelines.prefix_sum_1 = {};
    }

    if (!(has_group_buf in engine_ctx.cache.pipelines.prefix_sum_1))
    {   
        let shaderModule = engine_ctx.device.createShaderModule({ code: get_shader1(has_group_buf) });
        let bindGroupLayouts = [has_group_buf ? engine_ctx.cache.bindGroupLayouts.prefix_sum_1b : engine_ctx.cache.bindGroupLayouts.prefix_sum_1a];
        const pipelineLayoutDesc = { bindGroupLayouts };
        let layout = engine_ctx.device.createPipelineLayout(pipelineLayoutDesc);
        
        engine_ctx.cache.pipelines.prefix_sum_1[has_group_buf] = engine_ctx.device.createComputePipeline({
            layout,
            compute: {
                module: shaderModule,
                entryPoint: 'main',
            },
        });
    }

    return engine_ctx.cache.pipelines.prefix_sum_1[has_group_buf];
}


function get_shader2()
{
    return  ` 
@group(0) @binding(0)
var<storage, read_write> bData : array<i32>;    

@group(0) @binding(1)
var<storage, read> bGroup : array<i32>;

@compute @workgroup_size(${workgroup_size},1,1)
fn main(
    @builtin(local_invocation_id) LocalInvocationID : vec3<u32>,
    @builtin(workgroup_id) WorkgroupID : vec3<u32>)
{
    let threadIdx = LocalInvocationID.x;
    let blockIdx = WorkgroupID.x + 2;    
    let count = arrayLength(&bData);

    let add_idx = WorkgroupID.x / 2;
    let i = threadIdx + blockIdx*${workgroup_size};
    if (i<count)
    {
        let value = bData[i];
        bData[i] = value + bGroup[add_idx];
    }
}
`;
}

function GetPipeline2()
{
    if (!("prefix_sum_2" in engine_ctx.cache.pipelines))
    {
        let shaderModule = engine_ctx.device.createShaderModule({ code: get_shader2() });
        let bindGroupLayouts = [engine_ctx.cache.bindGroupLayouts.prefix_sum_2];
        const pipelineLayoutDesc = { bindGroupLayouts };
        let layout = engine_ctx.device.createPipelineLayout(pipelineLayoutDesc);

        engine_ctx.cache.pipelines.prefix_sum_2 = engine_ctx.device.createComputePipeline({
            layout,
            compute: {
                module: shaderModule,
                entryPoint: 'main',
            },
        });
    }

    return engine_ctx.cache.pipelines.prefix_sum_2;
}

export function PrefixSum(commandEncoder, psystem)
{
    const passEncoder = commandEncoder.beginComputePass();

    for (let i=0; i<psystem.dCellCountBufs.length; i++)
    {
        if (i<psystem.dCellCountBufs.length - 1)
        {
            let pipeline = GetPipeline1(true);           
            passEncoder.setPipeline(pipeline);
            passEncoder.setBindGroup(0, psystem.bind_group_prefix_sum1[i]);
            passEncoder.dispatchWorkgroupsIndirect(psystem.dConstant, (32+i*4)*4); 
        }
        else if (psystem.dCellCountBufSizes[i] > 1)
        {
            let pipeline = GetPipeline1(false);            
            passEncoder.setPipeline(pipeline);
            passEncoder.setBindGroup(0, psystem.bind_group_prefix_sum1[i]);
            passEncoder.dispatchWorkgroupsIndirect(psystem.dConstant, (32+i*4)*4); 
        }        
    }  

    for (let i = psystem.dCellCountBufs.length-2; i>=0; i--)
    {
        let pipeline = GetPipeline2();        
        passEncoder.setPipeline(pipeline);
        passEncoder.setBindGroup(0, psystem.bind_group_prefix_sum2[i]);
        passEncoder.dispatchWorkgroupsIndirect(psystem.dConstant, (44+i*4)*4);
    }

    passEncoder.end();

}


