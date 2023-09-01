const workgroup_size = 64;

const shader_code = `
@group(0) @binding(0)
var<storage, read_write> bCellCount : array<u32>;

@compute @workgroup_size(${workgroup_size},1,1)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>)
{
    let idx = GlobalInvocationID.x;
    if (idx >= arrayLength(&bCellCount)) 
    {
        return;
    }

    bCellCount[idx] = 0;
}
`;

function GetPipeline()
{
    if (!("clear_count" in engine_ctx.cache.pipelines))
    {
        let shaderModule = engine_ctx.device.createShaderModule({ code: shader_code });
        let bindGroupLayouts = [engine_ctx.cache.bindGroupLayouts.clear_count];
        const pipelineLayoutDesc = { bindGroupLayouts };
        let layout = engine_ctx.device.createPipelineLayout(pipelineLayoutDesc);

        engine_ctx.cache.pipelines.clear_count = engine_ctx.device.createComputePipeline({
            layout,
            compute: {
                module: shaderModule,
                entryPoint: 'main',
            },
        });
    }

    return engine_ctx.cache.pipelines.clear_count;
}

export function ClearCellCount(commandEncoder, psystem)
{
    let pipeline = GetPipeline();

    let num_cells= psystem.sizeGridBuf;
    let num_groups = Math.floor((num_cells + workgroup_size - 1)/workgroup_size);
    let bind_group = psystem.bind_group_clear_count;

    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bind_group);
    passEncoder.dispatchWorkgroups(num_groups, 1,1); 
    passEncoder.end();
}

