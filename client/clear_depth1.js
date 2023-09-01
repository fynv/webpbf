const shader_code = `
@group(0) @binding(3)
var<storage, read_write> bDepth : array<f32>;

@compute @workgroup_size(64,1,1)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>)
{
    let idx = GlobalInvocationID.x;
    if (idx >= arrayLength(&bDepth)) 
    {
        return;
    }

    bDepth[idx] = 3.40282346638528859812e+38;
}   
`;


function GetPipeline()
{
    if (!("clear_depth1" in engine_ctx.cache.pipelines))
    {
        let shaderModule = engine_ctx.device.createShaderModule({ code: shader_code });
        let bindGroupLayouts = [engine_ctx.cache.bindGroupLayouts.raycast1];
        const pipelineLayoutDesc = { bindGroupLayouts };
        let layout = engine_ctx.device.createPipelineLayout(pipelineLayoutDesc);
        
        engine_ctx.cache.pipelines.clear_depth1 = engine_ctx.device.createComputePipeline({
            layout,
            compute: {
                module: shaderModule,
                entryPoint: 'main',
            },
        });
    }
    return engine_ctx.cache.pipelines.clear_depth1;
}


export function ClearDepth(commandEncoder, target)
{
    let pipeline = GetPipeline();

    let num_particles= target.psystem.numParticles;
    let num_groups =  Math.floor((num_particles +63)/64);
    let bind_group = target.bind_group_raycast;

    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bind_group);
    passEncoder.dispatchWorkgroups(num_groups, 1,1); 
    passEncoder.end();
}


