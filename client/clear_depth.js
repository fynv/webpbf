const shader_code = `
@group(0) @binding(0)
var<storage, read_write> bDepth : array<f32>;

struct View
{
    width: i32,
    height: i32
};

@group(0) @binding(2)
var<uniform> uView: View;

@compute @workgroup_size(64,1,1)
fn main(
    @builtin(local_invocation_id) LocalInvocationID : vec3<u32>,
    @builtin(workgroup_id) WorkgroupID : vec3<u32>)
{
    let bw = (uView.width + 7)/8;
    let x = i32(LocalInvocationID.x)%8 + i32(WorkgroupID.x)%bw * 8;
    let y = i32(LocalInvocationID.x)/8 + i32(WorkgroupID.x)/bw * 8;

    if (x>=uView.width || y>uView.height)
    {
        return;
    }

    let idx = u32(x+y*uView.width); 

    bDepth[idx] = 3.40282346638528859812e+38;    
}
`;


function GetPipeline()
{
    if (!("clear_depth" in engine_ctx.cache.pipelines))
    {
        let shaderModule = engine_ctx.device.createShaderModule({ code: shader_code });
        let bindGroupLayouts = [engine_ctx.cache.bindGroupLayouts.raycast0];
        const pipelineLayoutDesc = { bindGroupLayouts };
        let layout = engine_ctx.device.createPipelineLayout(pipelineLayoutDesc);
        
        engine_ctx.cache.pipelines.clear_depth = engine_ctx.device.createComputePipeline({
            layout,
            compute: {
                module: shaderModule,
                entryPoint: 'main',
            },
        });
    }
    return engine_ctx.cache.pipelines.clear_depth;
}

export function ClearDepth(commandEncoder, target)
{
    let bw = (target.width + 7)/8;
    let bh = (target.height + 7)/8;

    let pipeline = GetPipeline();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);    
    passEncoder.setBindGroup(0, target.bind_group_raycast);
    passEncoder.dispatchWorkgroups(bw*bh, 1,1); 
    passEncoder.end();

}
