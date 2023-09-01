const shader_code =`
struct VSOut 
{
    @builtin(position) Position: vec4f,    
    @location(0) PosProj: vec2f
};

@vertex
fn vs_main(@builtin(vertex_index) vertId: u32) -> VSOut
{
    var vsOut: VSOut;
    let grid = vec2(f32((vertId<<1)&2), f32(vertId & 2));
    let pos_proj = grid * vec2(2.0, 2.0) + vec2(-1.0, -1.0);        
    vsOut.Position = vec4(pos_proj, 0.0, 1.0);
    vsOut.PosProj = pos_proj;
    return vsOut;
}

struct View
{
    width: i32,
    height: i32
};

@group(0) @binding(0)
var<uniform> uView: View;

@group(0) @binding(1)
var<storage, read> bDepth : array<f32>;

@fragment
fn fs_main(@location(0) PosProj: vec2f) -> @location(0) vec4f
{
    let uv = (PosProj + 1.0)*0.5;    
    let icoord2d = vec2i(uv * vec2(f32(uView.width), f32(uView.height)));
    let idx = icoord2d.x + icoord2d.y * uView.width;
    let d = bDepth[idx];
    let v = clamp((d-1.0)/5.0, 0.0, 1.0);
    let col = vec3(v);    
    return vec4(col, 1.0);
}
`;

function GetPipeline(view_format)
{
    if (!("visualize_depth" in engine_ctx.cache.pipelines))
    {
        const pipelineLayoutDesc = { bindGroupLayouts: [ engine_ctx.cache.bindGroupLayouts.depth_visualize] };
        let layout = engine_ctx.device.createPipelineLayout(pipelineLayoutDesc);
        let shaderModule = engine_ctx.device.createShaderModule({ code: shader_code });

        const vertex = {
            module: shaderModule,
            entryPoint: 'vs_main',
            buffers: []
        };

        const colorState = {
            format:  view_format,
            writeMask: GPUColorWrite.ALL
        };

        const fragment = {
            module: shaderModule,
            entryPoint: 'fs_main',
            targets: [colorState]
        };

        const primitive = {
            frontFace: 'cw',
            cullMode: 'none',
            topology: 'triangle-list'
        };

        const pipelineDesc = {
            layout,
    
            vertex,
            fragment,
    
            primitive
        };

        engine_ctx.cache.pipelines.visualize_depth = engine_ctx.device.createRenderPipeline(pipelineDesc);

    }

    return engine_ctx.cache.pipelines.visualize_depth;

}


export function VisualizeDepth(passEncoder, bind_group, target)
{
    let pipeline = GetPipeline(target.view_format);
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bind_group);   
    passEncoder.draw(3, 1);
}




