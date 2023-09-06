const shader_code =`
struct VSOut 
{
    @builtin(position) Position: vec4f,    
    @location(0) UV: vec2f
};


@vertex
fn vs_main(@builtin(vertex_index) vertId: u32) -> VSOut
{
    var vsOut: VSOut;
    let grid = vec2(f32((vertId<<1)&2), f32(vertId & 2));
    let pos_proj = grid * vec2(2.0, 2.0) + vec2(-1.0, -1.0);       
    let uv = vec2(grid.x, 1.0 - grid.y);
    vsOut.Position = vec4(pos_proj, 0.0, 1.0);
    vsOut.UV = uv;
    return vsOut;
}

@group(0) @binding(0)
var uTexDepth: texture_depth_2d;

@group(0) @binding(1)
var uSampler: sampler;

@fragment
fn fs_main(@location(0) UV: vec2f) -> @location(0) vec4f
{
    let d = textureSampleLevel(uTexDepth, uSampler, UV, 0);
    let v = clamp(d, 0.0, 1.0);
    let col = vec3(v);    
    return vec4(col, 1.0);
}
`;


function GetPipeline(view_format)
{
    if (!("visualize_height" in engine_ctx.cache.pipelines))
    {
        const pipelineLayoutDesc = { bindGroupLayouts: [ engine_ctx.cache.bindGroupLayouts.height_field] };
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

        engine_ctx.cache.pipelines.visualize_height = engine_ctx.device.createRenderPipeline(pipelineDesc);

    }

    return engine_ctx.cache.pipelines.visualize_height;

}


export function VisualizeHeight(passEncoder, bind_group, target)
{
    let pipeline = GetPipeline(target.view_format);
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bind_group);   
    passEncoder.draw(3, 1);
}




