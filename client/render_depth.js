const shader_code =`
struct Camera
{
    projMat: mat4x4f, 
    viewMat: mat4x4f,
    invProjMat: mat4x4f,
    invViewMat: mat4x4f,
    eyePos: vec4f,
    scissor: vec4f
};


@group(0) @binding(0)
var<uniform> uCamera: Camera;

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

@group(1) @binding(0)
var<uniform> uParams : Params;

struct VSIn 
{
    @builtin(vertex_index) vertId: u32,
    @location(0) position : vec4f,
};

struct VSOut 
{
    @builtin(position) Position: vec4f,
    @location(0) uv: vec2f,        
    @location(1) pos_view: vec3f,        
};

const c_uv = array(
    vec2(0.0, 0.0),
    vec2(1.0, 0.0),
    vec2(0.0, 1.0),
    vec2(1.0, 1.0),
    vec2(0.0, 1.0),
    vec2(1.0, 0.0)
);

@vertex
fn vs_main(input: VSIn) -> VSOut
{    
    let uv = c_uv[input.vertId];       
    let center_view = uCamera.viewMat * input.position;
    let r =  uParams.particleRadius* 2.0 * 0.8;
    let d = r*2.0;
    let pos_view = center_view + vec4(d*(uv-0.5), 0.0, 0.0);
    var pos_proj = uCamera.projMat * pos_view;
    pos_proj.z = (pos_proj.z + pos_proj.w) * 0.5;

    var out: VSOut;
    out.Position = pos_proj;
    out.uv = uv;  
    out.pos_view = pos_view.xyz;

    return out;
}

struct FSIn
{
    @location(0) uv: vec2f,    
    @location(1) pos_view: vec3f,        
};

struct FSOut
{
    @builtin(frag_depth) depth: f32,
};

@fragment
fn fs_main(input: FSIn) -> FSOut
{
    var N: vec3f;
    N = vec3(input.uv * 2.0 -1.0, 0.0);
    let mag = length(N.xy);
    if (mag>1.0)
    {
        discard;
    }
    N.z = sqrt(1-mag*mag);

    var pos_view = input.pos_view;
    let r =  uParams.particleRadius* 2.0 * 0.8;
    pos_view.z += N.z * r;    
    let z = uCamera.projMat[2][2] * pos_view.z + uCamera.projMat[3][2];
    let w = uCamera.projMat[2][3] * pos_view.z + uCamera.projMat[3][3];

    var output: FSOut;
    output.depth = (z/w + 1.0)*0.5;

    return output;
}
`;


function GetPipeline()
{
    if (!("render_particle_depth" in engine_ctx.cache.pipelines))
    {
        let camera_options = { has_reflector: false };
        let camera_signature =  JSON.stringify(camera_options);
        let camera_layout = engine_ctx.cache.bindGroupLayouts.perspective_camera[camera_signature];    
    
        const pipelineLayoutDesc = { bindGroupLayouts: [camera_layout, engine_ctx.cache.bindGroupLayouts.particle_render] };
        let layout = engine_ctx.device.createPipelineLayout(pipelineLayoutDesc);
        let shaderModule = engine_ctx.device.createShaderModule({ code: shader_code });
    
        const depthStencil = {
            depthWriteEnabled: true,
            depthCompare: 'less-equal',
            format: 'depth32float'
        };
    
        let vertex_bufs = [
            {            
                arrayStride: 4*4,
                stepMode: 'instance',
                attributes: [
                  {                
                    shaderLocation: 0,
                    offset: 0,
                    format: 'float32x4',
                  },             
                ],
            }
        ];
    
        const vertex = {
            module: shaderModule,
            entryPoint: 'vs_main',
            buffers: vertex_bufs
        };

        const fragment = {
            module: shaderModule,
            entryPoint: 'fs_main',
            targets: []
        };
    
        const primitive = {
            frontFace: 'ccw',
            cullMode:  "none",
            topology: 'triangle-list'
        };
    
        const pipelineDesc = {
            layout,
    
            vertex,
            fragment,
    
            primitive,
            depthStencil
        };
    
        engine_ctx.cache.pipelines.render_particle_depth = engine_ctx.device.createRenderPipeline(pipelineDesc); 
    }

    return engine_ctx.cache.pipelines.render_particle_depth;

}


export function RenderDepth(passEncoder, camera, psystem)
{
    let pipeline = GetPipeline();

    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, camera.bind_group);
    passEncoder.setBindGroup(1, psystem.bind_group_render);

    passEncoder.setVertexBuffer(0, psystem.dPos[0]);

    passEncoder.draw(6, psystem.numParticles);
}


