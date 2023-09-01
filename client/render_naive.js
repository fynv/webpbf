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

const cParticleRadius = 1.0 / 64.0;

struct VSIn 
{
    @builtin(vertex_index) vertId: u32,
    @location(0) position : vec4f,
    @location(1) color : vec4f,
};

struct VSOut 
{
    @builtin(position) Position: vec4f,
    @location(0) uv: vec2f,    
    @location(1) color : vec3f
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

    let d = cParticleRadius*2.0;
    let pos_view = center_view + vec4(d*(uv-0.5), 0.0, 0.0);
    var pos_proj = uCamera.projMat * pos_view;
    pos_proj.z = (pos_proj.z + pos_proj.w) * 0.5;

    var out: VSOut;
    out.Position = pos_proj;
    out.uv = uv;
    out.color = input.color.xyz;

    return out;
}

struct FSIn
{
    @location(0) uv: vec2f,
    @location(1) color : vec3f
};

@fragment
fn fs_main(input: FSIn) -> @location(0) vec4f
{
    let lightDir = vec3(0.577, 0.577, 0.577);

    var N: vec3f;
    N = vec3(input.uv * 2.0 -1.0, 0.0);
    let mag = length(N.xy);
    if (mag>1.0)
    {
        discard;
    }
    N.z = sqrt(1-mag*mag);
    let diffuse = max(0.0, dot(lightDir, N));
    return vec4(input.color * diffuse, 1.0);
}
`;

function GetPipeline(view_format, msaa)
{
    if (!("render_naive" in engine_ctx.cache.pipelines))
    {
        let camera_options = { has_reflector: false };
        let camera_signature =  JSON.stringify(camera_options);
        let camera_layout = engine_ctx.cache.bindGroupLayouts.perspective_camera[camera_signature];
    
    
        const pipelineLayoutDesc = { bindGroupLayouts: [camera_layout] };
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
            },
            {            
                arrayStride: 4*4,
                stepMode: 'instance',
                attributes: [
                  {                
                    shaderLocation: 1,
                    offset: 0,
                    format: 'float32x4',
                  },             
                ],
            },
        ];
    
        const vertex = {
            module: shaderModule,
            entryPoint: 'vs_main',
            buffers: vertex_bufs
        };
    
        const colorState = {
            format: view_format,        
            writeMask: GPUColorWrite.ALL
        };
    
        const fragment = {
            module: shaderModule,
            entryPoint: 'fs_main',
            targets: [colorState]
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

        if (msaa)
        {
            pipelineDesc.multisample ={
                count: 4,
            };
        }
    
        engine_ctx.cache.pipelines.render_naive = engine_ctx.device.createRenderPipeline(pipelineDesc); 
    }

    return engine_ctx.cache.pipelines.render_naive;

}


const ncolors = 7;
const c = [
    [ 1.0, 0.0, 0.0, ],
    [ 1.0, 0.5, 0.0, ],
    [ 1.0, 1.0, 0.0, ],
    [ 0.0, 1.0, 0.0, ],
    [ 0.0, 1.0, 1.0, ],
    [ 0.0, 0.0, 1.0, ],
    [ 1.0, 0.0, 1.0, ]
];

function create_colormap(psystem)
{
    let hColor = new Float32Array(psystem.numParticles * 4);
    for (let i=0; i<psystem.numParticles; i++) 
    {
        let t = i/psystem.numParticles;
        t *= (ncolors - 1);
        let j = Math.floor(t);
        let u =  t - j;
        hColor[i*4] = (1.0 - u) * c[j][0] + u * c[j+1][0];
        hColor[i*4 + 1] = (1.0 - u) * c[j][1] + u * c[j+1][1];
        hColor[i*4 + 2] = (1.0 - u) * c[j][2] + u * c[j+1][2];
        hColor[i*4 + 3] = 1.0;
    }

    psystem.dColor = engine_ctx.createBuffer(hColor.buffer, GPUBufferUsage.VERTEX);

}

export function RenderNaive(passEncoder, camera, psystem, target)
{
    if (!("dColor" in psystem))
    {        
        create_colormap(psystem);
    }

    let pipeline = GetPipeline(target.view_format, target.msaa);

    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, camera.bind_group);

    passEncoder.setVertexBuffer(0, psystem.dPos[0]);
    passEncoder.setVertexBuffer(1, psystem.dColor);

    passEncoder.draw(6, psystem.numParticles);
}

