import { EngineContext } from "./engine/EngineContext.js"
import { CanvasContext } from "./engine/CanvasContext.js"
import { GPURenderTarget } from "./engine/renderers/GPURenderTarget.js"
import { PerspectiveCameraEx } from "./engine/cameras/PerspectiveCameraEx.js"
import { OrbitControls } from "./engine/controls/OrbitControls.js"
import { Color } from "./engine/math/Color.js"

import { ParticleSystem } from "./particleSystem.js"

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
    N.z = sqrt(1-mag);
    let diffuse = max(0.0, dot(lightDir, N));
    return vec4(input.color * diffuse, 1.0);
}
`;


function GetPipelineRender(view_format)
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
        layout: layout,

        vertex,
        fragment,

        primitive,
        depthStencil
    };

    return engine_ctx.device.createRenderPipeline(pipelineDesc); 

}


export async function test()
{
    const canvas = document.getElementById('gfx');
    canvas.style.cssText = "position:absolute; width: 100%; height: 100%;";      

    const engine_ctx = new EngineContext();
    const canvas_ctx = new CanvasContext(canvas);
    await canvas_ctx.initialize();

    let resized = false;
    const size_changed = ()=>{
        canvas.width = canvas.clientWidth;
        canvas.height = canvas.clientHeight;        
        resized = true;
    };
    
    let observer = new ResizeObserver(size_changed);
    observer.observe(canvas);

    let msaa = false;
    let render_target = new GPURenderTarget(canvas_ctx, msaa);    

    let psystem = new ParticleSystem();   
   
    let camera = new PerspectiveCameraEx();
    camera.position.set(0, 1, 5); 

    let controls = new OrbitControls(camera, canvas);    
    controls.target.set(0,0,0); 
    controls.enableDamping = true; 

    let pipeline_render = null;

    const render = async () =>
    {
        controls.update();
        if (resized)
        {
            camera.aspect = canvas.width/canvas.height;
            camera.updateProjectionMatrix();
            resized = false;
        }

        render_target.update();

        psystem.update();        

        camera.updateMatrixWorld(false);
    	camera.updateConstant();

        let commandEncoder = engine_ctx.device.createCommandEncoder();

        {
            let clearColor = new Color(0.0, 0.0, 0.0);

            let colorAttachment =  {            
                view: render_target.view_video,
                clearValue: { r: clearColor.r, g: clearColor.g, b: clearColor.b, a: 1 },
                loadOp: 'clear',
                storeOp: 'store'
            };            
            
            let depthAttachment = {
                view: render_target.view_depth,
                depthClearValue: 1,
                depthLoadOp: 'clear',
                depthStoreOp: 'store',
            };
            
            let renderPassDesc = {
                colorAttachments: [colorAttachment],       
                depthStencilAttachment: depthAttachment 
            }; 
            let passEncoder = commandEncoder.beginRenderPass(renderPassDesc);

            passEncoder.setViewport(
                0,
                0,
                render_target.width,
                render_target.height,
                0,
                1
            );
        
            passEncoder.setScissorRect(
                0,
                0,
                render_target.width,
                render_target.height,
            );

            if (pipeline_render == null)
            {
                pipeline_render = GetPipelineRender(render_target.view_format);
            }

            passEncoder.setPipeline(pipeline_render);
            passEncoder.setBindGroup(0, camera.bind_group);

            passEncoder.setVertexBuffer(0, psystem.dPos[0]);
            passEncoder.setVertexBuffer(1, psystem.dColor);

            passEncoder.draw(6, psystem.numParticles);
          
            passEncoder.end();

        }

        let cmdBuf = commandEncoder.finish();
        engine_ctx.queue.submit([cmdBuf]);                
        
        requestAnimationFrame(render);

    }

    render();

}

