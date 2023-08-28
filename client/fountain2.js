import { EngineContext } from "./engine/EngineContext.js"
import { CanvasContext } from "./engine/CanvasContext.js"
import { GPURenderTarget } from "./engine/renderers/GPURenderTarget.js"
import { PerspectiveCameraEx } from "./engine/cameras/PerspectiveCameraEx.js"
import { OrbitControls } from "./engine/controls/OrbitControls.js"
import { Color } from "./engine/math/Color.js"
import { ParticleSystem } from "./particleSystem.js"
import { RenderNaive } from "./render_naive.js"

import { CubeBackground } from "./engine/backgrounds/Background.js"
import { ImageLoader } from "./engine/loaders/ImageLoader.js"
import { DrawSkyBox } from "./engine/renderers/routines/DrawSkyBox.js"
import { EnvironmentMapCreator} from "./engine/lights/EnvironmentMapCreator.js"
import { ParticleTarget } from "./ParticleTarget.js"
import { RenderDepth } from "./render_depth.js"
import { CurvatureFlow } from "./cuvature_flow.js"
import { RenderShading } from "./render_shading.js"
import { RenderThickness } from "./render_thickness.js"

function init_particles(psystem)
{
    let i=0;
    let spacing = psystem.particleRadius * 2.0; 

    let w = Math.ceil(2.0 / psystem.particleRadius / 2.0);
    let h = Math.ceil(0.25 / psystem.particleRadius / 2.0);

    for (let y=0; y<h; y++)
    {
        for (let x=0; x<w; x++)
        {
            for (let z=0; z<w; z++)
            {
                if (i< psystem.numParticles)
                {
                    psystem.hPos[i*4] = -1.0 + psystem.particleRadius + spacing * x;
                    psystem.hPos[i*4 + 1] = psystem.particleRadius + spacing * y;
                    psystem.hPos[i*4 + 2] = -1.0 +  psystem.particleRadius + spacing * z;
                    psystem.hPos[i*4 + 3] = 1.0;

                    psystem.hVel[i*4] = 0.0;
                    psystem.hVel[i*4 + 1] = 0.0;
                    psystem.hVel[i*4 + 2] = 0.0;
                    psystem.hVel[i*4 + 3] = 0.0;
                    i++;
                }
            }
        }
    }
    engine_ctx.queue.writeBuffer(psystem.dPos[0], 0, psystem.hPos.buffer, 0, psystem.hPos.length * 4);
    engine_ctx.queue.writeBuffer(psystem.dVel, 0, psystem.hVel.buffer, 0, psystem.hVel.length * 4);
}

let speed = 5.0;
let delta_t = 0.0;
let idx = 0;

function update_flow(psystem)
{
    let count = Math.pow(Math.ceil(0.25/psystem.particleRadius / 2.0), 2.0);
    let radius = Math.sqrt(count/Math.PI) * psystem.particleRadius*2.0;

    delta_t += psystem.time_step;
    let dist = speed * delta_t;

    while(dist>=psystem.particleRadius*2.0)
    {
        dist -= psystem.particleRadius*2.0;
        delta_t = dist / speed;        
        
        let jitter = Math.random();
        let i= idx;
        for (let j=0; j<count; j++)
        {
            let r = Math.sqrt((j+0.5)/count) * radius;
            let theta = j * 2.4 + jitter * 2.0 * Math.PI;   
            
            let x = r * Math.cos(theta);
            let y = r * Math.sin(theta);

            psystem.hPos[i*4] = x
            psystem.hPos[i*4 + 1] = psystem.particleRadius * 2.0 * 4.0 + dist;
            psystem.hPos[i*4 + 2] = y;
            psystem.hPos[i*4 + 3] = 1.0;

            psystem.hVel[i*4] = 0.0;
            psystem.hVel[i*4 + 1] = speed;
            psystem.hVel[i*4 + 2] = 0.0;
            psystem.hVel[i*4 + 3] = 0.0;

            i++;
        }
      
        engine_ctx.queue.writeBuffer(psystem.dPos[0], idx*4*4, psystem.hPos.buffer, idx*4*4, count*4*4);
        engine_ctx.queue.writeBuffer(psystem.dVel, idx*4*4, psystem.hVel.buffer, idx*4*4, count*4*4);

        idx = i % psystem.numParticles;
    }

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
    let particle_target = new ParticleTarget(); 

    let psystem = new ParticleSystem(); 
    init_particles(psystem);
   
    let camera = new PerspectiveCameraEx();
    camera.position.set(0, 1, 5); 

    let controls = new OrbitControls(camera, canvas);    
    controls.target.set(0,0,0); 
    controls.enableDamping = true; 

    let bg = new CubeBackground();

    let imageLoader = new ImageLoader();
    let cubeImg = await imageLoader.loadCubeFromFile([
        "./assets/textures/sky_cube_face0.jpg",
        "./assets/textures/sky_cube_face1.jpg",
        "./assets/textures/sky_cube_face2.jpg",
        "./assets/textures/sky_cube_face3.jpg",
        "./assets/textures/sky_cube_face4.jpg",
        "./assets/textures/sky_cube_face5.jpg"
    ]);    

    bg.setCubemap(cubeImg); 

    let envMapCreator = new EnvironmentMapCreator(); 
    let envMap = envMapCreator.create(bg);
    
    let sampler = engine_ctx.device.createSampler({
        magFilter: 'linear',
        minFilter: 'linear',
        mipmapFilter: "linear"     
    });

    let layout_entries_envmap = [
        {
            binding: 0,
            visibility: GPUShaderStage.FRAGMENT,
            buffer:{
                type: "uniform"
            }
        },
        {                
            binding: 1,
            visibility: GPUShaderStage.FRAGMENT,
            texture:{
                viewDimension: "cube"
            }
        },
        {
            binding: 2,
            visibility: GPUShaderStage.FRAGMENT,
            sampler:{}
        }
    ];

    let bindGroupLayoutEnvmap = engine_ctx.device.createBindGroupLayout({ entries: layout_entries_envmap });
    engine_ctx.cache.bindGroupLayouts.envmap = bindGroupLayoutEnvmap;    

    let group_entries_envmap = [
        {
            binding: 0,
            resource:{
                buffer: envMap.constant
            }
        },
        {                
            binding: 1,
            resource: envMap.reflection.texture.createView({
                dimension: 'cube'
            })
        },
        {
            binding: 2,
            resource: sampler
        }
    ];

    let bind_group_envmap = engine_ctx.device.createBindGroup({ layout: bindGroupLayoutEnvmap, entries: group_entries_envmap});

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
        particle_target.update(render_target.width, render_target.height);        

        psystem.update();     
        update_flow(psystem);     

        camera.updateMatrixWorld(false);
    	camera.updateConstant();

        let commandEncoder = engine_ctx.device.createCommandEncoder();

        {            
            let depthAttachment = {
                view: particle_target.view_depth[0],
                depthClearValue: 1,
                depthLoadOp: 'clear',
                depthStoreOp: 'store',
            };
            
            let renderPassDesc = {
                colorAttachments: [],
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

            RenderDepth(passEncoder, camera, psystem);
          
            passEncoder.end();
        }

        for (let i=0; i<15; i++)
        {
            {
                let depthAttachment = {
                    view: particle_target.view_depth[1],
                    depthClearValue: 1,
                    depthLoadOp: 'clear',
                    depthStoreOp: 'store',
                };
                
                let renderPassDesc = {
                    colorAttachments: [],
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

                CurvatureFlow(passEncoder, camera, particle_target.bind_groups[0]);
            
                passEncoder.end();

            }

            {
                let depthAttachment = {
                    view: particle_target.view_depth[0],
                    depthClearValue: 1,
                    depthLoadOp: 'clear',
                    depthStoreOp: 'store',
                };
                
                let renderPassDesc = {
                    colorAttachments: [],
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

                CurvatureFlow(passEncoder, camera, particle_target.bind_groups[1]);
            
                passEncoder.end();

            }
        }

        let clearColor = new Color(0.0, 0.0, 0.0);

        let colorAttachment =  {            
            view: render_target.view_video,
            clearValue: { r: clearColor.r, g: clearColor.g, b: clearColor.b, a: 1 },
            loadOp: 'clear',
            storeOp: 'store'
        };

        {               
            
            let renderPassDesc = {
                colorAttachments: [colorAttachment], 
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

            DrawSkyBox(passEncoder, render_target, camera, bg);

            RenderThickness(passEncoder, camera, psystem, render_target);
            RenderShading(passEncoder, camera, particle_target.bind_groups[0], bind_group_envmap, render_target);
          
            passEncoder.end();

        }

        let cmdBuf = commandEncoder.finish();
        engine_ctx.queue.submit([cmdBuf]);                
        
        requestAnimationFrame(render);

    }

    render();

}

