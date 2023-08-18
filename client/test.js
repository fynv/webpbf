import { EngineContext } from "./engine/EngineContext.js"
import { CanvasContext } from "./engine/CanvasContext.js"
import { GPURenderTarget } from "./engine/renderers/GPURenderTarget.js"
import { PerspectiveCameraEx } from "./engine/cameras/PerspectiveCameraEx.js"
import { OrbitControls } from "./engine/controls/OrbitControls.js"
import { Color } from "./engine/math/Color.js"
import { ParticleSystem } from "./particleSystem.js"
import { RenderNaive } from "./render_naive.js"

function init_particles(psystem)
{
    let s =  Math.ceil(Math.pow(psystem.numParticles, 1.0/3.0));
    let h = 1.5;

    let x0 = - s * psystem.particleRadius;
    let y0 = h  - s * psystem.particleRadius;
    let z0 = - s * psystem.particleRadius;

    let spacing = psystem.particleRadius * 2.0;  

    for (let z=0; z<s; z++)
    {
        for (let y=0; y<s; y++)
        {
            for (let x=0; x< s; x++)
            {
                let i = x + (y + z*s)*s;

                if (i<psystem.numParticles)
                {
                    psystem.hPos[i*4] = x0 + spacing * x;
                    psystem.hPos[i*4 + 1] = y0 + spacing * y;
                    psystem.hPos[i*4 + 2] = z0 + spacing * z;
                    psystem.hPos[i*4 + 3] = 1.0;

                    psystem.hVel[i*4] = 0.0;
                    psystem.hVel[i*4 + 1] = 0.0;
                    psystem.hVel[i*4 + 2] = 0.0;
                    psystem.hVel[i*4 + 3] = 0.0;
                }

            }
        }
    }

    engine_ctx.queue.writeBuffer(psystem.dPos[0], 0, psystem.hPos.buffer, 0, psystem.hPos.length * 4);
    engine_ctx.queue.writeBuffer(psystem.dVel, 0, psystem.hVel.buffer, 0, psystem.hVel.length * 4);
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
    init_particles(psystem);
   
    let camera = new PerspectiveCameraEx();
    camera.position.set(0, 1, 5); 

    let controls = new OrbitControls(camera, canvas);    
    controls.target.set(0,0,0); 
    controls.enableDamping = true; 

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

            RenderNaive(passEncoder, camera, psystem, render_target);
          
            passEncoder.end();

        }

        let cmdBuf = commandEncoder.finish();
        engine_ctx.queue.submit([cmdBuf]);                
        
        requestAnimationFrame(render);

    }

    render();

}

