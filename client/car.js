import { EngineContext } from "./engine/EngineContext.js"
import { CanvasContext } from "./engine/CanvasContext.js"
import { GPURenderTarget } from "./engine/renderers/GPURenderTarget.js"
import { PerspectiveCameraEx } from "./engine/cameras/PerspectiveCameraEx.js"
import { OrbitControls } from "./engine/controls/OrbitControls.js"
import { Color } from "./engine/math/Color.js"

import { CubeBackground } from "./engine/backgrounds/Background.js"
import { ImageLoader } from "./engine/loaders/ImageLoader.js"
import { DrawSkyBox } from "./engine/renderers/routines/DrawSkyBox.js"
import { EnvironmentMapCreator} from "./engine/lights/EnvironmentMapCreator.js"

import { GLTFLoader } from "./engine/loaders/GLTFLoader.js"

import { Vector2 } from "./engine/math/Vector2.js"
import { Vector3 } from "./engine/math/Vector3.js"
import { Vector4 } from "./engine/math/Vector4.js"
import { Matrix4 } from "./engine/math/Matrix4.js"
import {Lights} from "./engine/lights/Lights.js"
import { RenderDepth } from "./engine/renderers/routines/DepthOnly.js"
import { RenderStandard } from "./engine/renderers/routines/StandardRoutine.js"
import { ResolveWeightedOIT } from "./engine/renderers/routines/WeightedOIT.js"

import { CWBVH } from "./CWBVH.js"
import { RaycastTarget0 } from "./RaycastTarget0.js"
import { ClearDepth } from "./clear_depth.js"
import { RaycastDepth } from "./raycast_depth.js"
import { VisualizeDepth } from "./visualize_depth.js"

function toViewAABB(MV, min_pos, max_pos)
{
    let view_pos = [];
    {
        let pos = new Vector4(min_pos.x, min_pos.y, min_pos.z, 1.0);
        pos.applyMatrix4(MV);
        view_pos.push(pos);
    }

    {
        let pos = new Vector4(max_pos.x, min_pos.y, min_pos.z, 1.0);
        pos.applyMatrix4(MV);
        view_pos.push(pos);
    }

    {
        let pos = new Vector4(min_pos.x, max_pos.y, min_pos.z, 1.0);
        pos.applyMatrix4(MV);
        view_pos.push(pos);
    }
    
    {
        let pos = new Vector4(max_pos.x, max_pos.y, min_pos.z, 1.0);
        pos.applyMatrix4(MV);
        view_pos.push(pos);
    }

    {
        let pos = new Vector4(min_pos.x, min_pos.y, max_pos.z, 1.0);
        pos.applyMatrix4(MV);
        view_pos.push(pos);
    }

    {
        let pos = new Vector4(max_pos.x, min_pos.y, max_pos.z, 1.0);
        pos.applyMatrix4(MV);
        view_pos.push(pos);
    }

    {
        let pos = new Vector4(min_pos.x, max_pos.y, max_pos.z, 1.0);
        pos.applyMatrix4(MV);
        view_pos.push(pos);
    }
    
    {
        let pos = new Vector4(max_pos.x, max_pos.y, max_pos.z, 1.0);
        pos.applyMatrix4(MV);
        view_pos.push(pos);
    }

    let min_pos_view = new Vector3(Infinity,Infinity,Infinity);
    let max_pos_view = new Vector3(-Infinity, -Infinity, -Infinity);

    for (let k=0; k<8; k++)
    {
        let pos = new Vector3(view_pos[k].x, view_pos[k].y, view_pos[k].z);
        min_pos_view.min(pos);
        max_pos_view.max(pos);
    }

    return { min_pos_view, max_pos_view };
}


function visible(MV, P, min_pos, max_pos, scissor = { min_proj: new Vector2(-1,-1), max_proj: new Vector2(1,1) })
{
    let { min_pos_view, max_pos_view} = toViewAABB(MV, min_pos, max_pos);

    let invP = P.clone();        
    invP.invert();
    let view_far = new Vector4(0.0, 0.0, 1.0, 1.0);
    view_far.applyMatrix4(invP);
    view_far.multiplyScalar(1.0/view_far.w);
    let view_near = new Vector4(0.0, 0.0, -1.0, 1.0);
    view_near.applyMatrix4(invP);
    view_near.multiplyScalar(1.0/view_near.w);

    if (min_pos_view.z > view_near.z) return false;
    if (max_pos_view.z < view_far.z) return false;
    if (min_pos_view.z < view_far.z)
    {
        min_pos_view.z = view_far.z;
    }

    let min_pos_proj = new Vector4(min_pos_view.x, min_pos_view.y, min_pos_view.z, 1.0);
    min_pos_proj.applyMatrix4(P);
    min_pos_proj.multiplyScalar(1.0/min_pos_proj.w);

    let max_pos_proj = new Vector4(max_pos_view.x, max_pos_view.y, min_pos_view.z, 1.0);
    max_pos_proj.applyMatrix4(P);
    max_pos_proj.multiplyScalar(1.0/max_pos_proj.w);

    return max_pos_proj.x >= scissor.min_proj.x && min_pos_proj.x <= scissor.max_proj.x && max_pos_proj.y >= scissor.min_proj.y && min_pos_proj.y <= scissor.max_proj.y;

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

    let msaa = true;
    let render_target = new GPURenderTarget(canvas_ctx, msaa);        
    
    let camera = new PerspectiveCameraEx();
    camera.position.set(0, 1, 5); 

    let raycast_target = new RaycastTarget0(camera);

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
    
    let lights = new Lights();        
    lights.reflection_map = envMap.reflection;
    lights.environment_map = envMap;
    lights.update_bind_group(); 

    let model_loader = new GLTFLoader();
    let model = model_loader.loadModelFromFile("./assets/models/alfa_romeo_stradale_1967.glb"); 
    model.scale.set(10,10,10);
    model.rotateY(-3.14159*0.5);

    await model.meta_ready();
    for (let mesh of model.meshes)
    {     
        for (let primitive of mesh.primitives)
        {            
            primitive.cwbvh = new CWBVH(primitive, mesh.constant);
        }
    }

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
        raycast_target.update(Math.ceil(render_target.width*0.3), Math.ceil(render_target.height*0.3));        

        camera.updateMatrixWorld(false);
    	camera.updateConstant();

        model.updateWorldMatrix(false, false);
        model.updateMeshConstants();

        {
            let commandEncoder = engine_ctx.device.createCommandEncoder();
            ClearDepth(commandEncoder, raycast_target);
            for (let mesh of model.meshes)
            {
                for (let primitive of mesh.primitives)
                {
                    if (!primitive.cwbvh.is_ready) continue;
                    RaycastDepth(commandEncoder, primitive.cwbvh, raycast_target);
                }
            }
            let cmdBuf = commandEncoder.finish();
            engine_ctx.queue.submit([cmdBuf]); 
        }

        envMap.updateConstant();

        let commandEncoder = engine_ctx.device.createCommandEncoder();

        let clearColor = new Color(0.0, 0.0, 0.0);

        let colorAttachment =  {                            
            clearValue: { r: clearColor.r, g: clearColor.g, b: clearColor.b, a: 1 },
            loadOp: 'clear',
            storeOp: 'store'
        };       

        {              

            if (msaa)
            {
                colorAttachment.view = render_target.view_msaa;                
            }
            else
            {
                colorAttachment.view = render_target.view_video;
            }      
          
            let renderPassDesc = {
                colorAttachments: [colorAttachment]                
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
          
            passEncoder.end();

            colorAttachment.loadOp = 'load';
        }

        // depth-prepass

        let depthAttachment = {
            view: render_target.view_depth,
            depthClearValue: 1,
            depthLoadOp: 'clear',
            depthStoreOp: 'store',
        };

        {           

            let renderPassDesc_depth = {
                colorAttachments: [],
                depthStencilAttachment: depthAttachment
            }; 

            let passEncoder = commandEncoder.beginRenderPass(renderPassDesc_depth);

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

            for (let mesh of model.meshes)
            {
                let matrix = model.matrixWorld.clone();
                if (mesh.node_id >=0 && mesh.skin_id <0)
                {
                    let node = model.nodes[mesh.node_id];
                    matrix.multiply(node.g_trans);
                }
                let MV = new Matrix4();
                MV.multiplyMatrices(camera.matrixWorldInverse, matrix);

                for (let primitive of mesh.primitives)
                {
                    if (primitive.uuid == 0) continue;
                    if (!visible(MV, camera.projectionMatrix, primitive.min_pos, primitive.max_pos)) continue;

                    let idx_material = primitive.material_idx;
                    let material = model.materials[idx_material];
    
                    if (material.alphaMode != "Opaque") continue;
    
                    let params = {                    
                        target: render_target,
                        material_list: model.materials,
                        camera, 
                        primitive: primitive,
                    };

                    RenderDepth(passEncoder, params);
                }
            }
            
            passEncoder.end();
            depthAttachment.depthLoadOp = 'load';
        }

        // opaque pass
        {
            let renderPassDesc_opaque = {
                colorAttachments: [colorAttachment],
                depthStencilAttachment: depthAttachment
            };
             
            let passEncoder = commandEncoder.beginRenderPass(renderPassDesc_opaque);

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

            for (let mesh of model.meshes)
            {
                let matrix = model.matrixWorld.clone();
                if (mesh.node_id >=0 && mesh.skin_id <0)
                {
                    let node = model.nodes[mesh.node_id];
                    matrix.multiply(node.g_trans);
                }
                let MV = new Matrix4();
                MV.multiplyMatrices(camera.matrixWorldInverse, matrix);

                for (let primitive of mesh.primitives)
                {
                    if (primitive.uuid == 0) continue;
                    if (!visible(MV, camera.projectionMatrix, primitive.min_pos, primitive.max_pos)) continue;

                    let idx_material = primitive.material_idx;
                    let material = model.materials[idx_material];
    
                    if (material.alphaMode == "Blend") continue;

                    let params = {                    
                        target: render_target,
                        material_list: model.materials,
                        camera,          
                        primitive: primitive,
                        lights
                    };
                    
                    RenderStandard(passEncoder, params);    

                }
            }

            passEncoder.end();
        }

        // alpha pass
        render_target.update_oit_buffers();
        {
            let oitAttchment0 = {
                view: render_target.oit_view0,
                clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 0.0 },
                loadOp: 'clear',
                storeOp: 'store'
            }
    
            let oitAttchment1 = {
                view: render_target.oit_view1,
                clearValue: { r: 1.0, g: 1.0, b: 1.0, a: 1.0 },
                loadOp: 'clear',
                storeOp: 'store'
            }

            let renderPassDesc_alpha = {
                colorAttachments: [colorAttachment, oitAttchment0, oitAttchment1],
                depthStencilAttachment: depthAttachment
            }; 

            let passEncoder = commandEncoder.beginRenderPass(renderPassDesc_alpha);

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

            for (let mesh of model.meshes)
            {
                let matrix = model.matrixWorld.clone();
                if (mesh.node_id >=0 && mesh.skin_id <0)
                {
                    let node = model.nodes[mesh.node_id];
                    matrix.multiply(node.g_trans);
                }
                let MV = new Matrix4();
                MV.multiplyMatrices(camera.matrixWorldInverse, matrix);

                for (let primitive of mesh.primitives)
                {
                    if (primitive.uuid == 0) continue;
                    if (!visible(MV, camera.projectionMatrix, primitive.min_pos, primitive.max_pos)) continue;

                    let idx_material = primitive.material_idx;
                    let material = model.materials[idx_material];
    
                    if (material.alphaMode != "Blend") continue;

                    let params = {                    
                        target: render_target,
                        material_list: model.materials,
                        camera,          
                        primitive: primitive,
                        lights
                    };
                    
                    RenderStandard(passEncoder, params);    

                }
            }

            passEncoder.end();
            
        }

        {
            colorAttachment.resolveTarget = render_target.view_video;

            let renderPassDesc_oit = {
                colorAttachments: [colorAttachment],            
            }; 

            let passEncoder = commandEncoder.beginRenderPass(renderPassDesc_oit);

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

            ResolveWeightedOIT(passEncoder, render_target);

            passEncoder.end();
        }

        
        {
            let colorAttachment =  {   
                view: render_target.view_video,                
                loadOp: 'load',
                storeOp: 'store'
            };    

            let renderPassDesc = {
                colorAttachments: [colorAttachment],
            };
             
            let passEncoder = commandEncoder.beginRenderPass(renderPassDesc);

            passEncoder.setViewport(
                Math.ceil(render_target.width*0.6),
                Math.ceil(render_target.height*0.1),
                raycast_target.width,
                raycast_target.height,
                0,
                1
            );
        
            passEncoder.setScissorRect(
                Math.ceil(render_target.width*0.6),
                Math.ceil(render_target.height*0.1),
                raycast_target.width,
                raycast_target.height,
            );

            VisualizeDepth(passEncoder, raycast_target.bind_group_depth_visualize, render_target);

            passEncoder.end();
        }

        let cmdBuf = commandEncoder.finish();
        engine_ctx.queue.submit([cmdBuf]);                
        
        requestAnimationFrame(render);
    }

    render();

}