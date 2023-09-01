import {ClearDepth} from "./clear_depth1.js"

export class RaycastTarget
{
    constructor(psystem)
    {
        this.psystem = psystem;

        this.buf_depth = engine_ctx.createBuffer0(psystem.numParticles * 4, GPUBufferUsage.STORAGE);
        this.buf_normal = engine_ctx.createBuffer0(psystem.numParticles * 4, GPUBufferUsage.STORAGE);

        if (!("raycast1" in engine_ctx.cache.bindGroupLayouts))
        {   
            let layout_entries_raycast1 = [    
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer:{
                        type: "uniform"
                    }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer:{
                        type: "read-only-storage"
                    }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer:{
                        type: "read-only-storage"
                    }
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer:{
                        type: "storage"
                    }
                },
                {
                    binding: 4,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer:{
                        type: "storage"
                    }
                },

            ];

            engine_ctx.cache.bindGroupLayouts.raycast1 = engine_ctx.device.createBindGroupLayout({ entries: layout_entries_raycast1 });
        }        
        let bindGroupLayoutRaycast =  engine_ctx.cache.bindGroupLayouts.raycast1;


        /////////////////////

        {          

            let group_entries = [
                {
                    binding: 0,
                    resource:{
                        buffer: psystem.dConstant            
                    }
                },
                {
                    binding: 1,
                    resource:{
                        buffer: psystem.dSortedPos
                    }
                },
                {
                    binding: 2,
                    resource:{
                        buffer: psystem.dSortedVel   
                    }
                },
                {
                    binding: 3,
                    resource:{
                        buffer: this.buf_depth   
                    }
                },
                {
                    binding: 4,
                    resource:{
                        buffer: this.buf_normal   
                    }
                },
                 
            ];
            this.bind_group_raycast = engine_ctx.device.createBindGroup({ layout: bindGroupLayoutRaycast, entries: group_entries});
        }

    }

    clear(commandEncoder)
    {
        ClearDepth(commandEncoder, this);
    }

}

