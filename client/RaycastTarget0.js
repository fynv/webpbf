export class RaycastTarget0
{
    constructor(camera)
    {
        this.camera = camera;
        this.width = -1;
        this.height = -1;
        this.buf_depth = null;
        this.buf_constant= engine_ctx.createBuffer0(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    }

    update(width , height)
    {
        if (this.buf_depth == null || width!=this.width || height!=this.height)
        {
            this.width = width;
            this.height = height;
            this.count = this.width*this.height;

            const uniform = new Int32Array(4);
            uniform[0] = width;
            uniform[1] = height;
            engine_ctx.queue.writeBuffer(this.buf_constant, 0, uniform.buffer, uniform.byteOffset, uniform.byteLength);
            
            this.buf_depth = engine_ctx.createBuffer0(this.count * 4, GPUBufferUsage.STORAGE);
            
            if (!("raycast0" in engine_ctx.cache.bindGroupLayouts))
            {                
                let layout_entries_raycast0 = [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer:{
                            type: "storage"
                        }
                    },
                    {
                        binding: 1,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer:{
                            type: "uniform"
                        }
                    },
                    {
                        binding: 2,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer:{
                            type: "uniform"
                        }
                    }
                    
                ];                 
                
                engine_ctx.cache.bindGroupLayouts.raycast0 = engine_ctx.device.createBindGroupLayout({ entries: layout_entries_raycast0 });
            }

            let bindGroupLayoutRaycast =  engine_ctx.cache.bindGroupLayouts.raycast0;

            if (!("depth_visualize" in engine_ctx.cache.bindGroupLayouts))
            {
                let layout_entries_depth_visualize = [
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
                        buffer:{
                            type: "read-only-storage"
                        }
                    }
                ];
                
                engine_ctx.cache.bindGroupLayouts.depth_visualize = engine_ctx.device.createBindGroupLayout({ entries: layout_entries_depth_visualize });
            }

            let bindGroupLayoutVisualize = engine_ctx.cache.bindGroupLayouts.depth_visualize;

            {          

                let group_entries = [
                    {
                        binding: 0,
                        resource:{
                            buffer: this.buf_depth            
                        }
                    },
                    {
                        binding: 1,
                        resource:{
                            buffer: this.camera.constant            
                        }
                    },
                    {
                        binding: 2,
                        resource:{
                            buffer: this.buf_constant            
                        }
                    }
                     
                ];
                this.bind_group_raycast = engine_ctx.device.createBindGroup({ layout: bindGroupLayoutRaycast, entries: group_entries});
            }

            {
                let group_entries = [
                    {
                        binding: 0,
                        resource:{
                            buffer: this.buf_constant          
                        }
                    },
                    {
                        binding: 1,
                        resource: {
                            buffer: this.buf_depth            
                        }
                    }
                ];

                this.bind_group_depth_visualize = engine_ctx.device.createBindGroup({ layout: bindGroupLayoutVisualize, entries: group_entries});
            }
        }
    }

}
