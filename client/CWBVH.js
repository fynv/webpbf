import get_module from './CWBVH_Wasm.js'

let module = null;

export class CWBVH
{
    constructor(primitive, model_constant)
    {       
        this.primitive = primitive;
        this.model_constant = model_constant;
        
        this.is_ready = false;
        this.resolvers = [];

        this._init();
        
    }

    async ready()
    {
        return new Promise((resolve, reject) => {
            if (this.is_ready)
            {
                resolve(true);
            }
            else
            {
                this.resolvers.push(resolve);
            }
        });
    }

    set_ready()
    {
        this.is_ready = true;
        for (let resolve of this.resolvers) 
        {
            resolve(true);
        }
    }

    async _init()
    {
        if (module == null)
        {
            module = await get_module();
        }
        await this.primitive.geometry_ready();

        let p_indices = module.ccall("alloc", "number", ["number"], [this.primitive.cpu_indices.byteLength]);
        module.HEAPU8.set(new Uint8Array(this.primitive.cpu_indices), p_indices);

        let p_pos = module.ccall("alloc", "number", ["number"], [this.primitive.cpu_pos.byteLength]);
        module.HEAPU8.set(new Uint8Array(this.primitive.cpu_pos), p_pos);

        let p_cwbvh = module.ccall("CreateCWBVH", "number", ["number", "number", "number", "number", "number"], 
            [this.primitive.num_face, this.primitive.num_pos, p_indices, this.primitive.type_indices, p_pos]);

        let num_nodes =  module.ccall("CWBVH_NumNodes", "number", ["number"], [p_cwbvh]);
        let num_triangles =  module.ccall("CWBVH_NumTriangles", "number", ["number"], [p_cwbvh]);
        let p_bvh_nodes = module.ccall("CWBVH_Nodes", "number", ["number"], [p_cwbvh]);
        let p_bvh_indices = module.ccall("CWBVH_Indices", "number", ["number"], [p_cwbvh]);
        let p_bvh_triangles = module.ccall("CWBVH_Triangles", "number", ["number"], [p_cwbvh]);

        this.nodes = engine_ctx.createBuffer(module.HEAPU8.buffer, GPUBufferUsage.STORAGE, p_bvh_nodes, num_nodes * 5 * 4 * 4);
        this.indices = engine_ctx.createBuffer(module.HEAPU8.buffer, GPUBufferUsage.STORAGE, p_bvh_indices, num_triangles * 4);
        this.triangles = engine_ctx.createBuffer(module.HEAPU8.buffer, GPUBufferUsage.STORAGE, p_bvh_triangles, num_triangles * 3 * 4 * 4);

        module.ccall("DestroyCWBVH", null, ["number"], [p_cwbvh]);
        module.ccall("dealloc", null, ["number"], [p_indices]);
        module.ccall("dealloc", null, ["number"], [p_pos]);

        if (!("cwbvh" in engine_ctx.cache.bindGroupLayouts))
        {    
            let layout_entries = [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer:{
                        type: "read-only-storage"
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
                        type: "uniform"
                    }
                }
            ];                 
            
            engine_ctx.cache.bindGroupLayouts.cwbvh = engine_ctx.device.createBindGroupLayout({ entries: layout_entries });
        }

        let bindGroupLayout =  engine_ctx.cache.bindGroupLayouts.cwbvh;

        {
            let group_entries = [
                {
                    binding: 0,
                    resource:{
                        buffer: this.nodes     
                    }
                },
                {
                    binding: 1,
                    resource:{
                        buffer: this.indices            
                    }
                },
                {
                    binding: 2,
                    resource:{
                        buffer: this.triangles            
                    }
                },
                {
                    binding: 3,
                    resource:{
                        buffer: this.model_constant
                    }
                },                
            ];
            this.bind_group = engine_ctx.device.createBindGroup({ layout: bindGroupLayout, entries: group_entries});
        }

        this.set_ready();
    }

}
