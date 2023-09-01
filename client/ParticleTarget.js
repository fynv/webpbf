export class ParticleTarget
{
    constructor()
    {
        this.tex_thickness = null;
        this.view_thickness = null;
        this.tex_depth = [null, null];
        this.view_depth = [null, null];
        this.width = -1;
        this.height = -1;
        this.depth_width = -1;
        this.depth_height = -1;

        
        this.sampler = engine_ctx.device.createSampler({
            magFilter: 'linear',
            minFilter: 'linear',
            mipmapFilter: "linear"           
        });
    }

    update(width , height)
    {
        if (this.view_thickness == null || width!=this.width || height!=this.height)
        {
            this.tex_thickness = engine_ctx.device.createTexture({
                size: { width, height},
                dimension: "2d",
                format: "r8unorm",
                usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING 
            });
            
            this.view_thickness =  this.tex_thickness.createView();

            this.width = width;
            this.height = height;            

            let depth_height = this.height<540 ? this.height : 540;
            let depth_width = Math.ceil(depth_height * this.width / this.height);

            if(depth_width!= this.depth_width || depth_height!=this.depth_height)
            {
                this.tex_depth[0] = engine_ctx.device.createTexture({
                    size: [depth_width, depth_height],
                    dimension: "2d",
                    format: 'depth32float',
                    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
                });
                
                this.view_depth[0] = this.tex_depth[0].createView();

                this.tex_depth[1] = engine_ctx.device.createTexture({
                    size: [depth_width, depth_height],
                    dimension: "2d",
                    format: 'depth32float',
                    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
                });
                
                this.view_depth[1] = this.tex_depth[1].createView();

                this.depth_width = depth_width;
                this.depth_height = depth_height;

                if (!("particle_depth" in engine_ctx.cache.bindGroupLayouts))
                {
                    let layout_entries_particle_depth = [
                        {
                            binding: 0,
                            visibility: GPUShaderStage.FRAGMENT,
                            texture:{
                                viewDimension: "2d",
                                sampleType: "unfilterable-float"
                            }
                        }
                    ];   

                    engine_ctx.cache.bindGroupLayouts.particle_depth = engine_ctx.device.createBindGroupLayout({ entries: layout_entries_particle_depth });
                }

                let bindGroupLayout =  engine_ctx.cache.bindGroupLayouts.particle_depth;

                let group_entries0 = [
                    {
                        binding: 0,
                        resource: this.view_depth[0]
                    },
                ];
    
    
                let group_entries1 = [
                    {
                        binding: 0,
                        resource: this.view_depth[1]
                    },
                ];
    
                this.bind_groups_depth = [null, null];        
                this.bind_groups_depth[0] = engine_ctx.device.createBindGroup({ layout: bindGroupLayout, entries: group_entries0});
                this.bind_groups_depth[1] = engine_ctx.device.createBindGroup({ layout: bindGroupLayout, entries: group_entries1});
            }

            if (!("particle_frame" in engine_ctx.cache.bindGroupLayouts))
            {
                let layout_entries_particle_frame = [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.FRAGMENT,
                        texture:{
                            viewDimension: "2d",
                            sampleType: "unfilterable-float"
                        }
                    },
                    {
                        binding: 1,
                        visibility: GPUShaderStage.FRAGMENT,
                        texture:{
                            viewDimension: "2d",
                            sampleType: "depth"
                        }
                    },
                    {
                        binding: 2,
                        visibility: GPUShaderStage.FRAGMENT,
                        sampler:{}
                    }
                ]; 
                
                engine_ctx.cache.bindGroupLayouts.particle_frame = engine_ctx.device.createBindGroupLayout({ entries: layout_entries_particle_frame });
            }

            let bindGroupLayoutFrame =engine_ctx.cache.bindGroupLayouts.particle_frame;

            let group_entries_frame = [
                {
                    binding: 0,
                    resource: this.view_thickness
                },
                {
                    binding: 1,
                    resource: this.view_depth[0]
                },
                {
                    binding: 2,
                    resource: this.sampler
                }
            ];

            this.bind_group_frame = engine_ctx.device.createBindGroup({ layout: bindGroupLayoutFrame, entries: group_entries_frame});
        }
    }


}
