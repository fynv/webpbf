export class ParticleTarget
{
    constructor()
    {
        this.tex_depth = [null, null];
        this.view_depth = [null, null];
        this.width = -1;
        this.height = -1;
    }

    update(width , height)
    {
        if (this.view_depth == null || width!=this.width || height!=this.height)
        {
            this.tex_depth[0] = engine_ctx.device.createTexture({
                size: [width, height],
                dimension: "2d",
                format: 'depth32float',
                usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
            });
            
            this.view_depth[0] = this.tex_depth[0].createView();

            this.tex_depth[1] = engine_ctx.device.createTexture({
                size: [width, height],
                dimension: "2d",
                format: 'depth32float',
                usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
            });
            
            this.view_depth[1] = this.tex_depth[1].createView();

            this.width = width;
            this.height = height;          

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
            
            let bindGroupLayout = engine_ctx.device.createBindGroupLayout({ entries: layout_entries_particle_depth });
            engine_ctx.cache.bindGroupLayouts.particle_depth = bindGroupLayout;


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

            this.bind_groups = [null, null];        
            this.bind_groups[0] = engine_ctx.device.createBindGroup({ layout: bindGroupLayout, entries: group_entries0});
            this.bind_groups[1] = engine_ctx.device.createBindGroup({ layout: bindGroupLayout, entries: group_entries1});
        }
    }
}

