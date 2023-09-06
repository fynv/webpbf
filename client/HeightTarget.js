export class HeightTarget
{
    constructor()
    {
        this.tex_depth = null;
        this.view_depth = null;

        this.sampler = engine_ctx.device.createSampler({
            magFilter: 'linear',
            minFilter: 'linear',
            mipmapFilter: "linear"           
        });
    }

    update(width , height)
    {
        if (this.view_depth == null || width!=this.width || height!=this.height)
        {
            this.tex_depth = engine_ctx.device.createTexture({
                size: [width, height],
                dimension: "2d",
                format: 'depth32float',
                usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
            });
            this.view_depth = this.tex_depth.createView();

            this.width = width;
            this.height = height;      

            if (!("height_field" in engine_ctx.cache.bindGroupLayouts))
            {
                let layout_entries_height_field = [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.COMPUTE,
                        texture:{
                            viewDimension: "2d",
                            sampleType: "depth"
                        }
                    },
                    {
                        binding: 1,
                        visibility: GPUShaderStage.COMPUTE,
                        sampler:{}
                    }
                ];   

                engine_ctx.cache.bindGroupLayouts.height_field = engine_ctx.device.createBindGroupLayout({ entries: layout_entries_height_field });
            }

            let bindGroupLayout =engine_ctx.cache.bindGroupLayouts.height_field;

            let group_entries= [                
                {
                    binding: 0,
                    resource: this.view_depth
                },
                {
                    binding: 1,
                    resource: this.sampler
                }
            ];

            this.bind_group = engine_ctx.device.createBindGroup({ layout: bindGroupLayout, entries: group_entries});
        }
    }
}
