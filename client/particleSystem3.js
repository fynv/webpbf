import { ParticleReduction } from "./particleReduction.js"
import { UpdateConstant } from "./updateConstant.js"
import { ClearCellCount } from "./clear_cellcount.js"
import { HashCount } from "./hashCount.js"
import { PrefixSum } from "./prefix_sum.js"
import { Scatter } from "./scatter.js"
import { UpdateDensity } from "./updateDensity.js"
import { UpdatePosition } from "./updatePosition3.js"

const workgroup_size = 64;
const workgroup_size_2x = workgroup_size*2;

export class ParticleSystem
{
    constructor(global_min = [-1.0, 0.0, -1.0], global_max = [1.0, 2.0, 1.0])
    {
        this.density = 1000;
        this.numParticles = 1<< 15;
        this.volume = 1.0;
        this.particleRadius = Math.pow(this.volume / this.numParticles, 1.0/3.0) * 0.5;
        this.particleMass = this.density * this.volume / this.numParticles;
        
        this.H = this.particleRadius * 8.0;
        this.time_step = 1.0/60.0;         
        this.gas_const =  20.0;
        this.pg = 10.0;
        this.gravity = 9.81;
        this.pt = 0.6;
        this.pmin_sur_grad = 10.0;

        this._initialize(global_min, global_max);
    }

    _initialize(global_min, global_max)
    {
        this.global_min = global_min;
        this.global_max = global_max;
        this.hPos = new Float32Array(this.numParticles * 4);
        this.hVel = new Float32Array(this.numParticles * 4);


        this.dPos = [];
        
        let count = this.numParticles;    
        { 
            let buf = engine_ctx.createBuffer0(count*4*4, GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
            this.dPos.push(buf);
        }
        while(count>1)
        {
            count = Math.floor((count + workgroup_size_2x - 1)/workgroup_size_2x);
            let buf = engine_ctx.createBuffer0(count*4*4, GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
            this.dPos.push(buf);
        }               

        this.dVel = engine_ctx.createBuffer0(this.numParticles * 4 * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        this.dSortedPos = engine_ctx.createBuffer0(this.numParticles * 4 * 4, GPUBufferUsage.STORAGE);
        this.dSortedVel = engine_ctx.createBuffer0(this.numParticles * 4 * 4, GPUBufferUsage.STORAGE);
        this.dSortedDensity = engine_ctx.createBuffer0(this.numParticles * 4, GPUBufferUsage.STORAGE);

        this.dGridParticleHash  = engine_ctx.createBuffer0(this.numParticles * 4, GPUBufferUsage.STORAGE);
        this.dGridParticleIndexInCell = engine_ctx.createBuffer0(this.numParticles * 4, GPUBufferUsage.STORAGE);
        this.dGridParticleIndex  = engine_ctx.createBuffer0(this.numParticles * 4, GPUBufferUsage.STORAGE);        

        this.sizeGridBuf = 32*32*32;

        this.dCellCountBufs = [];
        this.dCellCountBufSizes = [];

        let buf_size = this.sizeGridBuf;
        while (buf_size>0)
        {
            let buf = engine_ctx.createBuffer0(buf_size * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);    
            this.dCellCountBufs.push(buf);
            this.dCellCountBufSizes.push(buf_size);
            buf_size = Math.floor((buf_size + workgroup_size_2x - 1)/workgroup_size_2x) - 1;
        }

        this.dConstant = engine_ctx.createBuffer0(52 * 4, GPUBufferUsage.UNIFORM | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE);

        ////////////////////////////

        let layout_entries_particle_reduction = [    
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
                    type: "storage"
                }
            }
        ];

        let bindGroupLayoutParticleReduction = engine_ctx.device.createBindGroupLayout({ entries: layout_entries_particle_reduction });
        engine_ctx.cache.bindGroupLayouts.particle_reduction = bindGroupLayoutParticleReduction;

        let layout_entries_update_constant = [    
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "storage"
                }
            }
        ];

        let bindGroupLayoutUpdateConstant = engine_ctx.device.createBindGroupLayout({ entries: layout_entries_update_constant });
        engine_ctx.cache.bindGroupLayouts.update_const = bindGroupLayoutUpdateConstant;

        let layout_entries_clear_count = [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "storage"
                }
            },
        ];

        let bindGroupLayoutClearCount = engine_ctx.device.createBindGroupLayout({ entries: layout_entries_clear_count });
        engine_ctx.cache.bindGroupLayouts.clear_count = bindGroupLayoutClearCount;

        let layout_entries_hash_count = [
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
                    type: "storage"
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
        let bindGroupLayoutHashCount = engine_ctx.device.createBindGroupLayout({ entries: layout_entries_hash_count });
        engine_ctx.cache.bindGroupLayouts.hash_count = bindGroupLayoutHashCount;

        let layout_entries_prefix_sum1 = [ 
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "storage"
                }
            }
        ];

        let bindGroupLayoutPrefixSum1A = engine_ctx.device.createBindGroupLayout({ entries: layout_entries_prefix_sum1 });
        engine_ctx.cache.bindGroupLayouts.prefix_sum_1a = bindGroupLayoutPrefixSum1A;

        layout_entries_prefix_sum1.push({
            binding: 1,
            visibility: GPUShaderStage.COMPUTE,
            buffer:{
                type: "storage"
            }
        });
    
        let bindGroupLayoutPrefixSum1B = engine_ctx.device.createBindGroupLayout({ entries: layout_entries_prefix_sum1 });
        engine_ctx.cache.bindGroupLayouts.prefix_sum_1b = bindGroupLayoutPrefixSum1B;

        let layout_entries_prefix_sum2 = [
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
                    type: "read-only-storage"
                }
            }
        ];
    
        let bindGroupLayoutPrefixSum2 = engine_ctx.device.createBindGroupLayout({ entries: layout_entries_prefix_sum2 });
        engine_ctx.cache.bindGroupLayouts.prefix_sum_2 = bindGroupLayoutPrefixSum2;

        let layout_entries_scatter = [ 
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
                    type: "read-only-storage"
                }
            },
            {
                binding: 4,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "read-only-storage"
                }
            },
            {
                binding: 5,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "storage"
                }
            },
            {
                binding: 6,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "storage"
                }
            },
            {
                binding: 7,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "storage"
                }
            },

        ];

        let bindGroupLayoutScatter = engine_ctx.device.createBindGroupLayout({ entries: layout_entries_scatter });
        engine_ctx.cache.bindGroupLayouts.scatter = bindGroupLayoutScatter;

        let layout_entries_update_density = [
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

        ];

        let bindGroupLayoutUpdateDensity = engine_ctx.device.createBindGroupLayout({ entries: layout_entries_update_density });
        engine_ctx.cache.bindGroupLayouts.update_density = bindGroupLayoutUpdateDensity;

        let layout_entries_update_position = [
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
                    type: "storage"
                }
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "storage"
                }
            },
            {
                binding: 3,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "read-only-storage"
                }
            },
            {
                binding: 4,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "read-only-storage"
                }
            },
            {
                binding: 5,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "read-only-storage"
                }
            },
            {
                binding: 6,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "read-only-storage"
                }
            },
            {
                binding: 7,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "read-only-storage"
                }
            }
        ];

        let bindGroupLayoutUpdatePosition = engine_ctx.device.createBindGroupLayout({ entries: layout_entries_update_position });
        engine_ctx.cache.bindGroupLayouts.update_position = bindGroupLayoutUpdatePosition;

        let layout_entries_render = [
            {
                binding: 0,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                buffer:{
                    type: "uniform"
                }
            },
        ];
        
        let bindGroupLayoutRender = engine_ctx.device.createBindGroupLayout({ entries: layout_entries_render });
        engine_ctx.cache.bindGroupLayouts.particle_render = bindGroupLayoutRender;

        /////////////////////////////////////

        this.bind_group_particle_reduction = [];

        for (let i=0; i<this.dPos.length-1; i++)
        {
            let group_entries = [
                {
                    binding: 0,
                    resource:{
                        buffer: this.dPos[i]            
                    }
                },
                {
                    binding: 1,
                    resource:{
                        buffer: this.dPos[i+1]
                    }
                }
            ];

            this.bind_group_particle_reduction.push(engine_ctx.device.createBindGroup({ layout: bindGroupLayoutParticleReduction, entries: group_entries}));
        }

        {
            let group_entries = [
                {
                    binding: 0,
                    resource:{
                        buffer: this.dConstant            
                    }
                }                
            ];
            this.bind_group_update_constant = engine_ctx.device.createBindGroup({ layout: bindGroupLayoutUpdateConstant, entries: group_entries});
        }

        let group_entries_clear_count = [
            {
                binding: 0,
                resource:{
                    buffer: this.dCellCountBufs[0]
                }
            },
        ];

        this.bind_group_clear_count = engine_ctx.device.createBindGroup({ layout: bindGroupLayoutClearCount, entries: group_entries_clear_count});

        let group_entries_hash_count = [
            {
                binding: 0,
                resource:{
                    buffer: this.dConstant            
                }
            },
            {
                binding: 1,
                resource:{
                    buffer: this.dPos[0]            
                }
            },
            {
                binding: 2,
                resource:{
                    buffer: this.dGridParticleHash
                }
            },
            {
                binding: 3,
                resource:{
                    buffer: this.dGridParticleIndexInCell
                }
            },
            {
                binding: 4,
                resource:{
                    buffer: this.dCellCountBufs[0]
                }
            },

        ];

        this.bind_group_hash_count = engine_ctx.device.createBindGroup({ layout: bindGroupLayoutHashCount, entries: group_entries_hash_count});

        this.bind_group_prefix_sum1 = [];
        this.bind_group_prefix_sum2 = [];

        for (let i=0; i<this.dCellCountBufs.length; i++)
        {
            if (i<this.dCellCountBufs.length - 1)
            {
                let group_entries = [ 
                    {
                        binding: 0,
                        resource:{
                            buffer: this.dCellCountBufs[i]            
                        }
                    },
                    {
                        binding: 1,
                        resource:{
                            buffer: this.dCellCountBufs[i+1]
                        }
                    }       
                ];
                {
                    let bind_group = engine_ctx.device.createBindGroup({ layout: bindGroupLayoutPrefixSum1B, entries: group_entries});
                    this.bind_group_prefix_sum1.push(bind_group);
                }
                {
                    let bind_group = engine_ctx.device.createBindGroup({ layout: bindGroupLayoutPrefixSum2, entries: group_entries});
                    this.bind_group_prefix_sum2.push(bind_group);
                }
            }
            else if (this.dCellCountBufSizes[i] > 1)
            {
                let group_entries = [ 
                    {
                        binding: 0,
                        resource:{
                            buffer: this.dCellCountBufs[i]            
                        }
                    }              
                ];
                let bind_group = engine_ctx.device.createBindGroup({ layout: bindGroupLayoutPrefixSum1A, entries: group_entries});
                this.bind_group_prefix_sum1.push(bind_group);
            }

        }

        let group_entries_scatter = [
            {
                binding: 0,
                resource:{
                    buffer: this.dPos[0]
                }
            },
            {
                binding: 1,
                resource:{
                    buffer: this.dVel            
                }
            },
            {
                binding: 2,
                resource:{
                    buffer: this.dCellCountBufs[0]            
                }
            },
            {
                binding: 3,
                resource:{
                    buffer: this.dGridParticleHash            
                }
            },
            {
                binding: 4,
                resource:{
                    buffer: this.dGridParticleIndexInCell            
                }
            },
            {
                binding: 5,
                resource:{
                    buffer: this.dSortedPos            
                }
            },
            {
                binding: 6,
                resource:{
                    buffer: this.dSortedVel            
                }
            },
            {
                binding: 7,
                resource:{
                    buffer: this.dGridParticleIndex            
                }
            },
        ];

        this.bind_group_scatter = engine_ctx.device.createBindGroup({ layout: bindGroupLayoutScatter, entries: group_entries_scatter});

        let group_entries_update_density = [
            {
                binding: 0,
                resource:{
                    buffer: this.dConstant
                }
            },
            {
                binding: 1,
                resource:{
                    buffer: this.dSortedPos
                }
            },
            {
                binding: 2,
                resource:{
                    buffer: this.dCellCountBufs[0]
                }
            },
            {
                binding: 3,
                resource:{
                    buffer: this.dSortedDensity            
                }
            },           
        ];

        this.bind_group_update_density = engine_ctx.device.createBindGroup({ layout: bindGroupLayoutUpdateDensity, entries: group_entries_update_density});

        let group_entries_update_position = [
            {
                binding: 0,
                resource:{
                    buffer: this.dConstant
                }
            },
            {
                binding: 1,
                resource:{
                    buffer: this.dPos[0]
                }
            },
            {
                binding: 2,
                resource:{
                    buffer: this.dVel
                }
            },
            {
                binding: 3,
                resource:{
                    buffer: this.dSortedPos
                }
            },
            {
                binding: 4,
                resource:{
                    buffer: this.dSortedVel
                }
            },
            {
                binding: 5,
                resource:{
                    buffer: this.dSortedDensity
                }
            },
            {
                binding: 6,
                resource:{
                    buffer: this.dGridParticleIndex
                }
            },
            {
                binding: 7,
                resource:{
                    buffer: this.dCellCountBufs[0]
                }
            }
        ];

        this.bind_group_update_position = engine_ctx.device.createBindGroup({ layout: bindGroupLayoutUpdatePosition, entries: group_entries_update_position});

        let group_entries_render = [
            {
                binding: 0,
                resource:{
                    buffer: this.dConstant
                }
            }
        ];

        this.bind_group_render = engine_ctx.device.createBindGroup({ layout: bindGroupLayoutRender, entries: group_entries_render});
    }

    update(height_target)
    {
        {
            const uniform = new Float32Array(32);
            const iuniform = new Int32Array(uniform.buffer);
            uniform[0] = this.global_min[0];
            uniform[1] = this.global_min[1];
            uniform[2] = this.global_min[2];
            uniform[4] = this.global_max[0];
            uniform[5] = this.global_max[1];
            uniform[6] = this.global_max[2];
            uniform[8] = 3.40282347e+38;
            uniform[9] = 3.40282347e+38;
            uniform[10] = 3.40282347e+38;
            uniform[12] = -3.40282347e+38;
            uniform[13] = -3.40282347e+38;
            uniform[14] = -3.40282347e+38;
            iuniform[16] = 0;
            iuniform[17] = 0;
            iuniform[18] = 0;
            iuniform[20] = this.numParticles;
            uniform[21] = this.particleRadius;
            iuniform[22] = 0;
            iuniform[23] = this.sizeGridBuf;
            uniform[24] = this.H;
            uniform[25] = this.particleMass;
            uniform[26] = this.time_step;
            uniform[27] = this.gas_const;
            uniform[28] = this.pg;
            uniform[29] = this.gravity;
            uniform[30] = this.pt;
            uniform[31] = this.pmin_sur_grad;

            engine_ctx.queue.writeBuffer(this.dConstant, 0, uniform.buffer, uniform.byteOffset, uniform.byteLength);
        }

        let commandEncoder = engine_ctx.device.createCommandEncoder();    
        ParticleReduction(commandEncoder, this);
        UpdateConstant(commandEncoder, this);
        ClearCellCount(commandEncoder, this);
        HashCount(commandEncoder, this);
        PrefixSum(commandEncoder, this);
        Scatter(commandEncoder, this);
        UpdateDensity(commandEncoder, this);
        UpdatePosition(commandEncoder, this, height_target);
         
        let cmdBuf = commandEncoder.finish();
        engine_ctx.queue.submit([cmdBuf]);

    }
}
