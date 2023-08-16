const shader_code = `
@group(0) @binding(0)
var<storage, read_write> bConst : array<i32>;

@compute @workgroup_size(1,1,1)
fn main()
{
    let particle_radius = bitcast<f32>(bConst[21]);
    let h = bitcast<f32>(bConst[24]);
    let grid_min = vec3(bitcast<f32>(bConst[8]),bitcast<f32>(bConst[9]), bitcast<f32>(bConst[10]));
    let grid_max = vec3(bitcast<f32>(bConst[12]),bitcast<f32>(bConst[13]), bitcast<f32>(bConst[14])) + vec3(particle_radius*2.0);
    bConst[12]= bitcast<i32>(grid_max.x);
    bConst[13]= bitcast<i32>(grid_max.y);
    bConst[14]= bitcast<i32>(grid_max.z);

    let grid_div = vec3i(ceil((grid_max - grid_min)/h));
    bConst[16]= grid_div.x;
    bConst[17]= grid_div.y;
    bConst[18]= grid_div.z;

    let num_cells = grid_div.x * grid_div.y * grid_div.z;
    bConst[22]= i32(num_cells);

    let count0 = num_cells;
    let num_groups0 = i32((count0 + 127)/128); 
    bConst[32] = num_groups0;
    bConst[33] = 1;
    bConst[34] = 1;
    bConst[35] = 0;
    let count1 = max(0, num_groups0 - 1);
    let num_groups1 = i32((count1 + 127)/128); 
    bConst[36] = num_groups1;
    bConst[37] = 1;
    bConst[38] = 1;
    bConst[39] = 0;
    let count2 = max(0, num_groups1 - 1);
    let num_groups2 = i32((count2 + 127)/128); 
    bConst[40] = num_groups2;
    bConst[41] = 1;
    bConst[42] = 1;
    bConst[43] = 0;

    {
        let num_groups = max(0, i32((count0 + 63)/64) - 2);
        bConst[44] = num_groups;
        bConst[45] = 1;
        bConst[46] = 1;
        bConst[47] = 0;
    }

    {
        let num_groups = max(0, i32((count1 + 63)/64) - 2);
        bConst[48] = num_groups;
        bConst[49] = 1;
        bConst[50] = 1;
        bConst[51] = 0;
    }
}
`;

function GetPipeline()
{
    if (!("update_const" in engine_ctx.cache.pipelines))
    {
        let shaderModule = engine_ctx.device.createShaderModule({ code: shader_code });
        let bindGroupLayouts = [engine_ctx.cache.bindGroupLayouts.update_const];
        const pipelineLayoutDesc = { bindGroupLayouts };
        let layout = engine_ctx.device.createPipelineLayout(pipelineLayoutDesc);
        
        engine_ctx.cache.pipelines.update_const = engine_ctx.device.createComputePipeline({
            layout,
            compute: {
                module: shaderModule,
                entryPoint: 'main',
            },
        });
    }
    return engine_ctx.cache.pipelines.update_const;
}


export function UpdateConstant(commandEncoder, psystem)
{
    const passEncoder = commandEncoder.beginComputePass();

    {
        let pipeline = GetPipeline();        
        passEncoder.setPipeline(pipeline);
        passEncoder.setBindGroup(0, psystem.bind_group_update_constant);
        passEncoder.dispatchWorkgroups(1, 1,1); 
    }

    passEncoder.end();

}
