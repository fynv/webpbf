const shader_code =`
struct Params
{
    globalMin: vec4f,
    globalMax: vec4f,
    gridMin: vec4f,
    gridMax: vec4f,
    gridDiv: vec4i,
    numParticles: u32,
    particleRadius: f32,
    numGridCells: u32,
    sizeGridBuf: u32,
    h: f32,
    particleMass: f32,
    time_step: f32,
    gas_const: f32,
    pg: f32,
    gravity: f32,
    pt: f32,
    pmin_sur_grad: f32
};

@group(0) @binding(0)
var<uniform> uParams : Params;

struct Model
{
    modelMat: mat4x4f,
    normalMat: mat4x4f
};
@group(1) @binding(0)
var<uniform> uModel: Model;

@vertex
fn vs_main(@location(0) aPos: vec3f) -> @builtin(position) vec4f
{
    let world_pos = uModel.modelMat*vec4(aPos, 1.0);
    var uvz = (world_pos.xzy - uParams.globalMin.xzy)/(uParams.globalMax.xzy - uParams.globalMin.xzy);
    uvz.y = 1.0 - uvz.y;
    return vec4(uvz.xy * 2.0 - 1.0, uvz.z, 1.0);
}

@fragment
fn fs_main()
{

}
`;


function GetPipelineHeight(options)
{
    let signature = JSON.stringify(options);
    if (!("height" in engine_ctx.cache.pipelines))
    {
        engine_ctx.cache.pipelines.height = {};
    }

    if (!(signature in engine_ctx.cache.pipelines.height))
    {
        let prim_options = {
            material: options.material_options,
            has_lightmap: options.has_lightmap,
            has_reflector: options.has_reflector,
            has_envmap: options.has_primtive_probe
        };
        let prim_signature = JSON.stringify(prim_options);
        let primitive_layout = engine_ctx.cache.bindGroupLayouts.primitive[prim_signature];

        let bindGroupLayouts = [engine_ctx.cache.bindGroupLayouts.particle_render, primitive_layout];

        const pipelineLayoutDesc = { bindGroupLayouts };
        let layout = engine_ctx.device.createPipelineLayout(pipelineLayoutDesc);        
        let shaderModule = engine_ctx.device.createShaderModule({ code: shader_code });

        const depthStencil = {
            depthWriteEnabled: true,
            depthCompare: 'greater-equal',
            format: 'depth32float'
        };

        let vertex_bufs = [];

        const positionAttribDesc = {
            shaderLocation: 0,
            offset: 0,
            format: 'float32x4'
        };

        const positionBufferDesc = {
            attributes: [positionAttribDesc],
            arrayStride: 4 * 4,
            stepMode: 'vertex'
        };

        vertex_bufs.push(positionBufferDesc);

        const vertex = {
            module: shaderModule,
            entryPoint: 'vs_main',
            buffers: vertex_bufs
        };

        const fragment = {
            module: shaderModule,
            entryPoint: 'fs_main',
            targets: []
        };

        const primitive = {
            frontFace: 'ccw',
            cullMode:  options.material_options.doubleSided ? "none" : "back",
            topology: 'triangle-list'
        };

        const pipelineDesc = {
            layout,
    
            vertex,
            fragment,
    
            primitive,
            depthStencil
        };

        engine_ctx.cache.pipelines.height[signature] = engine_ctx.device.createRenderPipeline(pipelineDesc); 
    }
    return engine_ctx.cache.pipelines.height[signature];    
}


export function RenderHeight(passEncoder, primitive, psystem)
{
    let index_type_map = { 1: 'uint8', 2: 'uint16', 4: 'uint32'};            
    let geo = primitive.geometry[primitive.geometry.length - 1]; 

    let options = {};        
    options.has_lightmap = primitive.has_lightmap;
    options.has_reflector = primitive.has_reflector;
    options.material_options = primitive.material_options;
    options.has_primtive_probe = primitive.envMap!=null;

    let pipeline = GetPipelineHeight(options);
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, psystem.bind_group_render);
    passEncoder.setBindGroup(1, primitive.bind_group); 
    passEncoder.setVertexBuffer(0, geo.pos_buf);

    if (primitive.index_buf!=null)
    {
        passEncoder.setIndexBuffer(primitive.index_buf, index_type_map[primitive.type_indices]);        
        passEncoder.drawIndexed(primitive.num_face * 3, 1);
    }
    else
    {
        passEncoder.draw(primitive.num_pos, 1);
    }
}