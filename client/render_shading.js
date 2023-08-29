const shader_code =`
struct Camera
{
    projMat: mat4x4f, 
    viewMat: mat4x4f,
    invProjMat: mat4x4f,
    invViewMat: mat4x4f,
    eyePos: vec3f
};

@group(0) @binding(0)
var<uniform> uCamera: Camera;

@group(1) @binding(0)
var uTex: texture_2d<f32>;

struct VSOut 
{
    @builtin(position) Position: vec4f,    
};

@vertex
fn vs_main(@builtin(vertex_index) vertId: u32) -> VSOut
{
    var vsOut: VSOut;
    let grid = vec2(f32((vertId<<1)&2), f32(vertId & 2));
    let pos_proj = grid * vec2(2.0, 2.0) + vec2(-1.0, -1.0);        
    vsOut.Position = vec4(pos_proj, 0.0, 1.0);
    return vsOut;
}

fn fetch_pos(id: vec2i) -> vec4f
{
    let size = textureDimensions(uTex);
    var UV = (vec2f(id)+0.5)/vec2f(size);
    UV.y = 1.0 - UV.y;
    let depth = textureLoad(uTex, id, 0).x; 
    let pos_clip = vec3(UV*2.0 -1.0, depth);
    var pos_view = uCamera.invProjMat * vec4(pos_clip, 1.0);
    pos_view *= 1.0/pos_view.w;
    return vec4(pos_view.xyz, depth);
}

fn MinDiff(P: vec3f, Pr: vec3f, Pl: vec3f) -> vec3f
{
    let V1 = Pr - P;
    let V2 = P - Pl;
    return select(V2, V1, dot(V1,V1) < dot(V2,V2));
}


fn ReconstructNormal(id: vec2i, P: vec3f) -> vec3f
{
    var Pr = fetch_pos(id + vec2(1, 0));
    var Pl = fetch_pos(id + vec2(-1, 0));
    var Pt = fetch_pos(id + vec2(0, -1));
    var Pb = fetch_pos(id + vec2(0, 1));
    return normalize(cross(MinDiff(P, Pr.xyz, Pl.xyz), MinDiff(P, Pt.xyz, Pb.xyz)));
}

struct EnvironmentMap
{
    SHCoefficients: array<vec4f, 9>,
    diffuseThresh: f32,
    diffuseHigh: f32,
    diffuseLow: f32,
    specularThresh: f32,
    specularHigh: f32,
    specularLow: f32,
};

@group(2) @binding(0)
var<uniform> uIndirectLight: EnvironmentMap;

fn getIrradiance(world_pos: vec3f, normal: vec3f) -> vec3f
{
    let x = normal.x;
    let y = normal.y;
    let z = normal.z;

    // band 0
    var result = uIndirectLight.SHCoefficients[0].xyz * 0.886227;

    // band 1
	result += uIndirectLight.SHCoefficients[1].xyz * 2.0 * 0.511664 * y;
	result += uIndirectLight.SHCoefficients[2].xyz * 2.0 * 0.511664 * z;
	result += uIndirectLight.SHCoefficients[3].xyz * 2.0 * 0.511664 * x;

    // band 2
	result += uIndirectLight.SHCoefficients[4].xyz * 2.0 * 0.429043 * x * y;
	result += uIndirectLight.SHCoefficients[5].xyz * 2.0 * 0.429043 * y * z;
	result += uIndirectLight.SHCoefficients[6].xyz * ( 0.743125 * z * z - 0.247708 );
	result += uIndirectLight.SHCoefficients[7].xyz * 2.0 * 0.429043 * x * z;
	result += uIndirectLight.SHCoefficients[8].xyz * 0.429043 * ( x * x - y * y );

    return result;
}

@group(2) @binding(1)
var uReflectionMap: texture_cube<f32>;

fn getReflRadiance(reflectVec: vec3f, roughness: f32) -> vec3f
{
    var gloss : f32;
    if (roughness < 0.053)
    {
        gloss = 1.0;        
    }
    else
    {
        let r2 = roughness * roughness;
        let r4 = r2*r2;
        gloss = log(2.0/r4 - 1.0)/log(2.0)/18.0;
    }
    let mip = (1.0-gloss)*6.0;
    return textureSampleLevel(uReflectionMap, uSampler, reflectVec, mip).xyz;
}

@group(2) @binding(2)
var uSampler: sampler;

@fragment
fn fs_main(@builtin(position) coord_pix: vec4f) -> @location(0) vec4f
{
    let icoord2d = vec2i(coord_pix.xy);
    let ViewPosition = fetch_pos(icoord2d);
    if (ViewPosition.w>=1.0)
    {
        discard;
    }
    let ViewNormal = ReconstructNormal(icoord2d, ViewPosition.xyz);   

    let WorldPos =  uCamera.invViewMat * vec4(ViewPosition.xyz, 1.0);
    let WorldNormal = uCamera.invViewMat * vec4(ViewNormal, 0.0);

    let viewDir = normalize(uCamera.eyePos.xyz - WorldPos.xyz);
    let reflectVec = reflect(-viewDir, WorldNormal.xyz);

    let col = getReflRadiance(reflectVec, 0.0);    
        
    return vec4(col * 0.3, 0.0);
}
`;

function GetPipeline(view_format)
{
    if (!("render_particle_shading" in engine_ctx.cache.pipelines))
    {
        let camera_options = { has_reflector: false };
        let camera_signature =  JSON.stringify(camera_options);
        let camera_layout = engine_ctx.cache.bindGroupLayouts.perspective_camera[camera_signature];

        const pipelineLayoutDesc = { bindGroupLayouts: [ camera_layout, engine_ctx.cache.bindGroupLayouts.particle_depth, engine_ctx.cache.bindGroupLayouts.envmap] };
        let layout = engine_ctx.device.createPipelineLayout(pipelineLayoutDesc);
        let shaderModule = engine_ctx.device.createShaderModule({ code: shader_code });

        const vertex = {
            module: shaderModule,
            entryPoint: 'vs_main',
            buffers: []
        };

        const colorState = {
            format:  view_format,           
            blend: {
                color: {
                    srcFactor: "one",
                    dstFactor: "one-minus-src-alpha"
                },
                alpha: {
                    srcFactor: "one",
                    dstFactor: "one-minus-src-alpha"
                }
            },
            writeMask: GPUColorWrite.ALL
        };

        const fragment = {
            module: shaderModule,
            entryPoint: 'fs_main',
            targets: [colorState]
        };

        const primitive = {
            frontFace: 'cw',
            cullMode: 'none',
            topology: 'triangle-list'
        };

        const pipelineDesc = {
            layout,
    
            vertex,
            fragment,
    
            primitive
        };

        engine_ctx.cache.pipelines.render_particle_shading = engine_ctx.device.createRenderPipeline(pipelineDesc);
    }

    return engine_ctx.cache.pipelines.render_particle_shading;

}

export function RenderShading(passEncoder, camera, bind_group_frame, bind_group_light, target)
{
    let pipeline = GetPipeline(target.view_format);

    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, camera.bind_group);
    passEncoder.setBindGroup(1, bind_group_frame);
    passEncoder.setBindGroup(2, bind_group_light);
    passEncoder.draw(3, 1);
}
