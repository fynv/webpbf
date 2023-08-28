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

const Z_THRESHOLD = 5.0;
const SMOOTH_DT = 0.0003;

fn depthToEyeSpaceZ(depth: f32) -> f32
{
    let ndc = 2.0 * depth - 1.0;
    let z = uCamera.invProjMat[2][2] * ndc + uCamera.invProjMat[3][2];
    let w = uCamera.invProjMat[2][3] * ndc + uCamera.invProjMat[3][3];
    return z/w;
}


fn depthToEyeSpaceZ_V(depth: vec4f) -> vec4f
{
    let ndc = 2.0 * depth - 1.0;
    let z = uCamera.invProjMat[2][2] * ndc + uCamera.invProjMat[3][2];
    let w = uCamera.invProjMat[2][3] * ndc + uCamera.invProjMat[3][3];
    return z/w;
}

struct FSOut
{
    @builtin(frag_depth) depth: f32,
};

@fragment
fn fs_main(@builtin(position) coord_pix: vec4f) -> FSOut
{
    let icoord2d = vec2i(coord_pix.xy);
    let res = vec2i(textureDimensions(uTex));
    let z = textureLoad(uTex, icoord2d, 0).x;
    if (z >=1.0) 
    {
        discard;
    }

    // Direct neighbors (dz/dx)
    let right = textureLoad(uTex, icoord2d + vec2(1,0), 0).x;
    let left = textureLoad(uTex, icoord2d + vec2(-1,0), 0).x;
    let top = textureLoad(uTex, icoord2d + vec2(0,-1), 0).x;
    let bottom = textureLoad(uTex, icoord2d + vec2(0,1), 0).x;

    let eyeZ = depthToEyeSpaceZ(z);
    let neighborEyeZ= depthToEyeSpaceZ_V(vec4(right, left, top, bottom));
    let zDiff = abs(neighborEyeZ - eyeZ);

    if (any(zDiff > vec4(Z_THRESHOLD)))
    {
        var output: FSOut;
        output.depth = z;
        return output;
    }

    // Gradient (first derivative) with border handling
    let dzdx = select(0.5 * (right - left), 0.0,  
        icoord2d.x <= 1 || icoord2d.x >= res.x - 1 || right == 1.0 || left == 1.0);    

    let dzdy = select(0.5 * (top - bottom), 0.0,
        icoord2d.y <= 1 || icoord2d.y >= res.y - 1 || top == 1.0 || bottom == 1.0);

    // Diagonal neighbors
    let topRight  = textureLoad(uTex, icoord2d + vec2(1,-1), 0).x;
    let bottomLeft  = textureLoad(uTex, icoord2d + vec2(-1,1), 0).x;
    let bottomRight  = textureLoad(uTex, icoord2d + vec2(1,1), 0).x;
    let topLeft  = textureLoad(uTex, icoord2d + vec2(-1,-1), 0).x;

    // Use central difference (for better results)
    let dzdxy = (topRight + bottomLeft - bottomRight - topLeft) * 0.25;

    // Equation (3)
    let Fx = -uCamera.projMat[0][0]; // 2n / (r-l)
    let Fy = -uCamera.projMat[1][1]; // 2n / (t-b)
    let Cx = 2.0 / (f32(res.x) * Fx);
    let Cy = 2.0 / (f32(res.y) * Fy);
    let Cy2 = Cy * Cy;
    let Cx2 = Cx * Cx;

    // Equation (5)
    let D = Cy2 * (dzdx * dzdx) + Cx2 * (dzdy * dzdy) + Cx2 * Cy2 * (z * z);
    let dzdx2 = right + left - z * 2.0;
    let dzdy2 = top + bottom - z * 2.0;
    let dDdx = 2.0 * Cy2 * dzdx * dzdx2 + 2.0 * Cx2 * dzdy * dzdxy + 2.0 * Cx2 * Cy2 * z * dzdx;
    let dDdy = 2.0 * Cy2 * dzdx * dzdxy + 2.0 * Cx2 * dzdy * dzdy2 + 2.0 * Cx2 * Cy2 * z * dzdy;

   // Mean Curvature (7)(8)(6)
    let Ex = 0.5 * dzdx * dDdx - dzdx2 * D;
    let Ey = 0.5 * dzdy * dDdy - dzdy2 * D;
    let H2 = (Cy * Ex + Cx * Ey) / pow(D, 3.0 / 2.0);

    var output: FSOut;
    output.depth = clamp(z + (0.5 * H2) * SMOOTH_DT, 0.001, 0.999);
    return output;
}
`;


function GetPipeline()
{
    if (!("curvature_flow" in engine_ctx.cache.pipelines))
    {
        let camera_options = { has_reflector: false };
        let camera_signature =  JSON.stringify(camera_options);
        let camera_layout = engine_ctx.cache.bindGroupLayouts.perspective_camera[camera_signature];

        const pipelineLayoutDesc = { bindGroupLayouts: [ camera_layout, engine_ctx.cache.bindGroupLayouts.particle_depth] };
        let layout = engine_ctx.device.createPipelineLayout(pipelineLayoutDesc);
        let shaderModule = engine_ctx.device.createShaderModule({ code: shader_code });

        const depthStencil = {
            depthWriteEnabled: true,
            depthCompare: 'less-equal',
            format: 'depth32float'
        };

        const vertex = {
            module: shaderModule,
            entryPoint: 'vs_main',
            buffers: []
        };

        const fragment = {
            module: shaderModule,
            entryPoint: 'fs_main',
            targets: []
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
    
            primitive,
            depthStencil
        };

        engine_ctx.cache.pipelines.curvature_flow = engine_ctx.device.createRenderPipeline(pipelineDesc);
    }

    return engine_ctx.cache.pipelines.curvature_flow;

}


export function CurvatureFlow(passEncoder, camera, bind_group_frame)
{
    let pipeline = GetPipeline();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, camera.bind_group);
    passEncoder.setBindGroup(1, bind_group_frame);
    passEncoder.draw(3, 1);
}

