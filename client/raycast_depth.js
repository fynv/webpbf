const BVH_STACK_SIZE = 32;
const SHARED_STACK_SIZE = 8;
const LOCAL_STACK_SIZE = (BVH_STACK_SIZE - SHARED_STACK_SIZE);

const shader_code = `
@group(0) @binding(0)
var<storage, read> bNodes : array<vec4f>;

@group(0) @binding(1)
var<storage, read> bIndices : array<u32>;

@group(0) @binding(2)
var<storage, read> bTriangles : array<vec4f>;

fn ray_get_octant_inv4(ray_direction: vec3f) -> u32
{
	return select(0x04040404u, 0u, ray_direction.x < 0.0) |
        select(0x02020202u, 0u, ray_direction.y < 0.0) |
        select(0x01010101u, 0u, ray_direction.z < 0.0);
}

struct BVH8Node
{
	node_0: vec4f,
	node_1: vec4f,
	node_2: vec4f,
	node_3: vec4f,
	node_4: vec4f 
};

struct Ray
{
	origin: vec3f,
	tmin: f32,
	direction: vec3f,
	tmax: f32
};

fn extract_byte(x: u32, i: u32) -> u32
{
	return (x >> (i * 8u)) & 0xffu;
}

fn sign_extend_s8x4(x: u32) -> u32
{
	return ((x >> 7u) & 0x01010101u) * 0xffu;
}

fn bvh8_node_intersect(ray: Ray, oct_inv4: u32, node: BVH8Node) -> u32
{
    let p = node.node_0.xyz;

    let e_imask = bitcast<u32>(node.node_0.w);
	let e_x = extract_byte(e_imask, 0u);
	let e_y = extract_byte(e_imask, 1u);
	let e_z = extract_byte(e_imask, 2u);

    let adjusted_ray_direction_inv = vec3(
		bitcast<f32>(e_x << 23u) / ray.direction.x,
		bitcast<f32>(e_y << 23u) / ray.direction.y,
		bitcast<f32>(e_z << 23u) / ray.direction.z
	);

    let adjusted_ray_origin = (p - ray.origin) / ray.direction;
    
    var hit_mask = 0u;
    for (var i = 0; i < 2; i++) 
	{
        let meta4 = bitcast<u32>(select(node.node_1.w, node.node_1.z, i==0));
        let is_inner4  = (meta4 & (meta4 << 1u)) & 0x10101010u;
        let inner_mask4 = sign_extend_s8x4(is_inner4 << 3u);
        let bit_index4  = (meta4 ^ (oct_inv4 & inner_mask4)) & 0x1f1f1f1fu;
        let child_bits4 = (meta4 >> 5u) & 0x07070707u;

        // Select near and far planes based on ray octant
		let q_lo_x = bitcast<u32>(select(node.node_2.y, node.node_2.x, i==0));
		let q_hi_x = bitcast<u32>(select(node.node_2.w, node.node_2.z, i==0));

        let q_lo_y = bitcast<u32>(select(node.node_3.y, node.node_3.x, i==0));
		let q_hi_y = bitcast<u32>(select(node.node_3.w, node.node_3.z, i==0));

        let q_lo_z = bitcast<u32>(select(node.node_4.y, node.node_4.x, i==0));
		let q_hi_z = bitcast<u32>(select(node.node_4.w, node.node_4.z, i==0));

        let x_min = select(q_lo_x, q_hi_x, ray.direction.x < 0.0);
		let x_max = select(q_hi_x, q_lo_x, ray.direction.x < 0.0);

		let y_min = select(q_lo_y, q_hi_y, ray.direction.y < 0.0);
		let y_max = select(q_hi_y, q_lo_y, ray.direction.y < 0.0);

		let z_min = select(q_lo_z, q_hi_z, ray.direction.z < 0.0);
		let z_max = select(q_hi_z, q_lo_z, ray.direction.z < 0.0);

        for (var j = 0u; j < 4u; j++) 
		{
            // Extract j-th byte
			var tmin3 = vec3(f32(extract_byte(x_min, j)), f32(extract_byte(y_min, j)), f32(extract_byte(z_min, j)));
			var tmax3 = vec3(f32(extract_byte(x_max, j)), f32(extract_byte(y_max, j)), f32(extract_byte(z_max, j)));

            // Account for grid origin and scale
			tmin3 = tmin3 * adjusted_ray_direction_inv + adjusted_ray_origin;
			tmax3 = tmax3 * adjusted_ray_direction_inv + adjusted_ray_origin;

            let tmin = max(max(tmin3.x, tmin3.y), max(tmin3.z, ray.tmin));
			let tmax = min(min(tmax3.x, tmax3.y), min(tmax3.z, ray.tmax));

            let intersected = tmin < tmax;
			if (intersected) 
			{
				let child_bits = extract_byte(child_bits4, j);
				let bit_index  = extract_byte(bit_index4,  j);
				hit_mask |= child_bits << bit_index;
			}
        }
    }
    return hit_mask;
}

struct Intersection
{
    hit: bool,
    t: f32,
    u: f32,
    v: f32
};

fn triangle_intersect(triangle_id: i32, ray: Ray) -> Intersection
{
	let pos0 = bTriangles[triangle_id*3].xyz;    
	let edge1 = bTriangles[triangle_id*3 + 1].xyz;    
	let edge2 = bTriangles[triangle_id*3 + 2].xyz;    
	
	let h = cross(ray.direction, edge2);
	let a = dot(edge1, h);

    var ret: Intersection;

	if (a==0.0) 
    {
        ret.hit = false;
        return ret;
    }
	
	let f = 1.0 / a;
	let s = ray.origin - pos0;   
	ret.u = f * dot(s, h);

	if (ret.u < 0.0 || ret.u > 1.0) 
    {
        ret.hit = false;
        return ret;
    }
	
	let q = cross(s, edge1);
	ret.v = f * dot(ray.direction, q);

	if (ret.v < 0.0 || (ret.u + ret.v)> 1.0) 
    {
        ret.hit = false;
        return ret;
    }

	ret.t = f * dot(edge2, q);
	if (ret.t <= ray.tmin) 
    {
        ret.hit = false;
        return ret;
    }
	
    ret.hit = ret.t <= ray.tmax;
    return ret;
}

var<private> threadIdx : u32;
var<workgroup> shared_stack_bvh8 : array<vec2u, ${SHARED_STACK_SIZE} * 64>;

fn SHARED_STACK_INDEX(offset: i32) -> i32
{
    let x = i32(threadIdx%32);
    let y = i32(threadIdx/32);
    return (y * ${SHARED_STACK_SIZE} + offset) * 32 + x;
}

var<private> g_ray : Ray;

fn intersect()
{
    var stack: array<vec2u, ${LOCAL_STACK_SIZE}>;
    var stack_size = 0;

    let oct_inv4 = ray_get_octant_inv4(g_ray.direction);
    var current_group = vec2(0u, 0x80000000u);

    while (stack_size > 0 || current_group.y!=0)
    {
        var triangle_group : vec2u;
        if ((current_group.y & 0xff000000u)!=0)
        {
            let hits_imask = current_group.y;
            let child_index_offset = 31u - countLeadingZeros(hits_imask);
            let child_index_base = current_group.x;

            // Remove n from current_group;
			current_group.y &= ~(1u << child_index_offset);

            // If the node group is not yet empty, push it on the stack
			if ((current_group.y & 0xff000000u)!=0) 
			{
				if (stack_size < ${SHARED_STACK_SIZE}) 
                {
                    shared_stack_bvh8[SHARED_STACK_INDEX(stack_size)] = current_group;
                } 
                else 
                {
                    stack[stack_size - ${SHARED_STACK_SIZE}] = current_group;
                }
                stack_size++;
			}

            let slot_index = (child_index_offset - 24u) ^ (oct_inv4 & 0xffu);
            let relative_index = countOneBits(hits_imask & ~(0xffffffffu << slot_index));
            let child_node_index = child_index_base + relative_index;
            var node: BVH8Node;
			node.node_0 = bNodes[child_node_index*5];         
			node.node_1 = bNodes[child_node_index*5 + 1];
			node.node_2 = bNodes[child_node_index*5 + 2];
			node.node_3 = bNodes[child_node_index*5 + 3];
			node.node_4 = bNodes[child_node_index*5 + 4];
			let hitmask = bvh8_node_intersect(g_ray, oct_inv4, node);

            let imask = extract_byte(bitcast<u32>(node.node_0.w), 3);	

            current_group.x = bitcast<u32>(node.node_1.x); // Child    base offset
			triangle_group.x = bitcast<u32>(node.node_1.y); // Triangle base offset

			current_group.y = (hitmask & 0xff000000u) | imask;
			triangle_group.y = (hitmask & 0x00ffffffu);
        }
        else 
		{
			triangle_group = current_group;
			current_group  = vec2(0u);
		}

        while (triangle_group.y != 0u)
		{
			let triangle_index = 31u - countLeadingZeros(triangle_group.y);
			triangle_group.y &= ~(1u << triangle_index);

			let tri_idx = i32(triangle_group.x + triangle_index);
			let intersection = triangle_intersect(tri_idx, g_ray);
			if (intersection.hit)
			{				
				g_ray.tmax = intersection.t;			
			}
		}			

		if ((current_group.y & 0xff000000u) == 0) 
		{
			if (stack_size == 0) 
            {
                break;
            }

            stack_size--;
            if (stack_size < ${SHARED_STACK_SIZE}) 
            {
                current_group =  shared_stack_bvh8[SHARED_STACK_INDEX(stack_size)];
            } 
            else 
            {
                current_group = stack[stack_size - ${SHARED_STACK_SIZE}];
            }			
		}
    }
}

struct Model
{
    modelMat: mat4x4f,
    normalMat: mat4x4f,
    inverseMat: mat4x4f,
};
@group(0) @binding(3)
var<uniform> uModel: Model;

var<private> g_id_io : u32;
var<private> g_origin: vec3f;
var<private> g_dir: vec3f;
var<private> g_tmin: f32;
var<private> g_tmax: f32;

fn render()
{
    let tmax = bDepth[g_id_io];
    g_tmax = min(tmax, g_tmax);

    g_ray.origin = g_origin;
    g_ray.direction = g_dir;
    g_ray.tmin = g_tmin;
    g_ray.tmax = g_tmax;

    intersect();

    if (g_ray.tmax < tmax)
    {
        bDepth[g_id_io] = g_ray.tmax;
    }
}

@group(1) @binding(0)
var<storage, read_write> bDepth : array<f32>;

struct Camera
{
    projMat: mat4x4f, 
    viewMat: mat4x4f,
    invProjMat: mat4x4f,
    invViewMat: mat4x4f,
    eyePos: vec4f,
    scissor: vec4f
};

@group(1) @binding(1)
var<uniform> uCamera: Camera;

struct View
{
    width: i32,
    height: i32
};

@group(1) @binding(2)
var<uniform> uView: View;

@compute @workgroup_size(64,1,1)
fn main(
    @builtin(local_invocation_id) LocalInvocationID : vec3<u32>,
    @builtin(workgroup_id) WorkgroupID : vec3<u32>)
{
    let bw = (uView.width + 7)/8;
    let x = i32(LocalInvocationID.x)%8 + i32(WorkgroupID.x)%bw * 8;
    let y = i32(LocalInvocationID.x)/8 + i32(WorkgroupID.x)/bw * 8;

    if (x>=uView.width || y>uView.height)
    {
        return;
    }

    threadIdx = LocalInvocationID.x;
        
    var uv = (vec2(f32(x),f32(y))+ 0.5)/vec2(f32(uView.width), f32(uView.height));  

    let clip0 = vec4(uv*2.0 - 1.0, -1.0, 1.0);
    let clip1 = vec4(uv*2.0 - 1.0, 1.0, 1.0);
    var view0 = uCamera.invProjMat *clip0; view0 /= view0.w;
    var view1 = uCamera.invProjMat *clip1; view1 /= view1.w;    
    let view_dir = normalize(view0.xyz);
    let world_dir = uCamera.invViewMat * vec4(view_dir, 0.0);
    let model_eye = (uModel.inverseMat*vec4(uCamera.eyePos.xyz, 1.0)).xyz;      
    let model_dir = (uModel.inverseMat*world_dir).xyz;    
    
    g_id_io = u32(x+y*uView.width); 
    g_origin = model_eye;
    g_dir = model_dir;
    g_tmin = length(view0.xyz);
    g_tmax = length(view1.xyz);

    render();
}
`;


function GetPipeline()
{
    if (!("raycast_depth" in engine_ctx.cache.pipelines))
    {
        let shaderModule = engine_ctx.device.createShaderModule({ code: shader_code });
        let bindGroupLayouts = [engine_ctx.cache.bindGroupLayouts.cwbvh, engine_ctx.cache.bindGroupLayouts.raycast0];
        const pipelineLayoutDesc = { bindGroupLayouts };
        let layout = engine_ctx.device.createPipelineLayout(pipelineLayoutDesc);
        
        engine_ctx.cache.pipelines.raycast_depth = engine_ctx.device.createComputePipeline({
            layout,
            compute: {
                module: shaderModule,
                entryPoint: 'main',
            },
        });
    }
    return engine_ctx.cache.pipelines.raycast_depth;
}

export function RaycastDepth(commandEncoder, cwbvh, target)
{
    let bw = (target.width + 7)/8;
    let bh = (target.height + 7)/8;

    let pipeline = GetPipeline();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, cwbvh.bind_group);
    passEncoder.setBindGroup(1, target.bind_group_raycast);
    passEncoder.dispatchWorkgroups(bw*bh, 1,1); 
    passEncoder.end();

}

