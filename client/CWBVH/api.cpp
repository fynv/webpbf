#include <memory.h>
#include <emscripten.h>

extern "C"
{
	EMSCRIPTEN_KEEPALIVE void* alloc(unsigned size);
	EMSCRIPTEN_KEEPALIVE void dealloc(void* ptr);
	EMSCRIPTEN_KEEPALIVE void zero(void* ptr, unsigned size);

    EMSCRIPTEN_KEEPALIVE void *CreateCWBVH(int num_face, int num_pos, const void* p_indices, int type_indices, const void* p_pos);
    EMSCRIPTEN_KEEPALIVE void DestroyCWBVH(void* p_bvh);
    EMSCRIPTEN_KEEPALIVE int CWBVH_NumNodes(void* p_bvh);
    EMSCRIPTEN_KEEPALIVE int CWBVH_NumTriangles(void* p_bvh);
    EMSCRIPTEN_KEEPALIVE void* CWBVH_Nodes(void* p_bvh);
    EMSCRIPTEN_KEEPALIVE void* CWBVH_Indices(void* p_bvh);
    EMSCRIPTEN_KEEPALIVE void* CWBVH_Triangles(void* p_bvh);
}

void* alloc(unsigned size)
{
	return malloc(size);
}

void dealloc(void* ptr)
{
	free(ptr);
}

void zero(void* ptr, unsigned size)
{
	memset(ptr, 0, size);
}


#include "BVH.h"
#include "BVH8Converter.h"

class CWBVH
{
public:
    CWBVH(int num_face, int num_pos, const void* p_indices, int type_indices, const glm::vec3* p_pos);

    flex_bvh::BVH8 m_bvh8;
    std::vector<glm::vec4> m_triangles;
};

template <typename T>
inline void t_get_indices(const T* indices, int face_id, unsigned& i0, unsigned& i1, unsigned& i2)
{
	i0 = indices[face_id * 3];
	i1 = indices[face_id * 3 + 1];
	i2 = indices[face_id * 3 + 2];
}

inline void get_indices(const void* indices, int type_indices, int face_id, unsigned& i0, unsigned& i1, unsigned& i2)
{
	if (type_indices == 1)
	{
		t_get_indices((const uint8_t*)indices, face_id, i0, i1, i2);
	}
	else if (type_indices == 2)
	{
		t_get_indices((const uint16_t*)indices, face_id, i0, i1, i2);
	}
	else if (type_indices == 4)
	{
		t_get_indices((const uint32_t*)indices, face_id, i0, i1, i2);
	}
}

CWBVH::CWBVH(int num_face, int num_pos, const void* p_indices, int type_indices, const glm::vec3* p_pos)
{
    std::vector<flex_bvh::Triangle> triangles;

    if (p_indices!=nullptr)
    {
        for (int i=0; i< num_face; i++)
        {
            unsigned i0, i1, i2;
			get_indices(p_indices, type_indices, i, i0, i1, i2);

            glm::vec4 v0, v1, v2;
			v0 = glm::vec4(p_pos[i0], 1.0f);
			v1 = glm::vec4(p_pos[i1], 1.0f);
			v2 = glm::vec4(p_pos[i2], 1.0f);
            triangles.emplace_back(flex_bvh::Triangle(v0, v1, v2));
        }
    }
    else
	{
		for (int i = 0; i < num_pos / 3; i++)
		{
			glm::vec4 v0, v1, v2;
            v0 = glm::vec4(p_pos[i * 3], 1.0f);
			v1 = glm::vec4(p_pos[i * 3 + 1], 1.0f);
			v2 = glm::vec4(p_pos[i * 3 + 2], 1.0f);
			triangles.emplace_back(flex_bvh::Triangle(v0, v1, v2));
		}
	}

	flex_bvh::BVH2 bvh2;
	bvh2.create_from_triangles(triangles);	
	flex_bvh::ConvertBVH2ToBVH8(bvh2, m_bvh8);

    m_triangles.resize(m_bvh8.indices.size() * 3);
    for (size_t i = 0; i < m_bvh8.indices.size(); i++)
	{
		int index = m_bvh8.indices[i];
		const flex_bvh::Triangle& tri = triangles[index];
		m_triangles[i * 3] = glm::vec4(tri.position_0, 1.0f);
		m_triangles[i * 3 + 1] = glm::vec4(tri.position_1 - tri.position_0, 0.0f);
		m_triangles[i * 3 + 2] = glm::vec4(tri.position_2 - tri.position_0, 0.0f);
	}

}

void *CreateCWBVH(int num_face, int num_pos, const void* p_indices, int type_indices, const void* p_pos)
{
    const glm::vec3* pos = (const glm::vec3*)p_pos;
    return new CWBVH(num_face, num_pos, p_indices, type_indices, pos);    
}

void DestroyCWBVH(void* p_bvh)
{
    delete (CWBVH*)p_bvh;
}

int CWBVH_NumNodes(void* p_bvh)
{
    CWBVH* bvh = (CWBVH*)p_bvh;
    return (int)bvh->m_bvh8.nodes.size();
}

int CWBVH_NumTriangles(void* p_bvh)
{
    CWBVH* bvh = (CWBVH*)p_bvh;
    return (int)bvh->m_bvh8.indices.size();
}

void* CWBVH_Nodes(void* p_bvh)
{
    CWBVH* bvh = (CWBVH*)p_bvh;
    return bvh->m_bvh8.nodes.data();
}

void* CWBVH_Indices(void* p_bvh)
{
    CWBVH* bvh = (CWBVH*)p_bvh;
    return bvh->m_bvh8.indices.data();
}

void* CWBVH_Triangles(void* p_bvh)
{
    CWBVH* bvh = (CWBVH*)p_bvh;
    return bvh->m_triangles.data();
}
