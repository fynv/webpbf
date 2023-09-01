cmake_minimum_required (VERSION 3.0)
project(CWBVH)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s EXPORTED_RUNTIME_METHODS=\"['ccall', 'cwrap']\" -s EXPORT_ES6=1 -s ALLOW_MEMORY_GROWTH=1")

set(INCLUDE_DIR
.
./glm
)

include_directories(${INCLUDE_DIR})


add_executable(CWBVH api.cpp AABB.cpp BVH.cpp BVH8Converter.cpp BVHPartitions.cpp SAHBuilder.cpp Triangle.cpp)
 
