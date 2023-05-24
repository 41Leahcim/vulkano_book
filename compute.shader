// Version 4.60 of GLSL, set the GLSL version at the start of every shader
#version 460

// The number of invocations per dimension per work group (at least 32, at most 64)
// y and z are 1 for 1-dimensional arrays, but can be higher for multi-dimensional arrays
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

// A descriptor of a buffer with the name buf, which contains a uint (u32) array
layout(set = 0, binding = 0) buffer Data{
    uint data[];
} buf;

// Main function
void main(){
    // The index of the current invocation
    uint idx = gl_GlobalInvocationID.x;

    // Multiply the value at the current index with 12
    buf.data[idx] *= 12;
}

