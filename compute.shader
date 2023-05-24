// Version 4.60 of GLSL, set the GLSL version at the start of every shader
#version 460

// The number of invocations per dimension per work group (at least 32, at most 64)
// z is 1 for 2-dimensional arrays, but can be higher for 3-dimensional arrays
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// A descriptor of an image2D with the name img, which has 4 channels of 8 bits per pixel
// Bound as the first binding of the first descriptor set
layout(set = 0, binding = 0, rgba8) uniform writeonly image2D img;

// Main function
void main(){
    // Normalize the coordinates
    vec2 norm_coordinates = (gl_GlobalInvocationID.xy + vec2(0.5)) / vec2(imageSize(img));

    // Calculate complex number that corresponds to the pixel of the image to modify
    vec2 c = (norm_coordinates - vec2(0.5)) * 2.0 - vec2(1.0, 0.0);

    // Search for the complex number in the mandelbrot set
    // It is in the mandelbrot set if z² + c diverges when iterated from z = 0 (z being a complex number)
    // It's diverging if length(z) > 4.0
    vec2 z = vec2(0.0, 0.0);
    float i;
    for(i = 0.0;i < 1.0;i += 0.005){
        z = vec2(
            z.x * z.x - z.y * z.y + c.x,
            z.y * z.x + z.x * z.y + c.y
        );
        if(length(z) > 4.0){
            break;
        }
    }

    // The closer c is to the set, the higher i will be, so use i for the current pixel
    vec4 to_write = vec4(vec3(i), 1.0);

    // Write to_write to the pixel with imageStore, to make sure correct type is written to the pixel
    imageStore(img, ivec2(gl_GlobalInvocationID.xy), to_write);
}
