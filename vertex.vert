// Version 4.60 of GLSL, set the GLSL version at the start of every shader
#version 460

layout(location = 0) in vec2 position;

void main(){
    gl_Position = vec4(position, 0.0, 1.0);
}
