#version 440
layout(location = 0) in vec3 lineColor;
layout(location = 0) out vec4 fragColor;
void main()
{
    fragColor = vec4(lineColor, 1.0);
}
