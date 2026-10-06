#version 450
#extension GL_EXT_samplerless_texture_functions : require

layout(push_constant) uniform CopyControl { uint sample_index; } copy_control;

#if COPY_STENCIL
#if COPY_MS
layout(binding = 0) uniform utexture2DMS source_image;
#else
layout(binding = 0) uniform utexture2D source_image;
#endif
layout(location = 0) out uvec4 color;
#else
#if COPY_MS
layout(binding = 0) uniform texture2DMS source_image;
#else
layout(binding = 0) uniform texture2D source_image;
#endif
layout(location = 0) out vec4 color;
#endif

void main() {
	ivec2 coord = ivec2(gl_FragCoord.xy);
#if COPY_MS
	color = texelFetch(source_image, coord, int(copy_control.sample_index));
#else
	color = texelFetch(source_image, coord, 0);
#endif
}
