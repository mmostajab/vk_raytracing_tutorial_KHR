#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable
#include "raycommon.glsl"

layout(location = 0) rayPayloadInEXT ddgiHitPayload prd;

layout(push_constant) uniform Constants
{
  vec4 clearColor;
};

void main()
{
  prd.irradiance = vec4(0.0f);
}
