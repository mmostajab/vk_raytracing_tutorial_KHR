#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable
#include "raycommon.glsl"
#include "wavefront.glsl"

hitAttributeEXT vec2 attribs;

// clang-format off
layout(location = 0) rayPayloadInEXT hitPayload prd;
layout(location = 1) rayPayloadEXT bool isShadowed;

layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;

layout(binding = 2, set = 1, scalar) buffer ScnDesc { sceneDesc i[]; } scnDesc;
layout(binding = 5, set = 1, scalar) buffer Vertices { Vertex v[]; } vertices[];
layout(binding = 6, set = 1) buffer Indices { uint i[]; } indices[];

layout(binding = 1, set = 1, scalar) buffer MatColorBufferObject { WaveFrontMaterial m[]; } materials[];
layout(binding = 3, set = 1) uniform sampler2D textureSamplers[];
layout(binding = 4, set = 1)  buffer MatIndexColorBuffer { int i[]; } matIndex[];

// clang-format on

layout(binding = 8, set = 1) uniform CommonConstants
{
  vec4 pointsOnSphereCosDist[512];
}
commonConstants;

layout(push_constant) uniform Constants
{
  vec4  clearColor;
  vec3  lightPosition;
  float lightIntensity;
  vec3  lightDirection;
  float lightSpotCutoff;
  float lightSpotOuterCutoff;
  int   lightType;
  int   frame;
  int   giMode;
}
pushC;

layout(location = 3) callableDataEXT rayLight cLight;


void main()
{
  // Object of this instance
  uint objId = scnDesc.i[gl_InstanceCustomIndexEXT].objId;

  // Indices of the triangle
  ivec3 ind = ivec3(indices[nonuniformEXT(objId)].i[3 * gl_PrimitiveID + 0],   //
                    indices[nonuniformEXT(objId)].i[3 * gl_PrimitiveID + 1],   //
                    indices[nonuniformEXT(objId)].i[3 * gl_PrimitiveID + 2]);  //
  // Vertex of the triangle
  Vertex v0 = vertices[nonuniformEXT(objId)].v[ind.x];
  Vertex v1 = vertices[nonuniformEXT(objId)].v[ind.y];
  Vertex v2 = vertices[nonuniformEXT(objId)].v[ind.z];

  const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

  // Computing the normal at hit position
  vec3 normal = v0.nrm * barycentrics.x + v1.nrm * barycentrics.y + v2.nrm * barycentrics.z;
  // Transforming the normal to world space
  normal = normalize(vec3(scnDesc.i[gl_InstanceCustomIndexEXT].transfoIT * vec4(normal, 0.0)));


  // Computing the coordinates of the hit position
  vec3 worldPos = v0.pos * barycentrics.x + v1.pos * barycentrics.y + v2.pos * barycentrics.z;
  // Transforming the position to world space
  worldPos = vec3(scnDesc.i[gl_InstanceCustomIndexEXT].transfo * vec4(worldPos, 1.0));

  cLight.inHitPosition = worldPos;
//#define DONT_USE_CALLABLE
#if defined(DONT_USE_CALLABLE)
  // Point light
  if(pushC.lightType == 0)
  {
    vec3  lDir              = pushC.lightPosition - cLight.inHitPosition;
    float lightDistance     = length(lDir);
    cLight.outIntensity     = pushC.lightIntensity / (lightDistance * lightDistance);
    cLight.outLightDir      = normalize(lDir);
    cLight.outLightDistance = lightDistance;
  }
  else if(pushC.lightType == 1)
  {
    vec3 lDir               = pushC.lightPosition - cLight.inHitPosition;
    cLight.outLightDistance = length(lDir);
    cLight.outIntensity =
        pushC.lightIntensity / (cLight.outLightDistance * cLight.outLightDistance);
    cLight.outLightDir  = normalize(lDir);
    float theta         = dot(cLight.outLightDir, normalize(-pushC.lightDirection));
    float epsilon       = pushC.lightSpotCutoff - pushC.lightSpotOuterCutoff;
    float spotIntensity = clamp((theta - pushC.lightSpotOuterCutoff) / epsilon, 0.0, 1.0);
    cLight.outIntensity *= spotIntensity;
  }
  else  // Directional light
  {
    cLight.outLightDir      = normalize(-pushC.lightDirection);
    cLight.outIntensity     = 1.0;
    cLight.outLightDistance = 10000000;
  }
#else
  executeCallableEXT(pushC.lightType, 3);
#endif

  // Material of the object
  int               matIdx = matIndex[nonuniformEXT(objId)].i[gl_PrimitiveID];
  WaveFrontMaterial mat    = materials[nonuniformEXT(objId)].m[matIdx];


  // Diffuse
  vec3 diffuse = computeDiffuse(mat, cLight.outLightDir, normal, pushC.giMode == 0);
  vec3 albedo = mat.diffuse;
  if(mat.textureId >= 0)
  {
    uint txtId = mat.textureId + scnDesc.i[gl_InstanceCustomIndexEXT].txtOffset;
    vec2 texCoord =
        v0.texCoord * barycentrics.x + v1.texCoord * barycentrics.y + v2.texCoord * barycentrics.z;
	vec3 albedoTex = texture(textureSamplers[nonuniformEXT(txtId)], texCoord).xyz;
    diffuse *= texture(textureSamplers[nonuniformEXT(txtId)], texCoord).xyz;
	albedo *= albedoTex;
  }

  vec3  specular    = vec3(0);
  float attenuation = 1;

  // Tracing shadow ray only if the light is visible from the surface
  //if(dot(normal, cLight.outLightDir) > 0)
  {
    float tMin   = 0.1;
    float tMax   = cLight.outLightDistance;
    vec3  origin = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    vec3  rayDir = cLight.outLightDir;
    uint  flags  = gl_RayFlagsSkipClosestHitShaderEXT;
    isShadowed   = true;
    traceRayEXT(topLevelAS,  // acceleration structure
                flags,       // rayFlags
                0xFF,        // cullMask
                0,           // sbtRecordOffset
                0,           // sbtRecordStride
                1,           // missIndex
                origin,      // ray origin
                tMin,        // ray min range
                rayDir,      // ray direction
                tMax,        // ray max range
                1            // payload (location = 1)
    );

    if(isShadowed)
    {
      attenuation = 0.15;
    }
    else
    {
      // Specular
      specular = computeSpecular(mat, gl_WorldRayDirectionEXT, cLight.outLightDir, normal);
	   attenuation = 1.0;
    }
  }

  // Reflection
  if(mat.illum == 3)
  {
    vec3 origin = worldPos;
    vec3 rayDir = reflect(gl_WorldRayDirectionEXT, normal);
    prd.attenuation *= mat.specular;
    prd.done      = 0;
    prd.rayOrigin = origin;
    prd.rayDir    = rayDir;
  }
  else
  {
	if(pushC.giMode == 1)
	{
	  vec3 origin = worldPos;
	  mat3 tangentCoords = GetTangentCoords(normal);
      vec3 rayDir   =  commonConstants.pointsOnSphereCosDist[prd.depth].xyz * tangentCoords;
	  prd.attenuation = vec3(1.0f);
      prd.done      = 0;
      prd.rayOrigin = origin;
      prd.rayDir    = rayDir;
	}
  }

  const float PI        = 3.14159265358979323846f;
  const float ONE_BY_PI = 1.0f / 3.14159265358979323846f;

  float cosThetha = cos(dot(prd.rayDir, normal));
  float pdf = cosThetha * ONE_BY_PI;

  vec3 directIllum = vec3(cLight.outIntensity * attenuation * (albedo * diffuse + specular));
  vec3 indirectIllum = vec3(0);

  prd.weight = albedo * ONE_BY_PI / pdf * cosThetha;
  prd.hitValue = directIllum + indirectIllum;
}
