#pragma once

#include <vulkan/vulkan.hpp>

#include "nvvk/descriptorsets_vk.hpp"
#include "vkalloc.hpp"

#include "nvmath/nvmath.h"
#include "nvvk/raytraceKHR_vk.hpp"

#include "aabb.hpp"
#include "obj.hpp"

#define MAX_SUBSAMPLES_PER_PROBE (6 * 16 * 16)
#define HEMISPHERE_RANDOM_DIR_COUNT (256)

struct GpuDDGIProperties
{
  nvmath::vec4f minPoint;
  nvmath::vec4f maxPoint;
  nvmath::vec4f probeDim;
  uint32_t      subSamplesPerProbe;
  uint32_t      samplesOnHemisphere;
  uint32_t      padding0;
  uint32_t      padding1;
  //nvmath::vec4f  subSampleDirs       [MAX_SUBSAMPLES_PER_PROBE];
  //nvmath::vec4ui subSampleStoreOffset[MAX_SUBSAMPLES_PER_PROBE];
  nvmath::vec4f hemisphereRandomDirs[HEMISPHERE_RANDOM_DIR_COUNT];
};

enum StorageScheme
{
  STORAGE_SCHEME_CUBEMAP = 0,
};

class DDGI
{
public:
  DDGI()  = default;
  ~DDGI() = default;

  void setup(const vk::Device&         device,
             const vk::PhysicalDevice& physicalDevice,
             nvvk::Allocator*          allocator,
             uint32_t                  queueFamily);
  void createRtDescriptorSet(const vk::AccelerationStructureKHR& tlas);
  void updateRtDescriptorSet(const vk::CommandBuffer& cmdBuf);
  void createRtPipeline(vk::DescriptorSetLayout& sceneDescLayout);
  void createRtShaderBindingTable();

  void build(const vk::CommandBuffer& cmdBuf,
             vk::DescriptorSet&       sceneDescSet,
             const nvmath::vec4f&     clearColor,
             ObjPushConstants&        sceneConstants);
  void update(uint32_t w, uint32_t h);
  void updateUniformBuffer(const vk::CommandBuffer& cmdBuf);

  void SetSceneBounds(const nvmath::vec3f& minPnt, const nvmath::vec3f& maxPnt)
  {
    minPoint = minPnt, maxPoint = maxPnt;
  }
  void SetProbeCountOnMaxDim(int probeCountOnMaxDim, StorageScheme scheme)
  {
    if(probeCountOnMaxDim == -1)
    {
      elems = nvmath::vec3ui(1, 1, 1);
      return;
    }

    uint8_t    maxDimIdx = 0;
    const auto dims      = maxPoint - minPoint;
    if(dims[0] <= dims[1])
      if(dims[0] <= dims[2])
        if(dims[1] <= dims[2])
          maxDimIdx = 2;
        else
          maxDimIdx = 1;
      else
        maxDimIdx = 1;
    else if(dims[0] <= dims[2])
      maxDimIdx = 2;
    else
      maxDimIdx = 0;

    for(uint8_t axis = 0; axis < 3; ++axis)
      elems[axis] = static_cast<unsigned int>(dims[axis] / dims[maxDimIdx] * probeCountOnMaxDim);

    storageScheme = scheme;
  }

  const nvvk::Texture& GetIrradianceTex() const { return irradianceTex; }
  const nvvk::Texture& GetVisibilityTex() const { return visibilityTex; }

  static nvmath::vec4f RandomPointOnSphereCosineDist();

  void destroy();

private:
  nvmath::vec3f  minPoint, maxPoint;
  nvmath::vec2ui resolution = {16, 16};
  nvmath::vec3ui elems;
  StorageScheme  storageScheme;

  bool          m_needsDescriptorSetUpdate = true;
  uint32_t      width, height;
  nvvk::Buffer  m_ddgiPropsBuff;
  nvvk::Texture irradianceTex;
  nvvk::Texture visibilityTex;

  nvvk::Allocator*   m_alloc{nullptr};  // Allocator for buffer, images, acceleration structures
  vk::PhysicalDevice m_physicalDevice;
  vk::Device         m_device;
  int                m_graphicsQueueIndex{0};
  nvvk::DebugUtil    m_debug;  // Utility to name objects


  vk::PhysicalDeviceRayTracingPipelinePropertiesKHR   m_rtProperties;
  nvvk::DescriptorSetBindings                         m_rtDescSetLayoutBind;
  vk::DescriptorPool                                  m_rtDescPool;
  vk::DescriptorSetLayout                             m_rtDescSetLayout;
  vk::DescriptorSet                                   m_rtDescSet;
  std::vector<vk::RayTracingShaderGroupCreateInfoKHR> m_rtShaderGroups;
  vk::PipelineLayout                                  m_rtPipelineLayout;
  vk::Pipeline                                        m_rtPipeline;
  nvvk::Buffer                                        m_rtSBTBuffer;

  struct RtPushConstants
  {
    nvmath::vec4f clearColor;
    nvmath::vec3f lightPosition;
    float         lightIntensity;
    nvmath::vec3f lightDirection{-1, -1, -1};
    float         lightSpotCutoff{deg2rad(12.5f)};
    float         lightSpotOuterCutoff{deg2rad(17.5f)};
    int           lightType{0};
    int           frame{0};
  } m_rtPushConstants;
};