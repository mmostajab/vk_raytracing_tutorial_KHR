#pragma once

#include <vulkan/vulkan.hpp>

#include "nvvk/descriptorsets_vk.hpp"
#include "vkalloc.hpp"

#include "nvmath/nvmath.h"
#include "nvvk/raytraceKHR_vk.hpp"

#define MAX_SUBSAMPLES_PER_PROBE 4 * 1024
struct GpuDDGIProperties
{
  nvmath::vec4f minPoint;
  nvmath::vec4f maxPoint;
  nvmath::vec4f probeDim;
  nvmath::vec4f subSamplesPerProbe;
  nvmath::vec4f subSampleDirs[MAX_SUBSAMPLES_PER_PROBE];
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
  void updateRtDescriptorSet();
  void createRtPipeline(vk::DescriptorSetLayout& sceneDescLayout);
  void createRtShaderBindingTable();

  void build(const vk::CommandBuffer& cmdBuf);
  void update(uint32_t w, uint32_t h);
  void updateUniformBuffer(const vk::CommandBuffer& cmdBuf);

  const nvvk::Texture& GetIrradianceTex() const { return irradianceTex; }
  const nvvk::Texture& GetVisibilityTex() const { return visibilityTex; }

private:
  nvmath::vec3f  minPoint, maxPoint;
  nvmath::vec3ui elems;

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