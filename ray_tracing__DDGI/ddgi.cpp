#include "ddgi.hpp"

#include "nvh/fileoperations.hpp"
#include "nvvk/descriptorsets_vk.hpp"

#include "nvh/alignment.hpp"
#include "nvvk/shaders_vk.hpp"

extern std::vector<std::string> defaultSearchPaths;

#include <random>
nvmath::vec4f DDGI::RandomPointOnSphereCosineDist()
{
  static std::default_random_engine             generator;
  static std::uniform_real_distribution<double> distribution(0.0, 1.0);

  double r1 = distribution(generator);
  double r2 = distribution(generator);

  //gpuDDGIprops.hemisphereRandomDirs[i].x = std::cos(2.0 * M_PI * r1) * 2.0f * std::sqrt(r2 * (1 - r2));
  //gpuDDGIprops.hemisphereRandomDirs[i].y = std::sin(2.0 * M_PI * r1) * 2.0f * std::sqrt(r2 * (1 - r2));
  //gpuDDGIprops.hemisphereRandomDirs[i].z = 1 - r2;

  constexpr double PI  = 3.14159265358979323846;
  double           phi = 2 * PI * r1;

  return nvmath::vec4f(static_cast<float>(std::cos(phi) * std::sqrt(r2)),
                       static_cast<float>(std::sin(phi) * std::sqrt(r2)),
                       static_cast<float>(std::sqrt(1 - r2)), 1.0f);
}

void DDGI::setup(const vk::Device&         device,
                 const vk::PhysicalDevice& physicalDevice,
                 nvvk::Allocator*          allocator,
                 uint32_t                  queueFamily)
{
  m_device             = device;
  m_physicalDevice     = physicalDevice;
  m_alloc              = allocator;
  m_graphicsQueueIndex = queueFamily;

  // Requesting ray tracing properties
  auto properties =
      m_physicalDevice.getProperties2<vk::PhysicalDeviceProperties2,
                                      vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
  m_rtProperties = properties.get<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
  m_debug.setup(device);

  if(!m_ddgiPropsBuff.buffer)
  {
    m_ddgiPropsBuff = m_alloc->createBuffer(sizeof(GpuDDGIProperties),
                                            vk::BufferUsageFlagBits::eUniformBuffer
                                                | vk::BufferUsageFlagBits::eTransferDst,
                                            vk::MemoryPropertyFlagBits::eDeviceLocal);
    m_debug.setObjectName(m_ddgiPropsBuff.buffer, "ddgiPropsBuffer");
  }
}

void DDGI::createRtDescriptorSet(const vk::AccelerationStructureKHR& tlas)
{
  using vkDT   = vk::DescriptorType;
  using vkSS   = vk::ShaderStageFlagBits;
  using vkDSLB = vk::DescriptorSetLayoutBinding;

  m_rtDescSetLayoutBind.addBinding(vkDSLB(0, vkDT::eAccelerationStructureKHR, 1,
                                          vkSS::eRaygenKHR | vkSS::eClosestHitKHR));  // TLAS
  m_rtDescSetLayoutBind.addBinding(
      vkDSLB(1, vkDT::eStorageImage, 1, vkSS::eRaygenKHR));  // Output image
  m_rtDescSetLayoutBind.addBinding(
      vkDSLB(2, vkDT::eStorageImage, 1, vkSS::eRaygenKHR));  // Output image
  m_rtDescSetLayoutBind.addBinding(
      vkDSLB(3, vkDT::eUniformBuffer, 1,
             vkSS::eRaygenKHR | vkSS::eClosestHitKHR | vkSS::eMissKHR));  // ddgi props
  m_rtDescSetLayoutBind.addBinding(
      vkDSLB(4, vkDT::eUniformBuffer, 1,
             vkSS::eRaygenKHR | vkSS::eClosestHitKHR | vkSS::eMissKHR));  // camera uniforms

  m_rtDescPool      = m_rtDescSetLayoutBind.createPool(m_device);
  m_rtDescSetLayout = m_rtDescSetLayoutBind.createLayout(m_device);
  m_rtDescSet       = m_device.allocateDescriptorSets({m_rtDescPool, 1, &m_rtDescSetLayout})[0];

  vk::WriteDescriptorSetAccelerationStructureKHR descASInfo;
  descASInfo.setAccelerationStructureCount(1);
  descASInfo.setPAccelerationStructures(&tlas);

  std::vector<vk::WriteDescriptorSet> writes;
  writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, 0, &descASInfo));

  m_device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}


//--------------------------------------------------------------------------------------------------
// Writes the output image to the descriptor set
// - Required when changing resolution
//
void DDGI::updateRtDescriptorSet(const vk::CommandBuffer& cmdBuf)
{
  if(!m_needsDescriptorSetUpdate)
    return;

  nvvk::cmdBarrierImageLayout(cmdBuf, irradianceTex.image, vk::ImageLayout::eUndefined,
                              vk::ImageLayout::eGeneral);

  nvvk::cmdBarrierImageLayout(cmdBuf, visibilityTex.image, vk::ImageLayout::eUndefined,
                              vk::ImageLayout::eGeneral);

  m_device.waitIdle();

  vk::DescriptorBufferInfo cameraBufferInfo{m_ddgiPropsBuff.buffer, 0, VK_WHOLE_SIZE};
  vk::DescriptorBufferInfo ddgiBufferInfo{m_ddgiPropsBuff.buffer, 0, VK_WHOLE_SIZE};

  VkDescriptorImageInfo irradianceTexImageView = irradianceTex.descriptor;
  VkDescriptorImageInfo visibilityTexImageView = visibilityTex.descriptor;

  irradianceTexImageView.imageLayout = visibilityTexImageView.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

  std::vector<vk::WriteDescriptorSet> writes;
  writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, 1, &irradianceTexImageView));
  writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, 2, &visibilityTexImageView));
  writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, 3, &cameraBufferInfo));
  writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, 4, &ddgiBufferInfo));
  m_device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

  m_needsDescriptorSetUpdate = false;
}


//--------------------------------------------------------------------------------------------------
// Pipeline for the ray tracer: all shaders, raygen, chit, miss
//
void DDGI::createRtPipeline(vk::DescriptorSetLayout& sceneDescLayout)
{
  std::vector<std::string> paths = defaultSearchPaths;

  vk::ShaderModule raygenSM =
      nvvk::createShaderModule(m_device,  //
                               nvh::loadFile("shaders/ddgi.rgen.spv", true, paths, true));
  vk::ShaderModule missSM =
      nvvk::createShaderModule(m_device,  //
                               nvh::loadFile("shaders/ddgi.rmiss.spv", true, paths, true));

  std::vector<vk::PipelineShaderStageCreateInfo> stages;

  // Raygen
  vk::RayTracingShaderGroupCreateInfoKHR rg{vk::RayTracingShaderGroupTypeKHR::eGeneral,
                                            VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR,
                                            VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR};
  stages.push_back({{}, vk::ShaderStageFlagBits::eRaygenKHR, raygenSM, "main"});
  rg.setGeneralShader(static_cast<uint32_t>(stages.size() - 1));
  m_rtShaderGroups.push_back(rg);  // 0

  // Miss
  vk::RayTracingShaderGroupCreateInfoKHR mg{vk::RayTracingShaderGroupTypeKHR::eGeneral,
                                            VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR,
                                            VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR};
  stages.push_back({{}, vk::ShaderStageFlagBits::eMissKHR, missSM, "main"});
  mg.setGeneralShader(static_cast<uint32_t>(stages.size() - 1));
  m_rtShaderGroups.push_back(mg);  // 1

  // Hit Group0 - Closest Hit + AnyHit
  vk::ShaderModule chitSM =
      nvvk::createShaderModule(m_device,  //
                               nvh::loadFile("shaders/ddgi.rchit.spv", true, paths, true));

  vk::RayTracingShaderGroupCreateInfoKHR hg{vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup,
                                            VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR,
                                            VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR};
  stages.push_back({{}, vk::ShaderStageFlagBits::eClosestHitKHR, chitSM, "main"});
  hg.setClosestHitShader(static_cast<uint32_t>(stages.size() - 1));
  m_rtShaderGroups.push_back(hg);  // 2

  // Callable shaders
  vk::RayTracingShaderGroupCreateInfoKHR callGroup{vk::RayTracingShaderGroupTypeKHR::eGeneral,
                                                   VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR,
                                                   VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR};

  vk::ShaderModule call0 =
      nvvk::createShaderModule(m_device,
                               nvh::loadFile("shaders/light_point.rcall.spv", true, paths, true));
  vk::ShaderModule call1 =
      nvvk::createShaderModule(m_device,
                               nvh::loadFile("shaders/light_spot.rcall.spv", true, paths, true));
  vk::ShaderModule call2 =
      nvvk::createShaderModule(m_device,
                               nvh::loadFile("shaders/light_inf.rcall.spv", true, paths, true));

  stages.push_back({{}, vk::ShaderStageFlagBits::eCallableKHR, call0, "main"});
  callGroup.setGeneralShader(static_cast<uint32_t>(stages.size() - 1));
  m_rtShaderGroups.push_back(callGroup);  // 3
  stages.push_back({{}, vk::ShaderStageFlagBits::eCallableKHR, call1, "main"});
  callGroup.setGeneralShader(static_cast<uint32_t>(stages.size() - 1));
  m_rtShaderGroups.push_back(callGroup);  // 4
  stages.push_back({{}, vk::ShaderStageFlagBits::eCallableKHR, call2, "main"});
  callGroup.setGeneralShader(static_cast<uint32_t>(stages.size() - 1));
  m_rtShaderGroups.push_back(callGroup);  // 5


  vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;

  // Push constant: we want to be able to update constants used by the shaders
  vk::PushConstantRange pushConstant{vk::ShaderStageFlagBits::eRaygenKHR
                                         | vk::ShaderStageFlagBits::eClosestHitKHR
                                         | vk::ShaderStageFlagBits::eMissKHR
                                         | vk::ShaderStageFlagBits::eCallableKHR,
                                     0, sizeof(RtPushConstants)};
  pipelineLayoutCreateInfo.setPushConstantRangeCount(1);
  pipelineLayoutCreateInfo.setPPushConstantRanges(&pushConstant);

  // Descriptor sets: one specific to ray tracing, and one shared with the rasterization pipeline
  std::vector<vk::DescriptorSetLayout> rtDescSetLayouts = {m_rtDescSetLayout, sceneDescLayout};
  pipelineLayoutCreateInfo.setSetLayoutCount(static_cast<uint32_t>(rtDescSetLayouts.size()));
  pipelineLayoutCreateInfo.setPSetLayouts(rtDescSetLayouts.data());

  m_rtPipelineLayout = m_device.createPipelineLayout(pipelineLayoutCreateInfo);

  // Assemble the shader stages and recursion depth info into the ray tracing pipeline
  vk::RayTracingPipelineCreateInfoKHR rayPipelineInfo;
  rayPipelineInfo.setStageCount(static_cast<uint32_t>(stages.size()));  // Stages are shaders
  rayPipelineInfo.setPStages(stages.data());

  rayPipelineInfo.setGroupCount(static_cast<uint32_t>(
      m_rtShaderGroups.size()));  // 1-raygen, n-miss, n-(hit[+anyhit+intersect])
  rayPipelineInfo.setPGroups(m_rtShaderGroups.data());

  rayPipelineInfo.setMaxPipelineRayRecursionDepth(2);  // Ray depth
  rayPipelineInfo.setLayout(m_rtPipelineLayout);
  m_rtPipeline = static_cast<const vk::Pipeline&>(
      m_device.createRayTracingPipelineKHR({}, {}, rayPipelineInfo));

  m_device.destroy(raygenSM);
  m_device.destroy(missSM);
  m_device.destroy(chitSM);
  m_device.destroy(call0);
  m_device.destroy(call1);
  m_device.destroy(call2);
}

//--------------------------------------------------------------------------------------------------
// The Shader Binding Table (SBT)
// - getting all shader handles and writing them in a SBT buffer
// - Besides exception, this could be always done like this
//   See how the SBT buffer is used in run()
//
void DDGI::createRtShaderBindingTable()
{
  auto groupCount =
      static_cast<uint32_t>(m_rtShaderGroups.size());               // 3 shaders: raygen, miss, chit
  uint32_t groupHandleSize = m_rtProperties.shaderGroupHandleSize;  // Size of a program identifier
  uint32_t groupSizeAligned =
      nvh::align_up(groupHandleSize, m_rtProperties.shaderGroupBaseAlignment);


  // Fetch all the shader handles used in the pipeline, so that they can be written in the SBT
  uint32_t sbtSize = groupCount * groupSizeAligned;

  std::vector<uint8_t> shaderHandleStorage(sbtSize);
  auto result = m_device.getRayTracingShaderGroupHandlesKHR(m_rtPipeline, 0, groupCount, sbtSize,
                                                            shaderHandleStorage.data());
  assert(result == vk::Result::eSuccess);

  // Write the handles in the SBT
  m_rtSBTBuffer = m_alloc->createBuffer(
      sbtSize,
      vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eShaderDeviceAddress
          | vk::BufferUsageFlagBits::eShaderBindingTableKHR,
      vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
  m_debug.setObjectName(m_rtSBTBuffer.buffer, std::string("SBT").c_str());

  // Write the handles in the SBT
  void* mapped = m_alloc->map(m_rtSBTBuffer);
  auto* pData  = reinterpret_cast<uint8_t*>(mapped);
  for(uint32_t g = 0; g < groupCount; g++)
  {
    memcpy(pData, shaderHandleStorage.data() + g * groupHandleSize, groupHandleSize);  // raygen
    pData += groupSizeAligned;
  }
  m_alloc->unmap(m_rtSBTBuffer);

  m_alloc->finalizeAndReleaseStaging();
}

//--------------------------------------------------------------------------------------------------
// Called at each frame to update the camera matrix
//
void DDGI::updateUniformBuffer(const vk::CommandBuffer& cmdBuf)
{
  // Prepare new UBO contents on host.
  GpuDDGIProperties gpuDDGIprops = {};
  gpuDDGIprops.maxPoint          = maxPoint;
  gpuDDGIprops.minPoint          = minPoint;
  gpuDDGIprops.probeDim =
      (maxPoint - minPoint) * nvmath::vec3f(1.0f / elems[0], 1.0f / elems[1], 1.0f / elems[2]);
  //elems[0] * elems[1] * elems[2];

#if 0
  int     i = 1, j = 1, k = 1;
  uint8_t fixedAxis = 0;

  float    du = 2.0f / (resolution[0] - 1), dv = 2.0f / (resolution[1] - 1);
  uint32_t samplesPerSide = resolution[0] * resolution[1];
  gpuDDGIprops.subSamplesPerProbe = 6 * samplesPerSide;
  for(uint8_t side = 0; side < 6; ++side)
  {
    if(side && (side & 1) == 0)
      ++fixedAxis;

    float sgn = +1.0f;
    if(side & 1)
      sgn = -1.0f;

    for(uint16_t j = 0; j < resolution[1]; ++j)
      for(uint16_t i = 0; i < resolution[0]; ++i)
      {
        nvmath::vec4f dir;
        dir[(fixedAxis + 0) % 3] = sgn;
        dir[(fixedAxis + 1) % 3] = -1.0f + (i + 0.5f) * du;
        dir[(fixedAxis + 2) % 3] = -1.0f + (j + 0.5f) * dv;
        dir[3]                   = 0.0f;
        dir /= dir.norm();

        uint32_t idx = side * samplesPerSide + j * resolution[0] + i;
        assert(idx < MAX_SUBSAMPLES_PER_PROBE);
        gpuDDGIprops.subSampleDirs[idx]        = dir;
        gpuDDGIprops.subSampleStoreOffset[idx] = nvmath::vec4ui(side * resolution[0] + i, j, 0, 0);
      }
  }
#else
  gpuDDGIprops.samplesOnHemisphere = std::min(64, HEMISPHERE_RANDOM_DIR_COUNT);
  for(int i = 0; i < HEMISPHERE_RANDOM_DIR_COUNT; ++i)
  {
    gpuDDGIprops.hemisphereRandomDirs[i] = DDGI::RandomPointOnSphereCosineDist();
  }
#endif

  // UBO on the device, and what stages access it.
  vk::Buffer         deviceUBO = m_ddgiPropsBuff.buffer;
  GpuDDGIProperties& hostUBO   = gpuDDGIprops;
  auto               uboUsageStages =
      vk::PipelineStageFlagBits::eVertexShader | vk::PipelineStageFlagBits::eRayTracingShaderKHR;

  // Ensure that the modified UBO is not visible to previous frames.
  vk::BufferMemoryBarrier beforeBarrier;
  beforeBarrier.setSrcAccessMask(vk::AccessFlagBits::eShaderRead);
  beforeBarrier.setDstAccessMask(vk::AccessFlagBits::eTransferWrite);
  beforeBarrier.setBuffer(deviceUBO);
  beforeBarrier.setOffset(0);
  beforeBarrier.setSize(sizeof hostUBO);
  cmdBuf.pipelineBarrier(uboUsageStages, vk::PipelineStageFlagBits::eTransfer,
                         vk::DependencyFlagBits::eDeviceGroup, {}, {beforeBarrier}, {});

  // Schedule the host-to-device upload. (hostUBO is copied into the cmd
  // buffer so it is okay to deallocate when the function returns).
  cmdBuf.updateBuffer<GpuDDGIProperties>(m_ddgiPropsBuff.buffer, 0, hostUBO);

  // Making sure the updated UBO will be visible.
  vk::BufferMemoryBarrier afterBarrier;
  afterBarrier.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite);
  afterBarrier.setDstAccessMask(vk::AccessFlagBits::eShaderRead);
  afterBarrier.setBuffer(deviceUBO);
  afterBarrier.setOffset(0);
  afterBarrier.setSize(sizeof hostUBO);
  cmdBuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, uboUsageStages,
                         vk::DependencyFlagBits::eDeviceGroup, {}, {afterBarrier}, {});
}

void DDGI::build(const vk::CommandBuffer& cmdBuf,
                 vk::DescriptorSet&       sceneDescSet,
                 const nvmath::vec4f&     clearColor,
                 ObjPushConstants&        sceneConstants)
{
  updateUniformBuffer(cmdBuf);
  updateRtDescriptorSet(cmdBuf);

  m_debug.beginLabel(cmdBuf, "ddgi");
  // Initializing push constant values
  m_rtPushConstants.clearColor           = clearColor;
  m_rtPushConstants.lightPosition        = sceneConstants.lightPosition;
  m_rtPushConstants.lightIntensity       = sceneConstants.lightIntensity;
  m_rtPushConstants.lightDirection       = sceneConstants.lightDirection;
  m_rtPushConstants.lightSpotCutoff      = sceneConstants.lightSpotCutoff;
  m_rtPushConstants.lightSpotOuterCutoff = sceneConstants.lightSpotOuterCutoff;
  m_rtPushConstants.lightType            = sceneConstants.lightType;
  m_rtPushConstants.frame                = sceneConstants.frame;

  cmdBuf.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, m_rtPipeline);
  cmdBuf.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR, m_rtPipelineLayout, 0,
                            {m_rtDescSet, sceneDescSet}, {});
  cmdBuf.pushConstants<RtPushConstants>(m_rtPipelineLayout,
                                        vk::ShaderStageFlagBits::eRaygenKHR
                                            | vk::ShaderStageFlagBits::eClosestHitKHR
                                            | vk::ShaderStageFlagBits::eMissKHR
                                            | vk::ShaderStageFlagBits::eCallableKHR,
                                        0, m_rtPushConstants);

  // Size of a program identifier
  uint32_t groupSize =
      nvh::align_up(m_rtProperties.shaderGroupHandleSize, m_rtProperties.shaderGroupBaseAlignment);
  uint32_t          groupStride = groupSize;
  vk::DeviceAddress sbtAddress  = m_device.getBufferAddress({m_rtSBTBuffer.buffer});

  using Stride = vk::StridedDeviceAddressRegionKHR;
  std::array<Stride, 4> strideAddresses{
      Stride{sbtAddress + 0u * groupSize, groupStride, groupSize * 1},   // raygen
      Stride{sbtAddress + 1u * groupSize, groupStride, groupSize * 1},   // miss
      Stride{sbtAddress + 2u * groupSize, groupStride, groupSize * 1},   // hit
      Stride{sbtAddress + 3u * groupSize, groupStride, groupSize * 1}};  // callable

  cmdBuf.traceRaysKHR(&strideAddresses[0], &strideAddresses[1], &strideAddresses[2],
                      &strideAddresses[3], 4 * 1024, 4 * 1024, 1);  //
  //elems[0] * 6*resolution[0] * resolution[1], elems[1], elems[2]);  //


  m_debug.endLabel(cmdBuf);
}

void DDGI::update(uint32_t w, uint32_t h)
{
  if(width == w && height == h)
    return;

  // Wait for GPU work to be finished
  m_device.waitIdle();
  m_alloc->destroy(irradianceTex);
  m_alloc->destroy(visibilityTex);

  vk::SamplerCreateInfo samplerCreateInfo{
      {}, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eNearest};
  samplerCreateInfo.setMaxLod(FLT_MAX);

  {
    auto imgSize         = vk::Extent2D(w, h);
    auto imageCreateInfo = nvvk::makeImage2DCreateInfo(imgSize, vk::Format::eR16G16B16A16Sfloat,
                                                       vk::ImageUsageFlagBits::eStorage
                                                           | vk::ImageUsageFlagBits::eSampled);

    // Creating the VKImage
    nvvk::Image image =
        m_alloc->createImage(imageCreateInfo, vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk::ImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, imageCreateInfo);
    irradianceTex                  = m_alloc->createTexture(image, ivInfo, samplerCreateInfo);
    m_debug.setObjectName(irradianceTex.image, "ddgiIrradianceTexture");
  }

  {
    auto imgSize         = vk::Extent2D(w, h);
    auto imageCreateInfo = nvvk::makeImage2DCreateInfo(imgSize, vk::Format::eR16G16Sfloat,
                                                       vk::ImageUsageFlagBits::eStorage
                                                           | vk::ImageUsageFlagBits::eSampled);

    // Creating the VKImage
    nvvk::Image image =
        m_alloc->createImage(imageCreateInfo, vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk::ImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, imageCreateInfo);
    visibilityTex                  = m_alloc->createTexture(image, ivInfo, samplerCreateInfo);

    m_debug.setObjectName(visibilityTex.image, "ddgiVisibilityTexture");
  }

  width  = w;
  height = h;

  m_needsDescriptorSetUpdate = true;
}

void DDGI::destroy()
{
  m_alloc->destroy(irradianceTex);
  m_alloc->destroy(visibilityTex);
  m_alloc->destroy(m_ddgiPropsBuff);

  m_device.destroy(m_rtDescPool);
  m_device.destroy(m_rtDescSetLayout);
  m_device.destroy(m_rtPipeline);
  m_device.destroy(m_rtPipelineLayout);
  m_alloc->destroy(m_rtSBTBuffer);
}