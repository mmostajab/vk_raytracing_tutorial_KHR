#pragma once

#include <nvmath_types.h>

class DDGI
{
public:
  DDGI();
  ~DDGI();

  void build();
  void update();

  const nvvk::Texture& GetIrradianceTex() const { return irradianceTex; }
  const nvvk::Texture& GetVisibilityTex() const { return visibilityTex; }

private:
  nvmath::vec3f  minPoint, maxPoint;
  nvmath::vec3ui elems;

  uint32_t      width, height;
  nvvk::Texture irradianceTex;
  nvvk::Texture visibilityTex;
};