#pragma once

#include <algorithm>
#include "nvmath/nvmath.h"

struct AABB
{
  nvmath::vec3f min = nvmath::vec3f(FLT_MAX, FLT_MAX, FLT_MAX);
  nvmath::vec3f max = -nvmath::vec3f(FLT_MAX, FLT_MAX, FLT_MAX);

  void Extend(const nvmath::vec3f& pnt)
  {
    min.x = std::min(pnt.x, min.x);
    min.y = std::min(pnt.y, min.y);
    min.z = std::min(pnt.z, min.z);

	max.x = std::max(pnt.x, max.x);
    max.y = std::max(pnt.x, max.y);
	max.z = std::max(pnt.x, max.z);
  }

  bool Includes(const nvmath::vec3f& pnt) const
  {
    return pnt.x >= min.x && pnt.y >= min.y && pnt.z >= min.z 
		&& pnt.x <= max.x && pnt.y <= max.y && pnt.z <= max.z;
  }

  nvmath::vec3f Extent() const { return max - min;  }
};