struct hitPayload
{
  vec3 hitValue;
  uint seed;
  int  depth;
  vec3 attenuation;
  int  done;
  vec3 rayOrigin;
  vec3 rayDir;
  vec3 weight;
};

struct ddgiHitPayload
{
  vec4   irradiance;
  vec3   normal;
  float  depth;
  float  depth2;
};

struct rayLight
{
  vec3  inHitPosition;
  float outLightDistance;
  vec3  outLightDir;
  float outIntensity;
};

struct Implicit
{
  vec3 minimum;
  vec3 maximum;
  int  objType;
  int  matId;
};

struct Sphere
{
  vec3  center;
  float radius;
};

struct Aabb
{
  vec3 minimum;
  vec3 maximum;
};

mat3 GetTangentCoords(vec3 n)
{
  vec3 z   = n;
  vec3 tmp = cross(z, vec3(1, 0, 0));
  vec3 x, y;
  if(length(tmp) < 0.01f)
  {
    x = cross(vec3(0, 1, 0), z);
    y = cross(z, x);
  }
  else
  {
    y = tmp;
    x = cross(y, z);
  }

  mat3 tangentCoords;
  tangentCoords[0] = normalize(x);
  tangentCoords[1] = normalize(y);
  tangentCoords[2] = normalize(z);

  return tangentCoords;
}

#define KIND_SPHERE 0
#define KIND_CUBE 1
