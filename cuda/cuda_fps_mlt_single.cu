// cuda_fps_mlt_single.cu
// Single-file CUDA + C++ + OpenCV
// - base mesh loaded from:   base/base.obj
// - morph targets loaded from shapes/*.obj
// - dynamic multi-shape morphing
// - Python/Tkinter/ttkbootstrap controller sends all weights by UDP localhost
//
// Build (Linux):
//   nvcc -O3 -std=c++17 --use_fast_math cuda_fps_mlt_single.cu `pkg-config --cflags --libs opencv4` -lX11 -o cuda_fps_mlt
//
// Run:
//   ./cuda_fps_mlt
//   ./cuda_fps_mlt [base_folder] [shapes_folder]
//
// Default folders:
//   base
//   shapes

#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <unordered_map>
#include <cctype>
#include <cstdlib>
#include <filesystem>

#ifdef __linux__
  #include <X11/Xlib.h>
  #include <X11/keysym.h>

  #include <sys/types.h>
  #include <sys/socket.h>
  #include <arpa/inet.h>
  #include <netinet/in.h>
  #include <fcntl.h>
  #include <unistd.h>
  #include <cerrno>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace fs = std::filesystem;

static const char* WIN_NAME = "CUDA FPS MLT";

// ============================
// CUDA error check
// ============================
#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  std::cerr << "CUDA error: " << cudaGetErrorString(e) << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
  std::exit(1); } } while(0)

// ============================
// Math (host+device)
// ============================
__host__ __device__ inline float3 make_f3(float x, float y, float z) { return make_float3(x,y,z); }

__host__ __device__ inline float3 operator+(const float3& a, const float3& b){ return make_float3(a.x+b.x,a.y+b.y,a.z+b.z); }
__host__ __device__ inline float3 operator-(const float3& a, const float3& b){ return make_float3(a.x-b.x,a.y-b.y,a.z-b.z); }
__host__ __device__ inline float3 operator-(const float3& a){ return make_float3(-a.x,-a.y,-a.z); }
__host__ __device__ inline float3 operator*(const float3& a, float b){ return make_float3(a.x*b,a.y*b,a.z*b); }
__host__ __device__ inline float3 operator*(float b, const float3& a){ return make_float3(a.x*b,a.y*b,a.z*b); }

__host__ __device__ inline float3 operator*(const float3& a, const float3& b){
  return make_float3(a.x*b.x, a.y*b.y, a.z*b.z);
}
__host__ __device__ inline float3 operator/(const float3& a, float b){ return make_float3(a.x/b,a.y/b,a.z/b); }

__host__ __device__ inline float dot3(const float3& a, const float3& b){ return a.x*b.x+a.y*b.y+a.z*b.z; }
__host__ __device__ inline float3 cross3(const float3& a, const float3& b){
  return make_float3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);
}
__host__ __device__ inline float len3(const float3& a){ return sqrtf(dot3(a,a)); }
__host__ __device__ inline float3 norm3(const float3& a){ float l=len3(a); return (l>0)? a/l : make_float3(0,0,0); }

__host__ __device__ inline float3 min3(const float3& a, const float3& b){
  return make_float3(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z));
}
__host__ __device__ inline float3 max3(const float3& a, const float3& b){
  return make_float3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z));
}

__host__ __device__ inline float clamp01f_scalar(float x){
  return fminf(1.0f, fmaxf(0.0f, x));
}

__host__ __device__ inline float3 clamp01f3(const float3& c){
  return make_float3(
    clamp01f_scalar(c.x),
    clamp01f_scalar(c.y),
    clamp01f_scalar(c.z)
  );
}

__host__ __device__ inline float luminance(const float3& c){
  return 0.2126f*c.x + 0.7152f*c.y + 0.0722f*c.z;
}

// ============================
// RNG (xorshift64*)
// ============================
__host__ __device__ inline uint64_t xorshift64star(uint64_t& s){
  s ^= s >> 12;
  s ^= s << 25;
  s ^= s >> 27;
  return s * 2685821657736338717ULL;
}
__host__ __device__ inline float rnd01(uint64_t& s){
  uint32_t v = (uint32_t)(xorshift64star(s) >> 40);
  return (float)v / (float)(1u<<24);
}

// ============================
// Geometry
// ============================
struct Tri {
  float3 v0, v1, v2;
  float3 n0, n1, n2;
  float3 albedo;
  float3 specular;
  float  glossiness;
};

struct AABB {
  float3 bmin;
  float3 bmax;
};

static inline AABB aabb_empty(){
  AABB b;
  b.bmin = make_f3( 1e30f, 1e30f, 1e30f);
  b.bmax = make_f3(-1e30f,-1e30f,-1e30f);
  return b;
}

static inline void aabb_extend(AABB& b, const float3& p){
  b.bmin = min3(b.bmin, p);
  b.bmax = max3(b.bmax, p);
}
static inline void aabb_extend(AABB& b, const AABB& o){
  b.bmin = min3(b.bmin, o.bmin);
  b.bmax = max3(b.bmax, o.bmax);
}
static inline AABB tri_aabb(const Tri& t){
  AABB b = aabb_empty();
  aabb_extend(b, t.v0); aabb_extend(b, t.v1); aabb_extend(b, t.v2);
  return b;
}
static inline float3 tri_centroid(const Tri& t){
  return (t.v0 + t.v1 + t.v2) / 3.0f;
}

// ============================
// BVH
// ============================
struct BVHNode {
  float3 bmin;
  float3 bmax;
  int left;
  int right;
  int start;
  int count;
};

static AABB tri_aabb_multi(const Tri& baseTri, const std::vector<std::vector<Tri>>* shapes, int triIndex){
  AABB b = tri_aabb(baseTri);
  if (shapes) {
    for (const auto& shape : *shapes) {
      if (triIndex >= 0 && triIndex < (int)shape.size()) {
        aabb_extend(b, shape[triIndex].v0);
        aabb_extend(b, shape[triIndex].v1);
        aabb_extend(b, shape[triIndex].v2);
      }
    }
  }
  return b;
}

static int build_bvh_recursive(
  std::vector<BVHNode>& nodes,
  std::vector<int>& triIdx,
  const std::vector<Tri>& tris,
  const std::vector<std::vector<Tri>>* morphShapes,
  int start, int end
){
  BVHNode node{};
  AABB box = aabb_empty();
  AABB cbox = aabb_empty();

  for(int i=start;i<end;i++){
    int ti = triIdx[i];
    aabb_extend(box, tri_aabb_multi(tris[ti], morphShapes, ti));
    aabb_extend(cbox, tri_centroid(tris[ti]));
  }

  node.bmin = box.bmin;
  node.bmax = box.bmax;
  node.left = node.right = -1;
  node.start = start;
  node.count = end - start;

  const int myIndex = (int)nodes.size();
  nodes.push_back(node);

  const int LEAF_N = 4;
  if (end - start <= LEAF_N) {
    nodes[myIndex].left = -1;
    nodes[myIndex].right = -1;
    return myIndex;
  }

  float3 ext = cbox.bmax - cbox.bmin;
  int axis = 0;
  if (ext.y > ext.x && ext.y > ext.z) axis = 1;
  else if (ext.z > ext.x && ext.z > ext.y) axis = 2;

  int mid = (start + end) / 2;
  std::nth_element(triIdx.begin()+start, triIdx.begin()+mid, triIdx.begin()+end,
    [&](int a, int b){
      float3 ca = tri_centroid(tris[a]);
      float3 cb = tri_centroid(tris[b]);
      if (axis==0) return ca.x < cb.x;
      if (axis==1) return ca.y < cb.y;
      return ca.z < cb.z;
    });

  int L = build_bvh_recursive(nodes, triIdx, tris, morphShapes, start, mid);
  int R = build_bvh_recursive(nodes, triIdx, tris, morphShapes, mid, end);

  nodes[myIndex].left = L;
  nodes[myIndex].right = R;
  nodes[myIndex].start = -1;
  nodes[myIndex].count = 0;
  return myIndex;
}

// ============================
// Path helpers
// ============================
static std::string trim(const std::string& s){
  size_t a = 0;
  while (a < s.size() && std::isspace((unsigned char)s[a])) a++;
  size_t b = s.size();
  while (b > a && std::isspace((unsigned char)s[b-1])) b--;
  return s.substr(a, b - a);
}

static std::string get_directory_part(const std::string& path){
  size_t p1 = path.find_last_of('/');
  size_t p2 = path.find_last_of('\\');
  size_t p = std::string::npos;
  if (p1 == std::string::npos) p = p2;
  else if (p2 == std::string::npos) p = p1;
  else p = std::max(p1, p2);

  if (p == std::string::npos) return "";
  return path.substr(0, p + 1);
}

static std::string join_path_simple(const std::string& baseDir, const std::string& fileName){
  if (baseDir.empty()) return fileName;
  return baseDir + fileName;
}

static std::string lower_copy(std::string s){
  for (char& c : s) c = (char)std::tolower((unsigned char)c);
  return s;
}

static std::vector<std::string> list_obj_files_sorted(const std::string& folder){
  std::vector<std::string> out;
  if (!fs::exists(folder) || !fs::is_directory(folder)) return out;

  for (const auto& entry : fs::directory_iterator(folder)) {
    if (!entry.is_regular_file()) continue;
    std::string ext = lower_copy(entry.path().extension().string());
    if (ext == ".obj") {
      out.push_back(entry.path().string());
    }
  }

  std::sort(out.begin(), out.end(), [](const std::string& a, const std::string& b){
    return lower_copy(fs::path(a).filename().string()) < lower_copy(fs::path(b).filename().string());
  });

  return out;
}

static std::vector<float> parse_float_list_loose(const std::string& s){
  std::string t = s;
  for (char& c : t) {
    if (c == ',' || c == '|' || c == '\t' || c == '\n' || c == '\r') c = ';';
  }

  std::vector<float> out;
  std::stringstream ss(t);
  std::string part;
  while (std::getline(ss, part, ';')) {
    part = trim(part);
    if (part.empty()) continue;
    try {
      out.push_back(std::stof(part));
    } catch (...) {
    }
  }
  return out;
}

// ============================
// Minimal MTL loader
// ============================
struct MaterialCPU {
  float3 Kd = make_f3(0.75f, 0.75f, 0.75f);
  float3 Ks = make_f3(0.00f, 0.00f, 0.00f);
  float  Ns = 32.0f;
};

static bool load_mtl(const std::string& mtlPath, std::unordered_map<std::string, MaterialCPU>& materials){
  std::ifstream f(mtlPath);
  if (!f) {
    std::cerr << "Warning: could not open MTL file: " << mtlPath << "\n";
    return false;
  }

  std::string line;
  std::string currentName;
  MaterialCPU currentMat;
  bool hasCurrent = false;

  while (std::getline(f, line)) {
    line = trim(line);
    if (line.empty()) continue;
    if (line[0] == '#') continue;

    std::stringstream ss(line);
    std::string tag;
    ss >> tag;

    if (tag == "newmtl") {
      if (hasCurrent && !currentName.empty()) {
        materials[currentName] = currentMat;
      }
      currentName.clear();
      ss >> currentName;
      currentMat = MaterialCPU{};
      hasCurrent = true;
    } else if (tag == "Kd") {
      float r=0.75f, g=0.75f, b=0.75f;
      ss >> r >> g >> b;
      currentMat.Kd = clamp01f3(make_f3(r,g,b));
    } else if (tag == "Ks") {
      float r=0.0f, g=0.0f, b=0.0f;
      ss >> r >> g >> b;
      currentMat.Ks = clamp01f3(make_f3(r,g,b));
    } else if (tag == "Ns") {
      float ns = 32.0f;
      ss >> ns;
      currentMat.Ns = fmaxf(1.0f, ns);
    }
  }

  if (hasCurrent && !currentName.empty()) {
    materials[currentName] = currentMat;
  }

  std::cerr << "Loaded materials: " << materials.size() << " from " << mtlPath << "\n";
  return true;
}

// ============================
// Minimal OBJ loader
// ============================
static bool parse_face_vertex(const std::string& tok, int& vi, int& ni){
  vi = 0; ni = 0;
  int a=0,b=0,c=0;
  char ch;
  std::stringstream ss(tok);

  if (!(ss >> a)) return false;
  vi = a;

  if (ss.peek() == '/') {
    ss >> ch;
    if (ss.peek() == '/') {
      ss >> ch;
      if (ss >> c) ni = c;
    } else {
      if (ss >> b) {
        if (ss.peek() == '/') {
          ss >> ch;
          if (ss >> c) ni = c;
        }
      }
    }
  }
  return true;
}

static bool load_obj(const std::string& path, std::vector<Tri>& outTris){
  std::ifstream f(path);
  if (!f) return false;

  std::vector<float3> V;
  std::vector<float3> N;

  std::unordered_map<std::string, MaterialCPU> materials;
  std::string currentMaterialName;
  MaterialCPU currentMaterial;
  currentMaterial.Kd = make_f3(0.75f, 0.75f, 0.75f);
  currentMaterial.Ks = make_f3(0.00f, 0.00f, 0.00f);
  currentMaterial.Ns = 32.0f;

  const std::string baseDir = get_directory_part(path);

  std::string line;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    if (line[0] == '#') continue;

    std::stringstream ss(line);
    std::string tag;
    ss >> tag;

    if (tag == "v") {
      float x,y,z; ss >> x >> y >> z;
      V.push_back(make_f3(x,y,z));
    } else if (tag == "vn") {
      float x,y,z; ss >> x >> y >> z;
      N.push_back(norm3(make_f3(x,y,z)));
    } else if (tag == "mtllib") {
      std::string mtlFile;
      ss >> mtlFile;
      if (!mtlFile.empty()) {
        std::string mtlPath = join_path_simple(baseDir, mtlFile);
        load_mtl(mtlPath, materials);
      }
    } else if (tag == "usemtl") {
      ss >> currentMaterialName;
      auto it = materials.find(currentMaterialName);
      if (it != materials.end()) currentMaterial = it->second;
      else currentMaterial = MaterialCPU{};
    } else if (tag == "f") {
      std::vector<int> vis;
      std::vector<int> nis;
      std::string tok;
      while (ss >> tok) {
        int vi=0, ni=0;
        if (!parse_face_vertex(tok, vi, ni)) continue;

        if (vi < 0) vi = (int)V.size() + 1 + vi;
        if (ni < 0) ni = (int)N.size() + 1 + ni;

        vis.push_back(vi - 1);
        nis.push_back(ni - 1);
      }
      if (vis.size() < 3) continue;

      for (size_t i=1; i+1<vis.size(); i++) {
        if (vis[0] < 0 || vis[i] < 0 || vis[i+1] < 0) continue;
        if (vis[0] >= (int)V.size() || vis[i] >= (int)V.size() || vis[i+1] >= (int)V.size()) continue;

        Tri t{};
        t.v0 = V[vis[0]];
        t.v1 = V[vis[i]];
        t.v2 = V[vis[i+1]];

        bool hasN = (nis[0] >= 0 && nis[i] >= 0 && nis[i+1] >= 0
                     && nis[0] < (int)N.size() && nis[i] < (int)N.size() && nis[i+1] < (int)N.size());

        if (hasN) {
          t.n0 = N[nis[0]];
          t.n1 = N[nis[i]];
          t.n2 = N[nis[i+1]];
        } else {
          float3 fn = norm3(cross3(t.v1 - t.v0, t.v2 - t.v0));
          t.n0 = t.n1 = t.n2 = fn;
        }

        t.albedo = currentMaterial.Kd;
        t.specular = currentMaterial.Ks;
        t.glossiness = fmaxf(2.0f, currentMaterial.Ns);

        outTris.push_back(t);
      }
    }
  }

  if (outTris.empty()) {
    std::cerr << "OBJ loaded but no triangles found: " << path << "\n";
    return false;
  }

  return true;
}

static bool morph_topology_compatible(const std::vector<Tri>& a, const std::vector<Tri>& b){
  return a.size() == b.size();
}

// ============================
// GPU scene buffers
// ============================
struct GpuScene {
  Tri* d_tris = nullptr;
  Tri* d_baseTris = nullptr;
  Tri* d_shapeTris = nullptr;
  float* d_shapeWeights = nullptr;
  int* d_triIdx = nullptr;
  BVHNode* d_nodes = nullptr;

  int triCount = 0;
  int nodeCount = 0;
  int root = 0;
  int shapeCount = 0;
  bool morphReady = false;
};

static void free_scene(GpuScene& s){
  if (s.d_tris) CUDA_CHECK(cudaFree(s.d_tris));
  if (s.d_baseTris) CUDA_CHECK(cudaFree(s.d_baseTris));
  if (s.d_shapeTris) CUDA_CHECK(cudaFree(s.d_shapeTris));
  if (s.d_shapeWeights) CUDA_CHECK(cudaFree(s.d_shapeWeights));
  if (s.d_triIdx) CUDA_CHECK(cudaFree(s.d_triIdx));
  if (s.d_nodes) CUDA_CHECK(cudaFree(s.d_nodes));
  s = {};
}

static bool upload_scene(
  const std::vector<Tri>& tris,
  const std::vector<std::vector<Tri>>& shapes,
  GpuScene& out
){
  std::vector<int> triIdx(tris.size());
  for (int i=0;i<(int)triIdx.size();i++) triIdx[i]=i;

  std::vector<BVHNode> nodes;
  nodes.reserve(tris.size()*2);
  int root = build_bvh_recursive(nodes, triIdx, tris, shapes.empty() ? nullptr : &shapes, 0, (int)triIdx.size());

  out.triCount = (int)tris.size();
  out.nodeCount = (int)nodes.size();
  out.root = root;
  out.shapeCount = (int)shapes.size();
  out.morphReady = out.shapeCount > 0;

  CUDA_CHECK(cudaMalloc(&out.d_tris, sizeof(Tri)*out.triCount));
  CUDA_CHECK(cudaMemcpy(out.d_tris, tris.data(), sizeof(Tri)*out.triCount, cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&out.d_baseTris, sizeof(Tri)*out.triCount));
  CUDA_CHECK(cudaMemcpy(out.d_baseTris, tris.data(), sizeof(Tri)*out.triCount, cudaMemcpyHostToDevice));

  if (out.shapeCount > 0) {
    std::vector<Tri> flat;
    flat.reserve((size_t)out.shapeCount * (size_t)out.triCount);
    for (int s=0; s<out.shapeCount; ++s) {
      flat.insert(flat.end(), shapes[s].begin(), shapes[s].end());
    }

    CUDA_CHECK(cudaMalloc(&out.d_shapeTris, sizeof(Tri)*flat.size()));
    CUDA_CHECK(cudaMemcpy(out.d_shapeTris, flat.data(), sizeof(Tri)*flat.size(), cudaMemcpyHostToDevice));

    std::vector<float> zeroWeights(out.shapeCount, 0.0f);
    CUDA_CHECK(cudaMalloc(&out.d_shapeWeights, sizeof(float)*out.shapeCount));
    CUDA_CHECK(cudaMemcpy(out.d_shapeWeights, zeroWeights.data(), sizeof(float)*out.shapeCount, cudaMemcpyHostToDevice));
  }

  CUDA_CHECK(cudaMalloc(&out.d_triIdx, sizeof(int)*out.triCount));
  CUDA_CHECK(cudaMemcpy(out.d_triIdx, triIdx.data(), sizeof(int)*out.triCount, cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&out.d_nodes, sizeof(BVHNode)*out.nodeCount));
  CUDA_CHECK(cudaMemcpy(out.d_nodes, nodes.data(), sizeof(BVHNode)*out.nodeCount, cudaMemcpyHostToDevice));

  return true;
}

// ============================
// Rendering buffers
// ============================
static float4* d_accum = nullptr;
static uchar3* d_out = nullptr;
static uchar3* d_out_denoised = nullptr;
static uint64_t* d_seed = nullptr;
static float3* d_chainL = nullptr;
static int gW=0, gH=0;

static void init_renderer(int W, int H){
  gW=W; gH=H;
  CUDA_CHECK(cudaMalloc(&d_accum, sizeof(float4)*W*H));
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(uchar3)*W*H));
  CUDA_CHECK(cudaMalloc(&d_out_denoised, sizeof(uchar3)*W*H));
  CUDA_CHECK(cudaMalloc(&d_seed, sizeof(uint64_t)*W*H));
  CUDA_CHECK(cudaMalloc(&d_chainL, sizeof(float3)*W*H));
  CUDA_CHECK(cudaMemset(d_accum, 0, sizeof(float4)*W*H));
  CUDA_CHECK(cudaMemset(d_chainL, 0, sizeof(float3)*W*H));
  CUDA_CHECK(cudaMemset(d_out, 0, sizeof(uchar3)*W*H));
  CUDA_CHECK(cudaMemset(d_out_denoised, 0, sizeof(uchar3)*W*H));

  std::vector<uint64_t> seeds(W*H);
  uint64_t base = 1469598103934665603ULL;
  for (int i=0;i<W*H;i++){
    uint64_t s = base ^ (uint64_t)i * 1099511628211ULL;
    s ^= s >> 33; s *= 0xff51afd7ed558ccdULL;
    s ^= s >> 33; s *= 0xc4ceb9fe1a85ec53ULL;
    s ^= s >> 33;
    seeds[i] = s ? s : 1ULL;
  }
  CUDA_CHECK(cudaMemcpy(d_seed, seeds.data(), sizeof(uint64_t)*W*H, cudaMemcpyHostToDevice));
}

static void shutdown_renderer(){
  if (d_accum) CUDA_CHECK(cudaFree(d_accum));
  if (d_out) CUDA_CHECK(cudaFree(d_out));
  if (d_out_denoised) CUDA_CHECK(cudaFree(d_out_denoised));
  if (d_seed) CUDA_CHECK(cudaFree(d_seed));
  if (d_chainL) CUDA_CHECK(cudaFree(d_chainL));
  d_accum=nullptr; d_out=nullptr; d_out_denoised=nullptr; d_seed=nullptr; d_chainL=nullptr;
  gW=gH=0;
}

static void reset_accumulation(){
  if (d_accum) CUDA_CHECK(cudaMemset(d_accum, 0, sizeof(float4)*gW*gH));
  if (d_chainL) CUDA_CHECK(cudaMemset(d_chainL, 0, sizeof(float3)*gW*gH));
}

static void reset_mlt_chains_only(){
  if (d_chainL) CUDA_CHECK(cudaMemset(d_chainL, 0, sizeof(float3)*gW*gH));
}

// ============================
// Ray + intersection
// ============================
struct Ray { float3 o, d; };

__device__ inline bool intersect_aabb(const Ray& r, const float3& bmin, const float3& bmax, float tmax){
  float t0 = 0.0f;
  float t1 = tmax;

  for (int ax=0; ax<3; ax++){
    float ro = (ax==0)? r.o.x : (ax==1)? r.o.y : r.o.z;
    float rd = (ax==0)? r.d.x : (ax==1)? r.d.y : r.d.z;
    float mn = (ax==0)? bmin.x : (ax==1)? bmin.y : bmin.z;
    float mx = (ax==0)? bmax.x : (ax==1)? bmax.y : bmax.z;

    float inv = 1.0f / (fabsf(rd) > 1e-12f ? rd : (rd>=0?1e-12f:-1e-12f));
    float tNear = (mn - ro) * inv;
    float tFar  = (mx - ro) * inv;
    if (tNear > tFar) { float tmp=tNear; tNear=tFar; tFar=tmp; }
    t0 = fmaxf(t0, tNear);
    t1 = fminf(t1, tFar);
    if (t0 > t1) return false;
  }
  return true;
}

__device__ inline bool intersect_tri(const Ray& r, const Tri& t, float& outT, float& u, float& v){
  float3 e1 = t.v1 - t.v0;
  float3 e2 = t.v2 - t.v0;
  float3 p = cross3(r.d, e2);
  float det = dot3(e1, p);
  if (fabsf(det) < 1e-8f) return false;
  float inv = 1.0f / det;
  float3 s = r.o - t.v0;
  u = dot3(s, p) * inv;
  if (u < 0.0f || u > 1.0f) return false;
  float3 q = cross3(s, e1);
  v = dot3(r.d, q) * inv;
  if (v < 0.0f || (u+v) > 1.0f) return false;
  float tt = dot3(e2, q) * inv;
  if (tt <= 1e-4f) return false;
  outT = tt;
  return true;
}

struct Hit {
  float t;
  float3 p;
  float3 n;
  float3 albedo;
  float3 specular;
  float  glossiness;
  int hit;
};

__device__ inline Hit trace_scene(const GpuScene& S, const Ray& r, float floorY, float3 floorAlbedo){
  (void)floorY;
  (void)floorAlbedo;

  Hit best{};
  best.t = 1e30f;
  best.hit = 0;
  best.albedo = make_f3(0,0,0);
  best.specular = make_f3(0,0,0);
  best.glossiness = 1.0f;

  const int STACK_MAX = 64;
  int stack[STACK_MAX];
  int sp = 0;
  stack[sp++] = S.root;

  while (sp > 0) {
    int ni = stack[--sp];
    BVHNode node = S.d_nodes[ni];
    if (!intersect_aabb(r, node.bmin, node.bmax, best.t)) continue;

    if (node.left < 0 && node.right < 0) {
      for (int i=0;i<node.count;i++){
        int idx = S.d_triIdx[node.start + i];
        Tri t = S.d_tris[idx];
        float tt,u,v;
        if (intersect_tri(r, t, tt, u, v) && tt < best.t) {
          float w = 1.0f - u - v;
          float3 n = norm3(t.n0*w + t.n1*u + t.n2*v);
          best.t = tt;
          best.p = r.o + r.d * tt;
          best.n = n;
          best.albedo = t.albedo;
          best.specular = t.specular;
          best.glossiness = t.glossiness;
          best.hit = 1;
        }
      }
    } else {
      int L = node.left;
      int R = node.right;
      if (sp + 2 <= STACK_MAX) {
        stack[sp++] = L;
        stack[sp++] = R;
      }
    }
  }

  return best;
}

// ============================
// Sphere emitter
// ============================
struct SphereLight {
  float3 pos;
  float radius;
  float3 emission;
};

__device__ inline bool intersect_sphere_light(const Ray& r, const SphereLight& L, float& tHit, float3& pHit, float3& nHit){
  float3 oc = r.o - L.pos;
  float a = dot3(r.d, r.d);
  float b = 2.0f * dot3(oc, r.d);
  float c = dot3(oc, oc) - L.radius * L.radius;

  float disc = b*b - 4.0f*a*c;
  if (disc < 0.0f) return false;

  float sdisc = sqrtf(disc);
  float t0 = (-b - sdisc) / (2.0f * a);
  float t1 = (-b + sdisc) / (2.0f * a);

  float t = 1e30f;
  if (t0 > 1e-4f) t = t0;
  else if (t1 > 1e-4f) t = t1;
  else return false;

  tHit = t;
  pHit = r.o + r.d * t;
  nHit = norm3(pHit - L.pos);
  return true;
}

__device__ inline float3 sample_sphere_light_point(const SphereLight& L, uint64_t& rng){
  float u = rnd01(rng);
  float v = rnd01(rng);

  float z = 1.0f - 2.0f * u;
  float r = sqrtf(fmaxf(0.0f, 1.0f - z*z));
  float phi = 2.0f * (float)M_PI * v;

  float x = r * cosf(phi);
  float y = r * sinf(phi);

  return L.pos + make_f3(x, y, z) * L.radius;
}

__device__ inline bool visible_to_light_sample(const GpuScene& S, const float3& from, const float3& to){
  float3 d = to - from;
  float dist = len3(d);
  if (dist <= 1e-4f) return false;
  float3 dir = d / dist;

  Ray r;
  r.o = from;
  r.d = dir;

  Hit h = trace_scene(S, r, 0.0f, make_f3(0,0,0));
  if (h.hit && h.t < dist - 1e-3f) return false;
  return true;
}

// ============================
// Sampling helpers
// ============================
__device__ inline void build_onb(const float3& n, float3& t, float3& b){
  float3 up = (fabsf(n.z) < 0.999f) ? make_f3(0,0,1) : make_f3(0,1,0);
  t = norm3(cross3(up, n));
  b = cross3(n, t);
}

__device__ inline float3 cosine_hemisphere(const float3& n, uint64_t& rng){
  float r1 = rnd01(rng);
  float r2 = rnd01(rng);
  float phi = 2.0f * (float)M_PI * r1;
  float r = sqrtf(r2);
  float x = r * cosf(phi);
  float y = r * sinf(phi);
  float z = sqrtf(fmaxf(0.0f, 1.0f - r2));

  float3 t,b;
  build_onb(n, t, b);
  return norm3(t*x + b*y + n*z);
}

// ============================
// Transport params
// ============================
struct RenderParams {
  int W, H;
  int maxBounces;
  int spp;
  int useMLT;

  float3 camPos;
  float3 camFwd;
  float3 camRgt;
  float3 camUp;
  float fovY;

  float exposure;
  float persistAlpha;

  SphereLight light;
};

__device__ inline Ray make_camera_ray(const RenderParams& P, int x, int y, uint64_t& rng){
  float u = ((x + rnd01(rng)) / (float)P.W) * 2.0f - 1.0f;
  float v = ((y + rnd01(rng)) / (float)P.H) * 2.0f - 1.0f;

  float aspect = (float)P.W / (float)P.H;
  float tanF = tanf(0.5f * P.fovY);

  float3 dir = norm3(P.camFwd + P.camRgt * (u * aspect * tanF) + P.camUp * (-v * tanF));
  Ray r; r.o = P.camPos; r.d = dir;
  return r;
}

__device__ inline float3 direct_light_from_sphere(
  const GpuScene& S,
  const RenderParams& P,
  const float3& p,
  const float3& n,
  const float3& albedo,
  const float3& specular,
  float glossiness,
  const float3& viewDir,
  uint64_t& rng
){
  float3 lp = sample_sphere_light_point(P.light, rng);
  float3 toL = lp - p;
  float dist2 = dot3(toL, toL);
  if (dist2 <= 1e-8f) return make_f3(0,0,0);

  float dist = sqrtf(dist2);
  float3 wi = toL / dist;

  float cosSurface = fmaxf(0.0f, dot3(n, wi));
  if (cosSurface <= 0.0f) return make_f3(0,0,0);

  if (!visible_to_light_sample(S, p + n*1e-3f, lp)) {
    return make_f3(0,0,0);
  }

  float area = 4.0f * (float)M_PI * P.light.radius * P.light.radius;
  float pdfA = 1.0f / fmaxf(area, 1e-6f);

  float3 brdfDiffuse = albedo * (1.0f / (float)M_PI);

  float3 h = wi + viewDir;
  float hl = len3(h);
  float3 brdfSpec = make_f3(0,0,0);

  if (hl > 1e-8f) {
    h = h / hl;
    float ndh = fmaxf(0.0f, dot3(n, h));
    float specPow = powf(ndh, glossiness);
    brdfSpec = specular * specPow;
  }

  float3 brdf = brdfDiffuse + brdfSpec;
  return brdf * P.light.emission * cosSurface / fmaxf(dist2 * pdfA, 1e-6f);
}

__device__ inline float3 radiance_for_pixel(const GpuScene& S, const RenderParams& P, int px, int py, uint64_t& rng){
  Ray ray = make_camera_ray(P, px, py, rng);

  float3 L = make_f3(0,0,0);
  float3 T = make_f3(1,1,1);

  for (int bounce=0; bounce<P.maxBounces; bounce++){
    Hit h = trace_scene(S, ray, 0.0f, make_f3(0,0,0));

    float tLight;
    float3 pLight, nLight;
    bool hitLight = intersect_sphere_light(ray, P.light, tLight, pLight, nLight);

    bool sceneFirst = h.hit && (!hitLight || h.t < tLight);
    bool lightFirst = hitLight && (!h.hit || tLight < h.t);

    if (!sceneFirst && !lightFirst) {
      break;
    }

    if (lightFirst) {
      L = L + T * P.light.emission;
      break;
    }

    float3 n = h.n;
    float3 viewDir = norm3(-ray.d);

    float3 dl = direct_light_from_sphere(
      S, P,
      h.p, n,
      h.albedo,
      h.specular,
      h.glossiness,
      viewDir,
      rng
    );
    L = L + T * dl;

    float3 newDir = cosine_hemisphere(n, rng);
    ray.o = h.p + n * 1e-3f;
    ray.d = newDir;

    T = T * h.albedo;

    if (bounce >= 2) {
      float pRR = fmaxf(0.05f, fminf(0.95f, luminance(T)));
      if (rnd01(rng) > pRR) break;
      T = T / pRR;
    }
  }

  return L;
}

__global__ void k_render_pt(GpuScene S, RenderParams P, uint64_t frameIndex, float4* accum, uchar3* out, uint64_t* seeds){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= P.W || y >= P.H) return;
  int idx = y * P.W + x;

  uint64_t rng = seeds[idx] ^ (frameIndex * 0x9e3779b97f4a7c15ULL);

  float3 sum = make_f3(0,0,0);
  for (int s=0; s<P.spp; s++){
    sum = sum + radiance_for_pixel(S, P, x, y, rng);
  }
  seeds[idx] = rng;

  float3 c = sum / (float)P.spp;

  float4 a = accum[idx];
  float alpha = P.persistAlpha;
  a.x = a.x * alpha + c.x * (1.0f - alpha);
  a.y = a.y * alpha + c.y * (1.0f - alpha);
  a.z = a.z * alpha + c.z * (1.0f - alpha);
  a.w = 1.0f;
  accum[idx] = a;

  float3 avg = make_f3(a.x, a.y, a.z);
  float3 mapped = make_f3(1.0f,1.0f,1.0f) - make_f3(expf(-avg.x*P.exposure), expf(-avg.y*P.exposure), expf(-avg.z*P.exposure));
  mapped = clamp01f3(mapped);

  out[idx] = make_uchar3(
    (unsigned char)(mapped.z*255.0f),
    (unsigned char)(mapped.y*255.0f),
    (unsigned char)(mapped.x*255.0f)
  );
}

__device__ inline uint64_t mutate_seed_small(uint64_t s, uint64_t& rng){
  uint64_t r = xorshift64star(rng);
  uint64_t m = r & 0x0000FFFFFFFFFFFFULL;
  s ^= (m | 1ULL);
  s ^= (s << 7) ^ (s >> 9);
  return s ? s : 1ULL;
}

__global__ void k_render_metro(GpuScene S, RenderParams P, uint64_t frameIndex, float4* accum, uchar3* out, uint64_t* seeds, float3* chainL){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= P.W || y >= P.H) return;
  int idx = y * P.W + x;

  uint64_t curSeed = seeds[idx];
  uint64_t rng = (curSeed ^ (frameIndex * 0xD1342543DE82EF95ULL)) | 1ULL;

  float3 Lcur = chainL[idx];
  if (frameIndex == 0 || (Lcur.x==0 && Lcur.y==0 && Lcur.z==0)) {
    uint64_t tmp = curSeed ^ 0xA5A5A5A5A5A5A5A5ULL;
    Lcur = radiance_for_pixel(S, P, x, y, tmp);
  }

  uint64_t propSeed = mutate_seed_small(curSeed, rng);
  uint64_t tmp = propSeed ^ 0xC3C3C3C3C3C3C3C3ULL;
  float3 Lnew = radiance_for_pixel(S, P, x, y, tmp);

  float lc = fmaxf(1e-6f, luminance(Lcur));
  float ln = fmaxf(1e-6f, luminance(Lnew));
  float a = fminf(1.0f, ln / lc);

  if (rnd01(rng) < a) {
    curSeed = propSeed;
    Lcur = Lnew;
  }

  seeds[idx] = curSeed;
  chainL[idx] = Lcur;

  float3 c = Lcur;

  float4 acc = accum[idx];
  float alpha = P.persistAlpha;
  acc.x = acc.x * alpha + c.x * (1.0f - alpha);
  acc.y = acc.y * alpha + c.y * (1.0f - alpha);
  acc.z = acc.z * alpha + c.z * (1.0f - alpha);
  acc.w = 1.0f;
  accum[idx] = acc;

  float3 avg = make_f3(acc.x, acc.y, acc.z);
  float3 mapped = make_f3(1.0f,1.0f,1.0f) - make_f3(expf(-avg.x*P.exposure), expf(-avg.y*P.exposure), expf(-avg.z*P.exposure));
  mapped = clamp01f3(mapped);

  out[idx] = make_uchar3(
    (unsigned char)(mapped.z*255.0f),
    (unsigned char)(mapped.y*255.0f),
    (unsigned char)(mapped.x*255.0f)
  );
}

// ============================
// Cheap GPU denoise
// ============================
__device__ inline int fast_luma_bgr_u8(const uchar3& c){
  return (29 * (int)c.x + 150 * (int)c.y + 77 * (int)c.z) >> 8;
}

__global__ void k_cheap_denoise_contrast(
  const uchar3* src,
  uchar3* dst,
  int W,
  int H,
  int radius,
  int contrastThreshold
){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= W || y >= H) return;

  int minL = 255;
  int maxL = 0;
  int sumB = 0, sumG = 0, sumR = 0;
  int count = 0;

  for (int ky = -radius; ky <= radius; ++ky) {
    int yy = y + ky;
    if (yy < 0) yy = 0;
    if (yy >= H) yy = H - 1;

    for (int kx = -radius; kx <= radius; ++kx) {
      int xx = x + kx;
      if (xx < 0) xx = 0;
      if (xx >= W) xx = W - 1;

      uchar3 p = src[yy * W + xx];
      int l = fast_luma_bgr_u8(p);

      if (l < minL) minL = l;
      if (l > maxL) maxL = l;

      sumB += (int)p.x;
      sumG += (int)p.y;
      sumR += (int)p.z;
      count++;
    }
  }

  uchar3 center = src[y * W + x];
  int contrast = maxL - minL;

  if (contrast < contrastThreshold) {
    dst[y * W + x] = make_uchar3(
      (unsigned char)(sumB / count),
      (unsigned char)(sumG / count),
      (unsigned char)(sumR / count)
    );
  } else {
    dst[y * W + x] = center;
  }
}

// ============================
// Multi-shape morph kernel
// ============================
__global__ void k_apply_multi_morph(
  Tri* dst,
  const Tri* base,
  const Tri* shapeTris,
  const float* weights,
  int triCount,
  int shapeCount
){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= triCount) return;

  Tri b = base[i];
  Tri o = b;

  float3 v0 = b.v0;
  float3 v1 = b.v1;
  float3 v2 = b.v2;

  float3 n0 = b.n0;
  float3 n1 = b.n1;
  float3 n2 = b.n2;

  for (int s=0; s<shapeCount; ++s) {
    float w = weights[s];
    if (fabsf(w) < 1e-8f) continue;

    const Tri& m = shapeTris[s * triCount + i];

    v0 = v0 + (m.v0 - b.v0) * w;
    v1 = v1 + (m.v1 - b.v1) * w;
    v2 = v2 + (m.v2 - b.v2) * w;

    n0 = n0 + (m.n0 - b.n0) * w;
    n1 = n1 + (m.n1 - b.n1) * w;
    n2 = n2 + (m.n2 - b.n2) * w;
  }

  o.v0 = v0;
  o.v1 = v1;
  o.v2 = v2;

  o.n0 = norm3(n0);
  o.n1 = norm3(n1);
  o.n2 = norm3(n2);

  o.albedo = b.albedo;
  o.specular = b.specular;
  o.glossiness = b.glossiness;

  dst[i] = o;
}

// ============================
// X11 keyboard polling
// ============================
#ifdef __linux__
struct X11Keyboard {
  Display* dpy = nullptr;
  bool ok = false;

  void init() {
    dpy = XOpenDisplay(nullptr);
    ok = (dpy != nullptr);
  }

  void shutdown() {
    if (dpy) XCloseDisplay(dpy);
    dpy = nullptr;
    ok = false;
  }

  bool keyDown(KeySym ks) const {
    if (!ok) return false;
    char keys[32];
    XQueryKeymap(dpy, keys);
    KeyCode kc = XKeysymToKeycode(dpy, ks);
    if (kc == 0) return false;
    return (keys[kc >> 3] & (1 << (kc & 7))) != 0;
  }
};
#endif

// ============================
// UDP morph receiver
// ============================
#ifdef __linux__
struct UdpMorphReceiver {
  int sock = -1;
  bool ok = false;
  int port = 5005;
  std::vector<float> latestWeights;

  bool init(int listenPort, int shapeCount) {
    port = listenPort;
    latestWeights.assign(shapeCount, 0.0f);

    sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
      std::cerr << "UDP socket creation failed\n";
      return false;
    }

    int yes = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons((uint16_t)port);

    if (bind(sock, (sockaddr*)&addr, sizeof(addr)) < 0) {
      std::cerr << "UDP bind failed on 127.0.0.1:" << port << "\n";
      close(sock);
      sock = -1;
      return false;
    }

    int flags = fcntl(sock, F_GETFL, 0);
    if (flags >= 0) {
      fcntl(sock, F_SETFL, flags | O_NONBLOCK);
    }

    ok = true;
    std::cerr << "UDP morph receiver listening on 127.0.0.1:" << port << "\n";
    return true;
  }

  void shutdown() {
    if (sock >= 0) close(sock);
    sock = -1;
    ok = false;
  }

  bool poll(bool& changed) {
    changed = false;
    if (!ok) return false;

    char buf[4096];
    bool gotAny = false;

    for (;;) {
      ssize_t n = recv(sock, buf, sizeof(buf) - 1, 0);
      if (n <= 0) break;

      gotAny = true;
      buf[n] = '\0';
      std::string s(buf);
      s = trim(s);
      if (s.empty()) continue;

      std::vector<float> vals = parse_float_list_loose(s);
      if (vals.empty()) continue;

      size_t limit = std::min(vals.size(), latestWeights.size());
      for (size_t i = 0; i < limit; ++i) {
        float v = clamp01f_scalar(vals[i]);
        if (fabsf(v - latestWeights[i]) > 1e-6f) {
          latestWeights[i] = v;
          changed = true;
        }
      }
    }

    return gotAny;
  }
};
#endif

// ============================
// Camera
// ============================
struct Camera {
  float3 pos = make_f3(0, 0, 5);
  float fovY = 10.0f;

  float3 forward() const { return make_f3(0.0f, 0.0f, -1.0f); }
  float3 right()   const { return make_f3(1.0f, 0.0f, 0.0f); }
  float3 up()      const { return make_f3(0.0f, 1.0f, 0.0f); }
};

static bool nearly_equal_vectors(const std::vector<float>& a, const std::vector<float>& b, float eps = 1e-6f){
  if (a.size() != b.size()) return false;
  for (size_t i=0;i<a.size();++i){
    if (fabsf(a[i] - b[i]) > eps) return false;
  }
  return true;
}

static float pov_alpha_from_key(int key01to09){
  float t = (float)(key01to09 - 1) / 8.0f;
  return 0.98f * t;
}

int main(int argc, char** argv){
  std::string baseFolder   = (argc >= 2) ? argv[1] : "base";
  std::string shapesFolder = (argc >= 3) ? argv[2] : "shapes";

  std::string baseObjPath = (fs::path(baseFolder) / "base.obj").string();

  std::vector<Tri> tris;
  if (!load_obj(baseObjPath, tris)) {
    std::cerr << "Failed to load base OBJ: " << baseObjPath << "\n";
    return 1;
  }
  std::cerr << "Loaded base triangles: " << tris.size() << "\n";

  std::vector<std::string> shapePaths = list_obj_files_sorted(shapesFolder);
  std::vector<std::vector<Tri>> shapeTris;
  std::vector<std::string> shapeNames;

  for (const std::string& path : shapePaths) {
    std::vector<Tri> tmp;
    if (!load_obj(path, tmp)) {
      std::cerr << "Skipping shape (load failed): " << path << "\n";
      continue;
    }

    if (!morph_topology_compatible(tris, tmp)) {
      std::cerr << "Skipping incompatible shape: " << path << "\n";
      std::cerr << "Base tris: " << tris.size() << " | Shape tris: " << tmp.size() << "\n";
      continue;
    }

    shapeTris.push_back(std::move(tmp));
    shapeNames.push_back(fs::path(path).stem().string());
  }

  std::cerr << "Detected valid shapes: " << shapeTris.size() << "\n";
  for (size_t i=0; i<shapeNames.size(); ++i) {
    std::cerr << "  [" << i << "] " << shapeNames[i] << "\n";
  }

  GpuScene scene{};
  if (!upload_scene(tris, shapeTris, scene)) {
    std::cerr << "Failed to upload scene to GPU.\n";
    return 1;
  }
  std::cerr << "BVH nodes: " << scene.nodeCount << "\n";

  const int W = 540, H = 960;
  init_renderer(W, H);

  cv::namedWindow(WIN_NAME, cv::WINDOW_NORMAL);
  cv::resizeWindow(WIN_NAME, W, H);

#ifdef __linux__
  X11Keyboard xkb;
  xkb.init();

  UdpMorphReceiver morphRx;
  morphRx.init(5005, scene.shapeCount);
#endif

  Camera cam;
  cam.pos = make_f3(0.0f, 0.0f, 5.0f);

  float3 vel = make_f3(0,0,0);

  RenderParams P{};
  P.W = W;
  P.H = H;
  P.maxBounces = 4;
  P.spp = 1;
  P.useMLT = 0;

  const float exposureDefault = 1.2f;
  P.exposure = exposureDefault;

  float basePersistAlpha = 0.92f;
  P.persistAlpha = basePersistAlpha;

  const int morphSettleFramesMax = 10;
  const float morphSettleAlpha = 0.25f;
  int morphSettleFrames = 0;

  float3 lightPos = make_f3(0.0f, 4.0f, 2.0f);
  float  lightRadius = 3.5f;
  float3 lightColor = make_f3(1.0f, 1.0f, 1.0f);
  float  lightEnergyMultiplier = 1.0f;

  P.light.pos = lightPos;
  P.light.radius = lightRadius;
  P.light.emission = lightColor * lightEnergyMultiplier;

  std::cerr << "Sphere emitter:\n";
  std::cerr << "  pos=(" << lightPos.x << "," << lightPos.y << "," << lightPos.z << ")\n";
  std::cerr << "  radius=" << lightRadius << "\n";
  std::cerr << "  energy=" << lightEnergyMultiplier << "\n";

  std::vector<cv::Vec3b> hostFrame(W * H);

  bool denoiseOn = true;
  bool autoExposure = false;
  float autoTargetLuma = 0.22f;
  float autoStrength = 0.12f;

  cv::Mat disp8u(H, W, CV_8UC3);
  cv::Mat dispF(H, W, CV_32FC3);
  cv::Mat dispEMA = cv::Mat::zeros(H, W, CV_32FC3);

  float displayAlpha = 0.88f;

  // Cheap GPU denoise parameters
  int cheapDenoiseEvery = 25;       // apply every frame
  int cheapDenoiseRadius = 1;      // 1=3x3
  int cheapContrastThreshold = 2; // lower preserves more detail

  uint64_t frameIndex = 0;
  double lastT = (double)cv::getTickCount();

  bool fullscreen = false;

  std::vector<float> appliedWeights(scene.shapeCount, 0.0f);

  auto apply_current_weights_to_gpu = [&](const std::vector<float>& weights){
    if (!scene.morphReady || scene.shapeCount <= 0) return;

    CUDA_CHECK(cudaMemcpy(scene.d_shapeWeights, weights.data(), sizeof(float)*scene.shapeCount, cudaMemcpyHostToDevice));

    int block1D = 256;
    int grid1D = (scene.triCount + block1D - 1) / block1D;
    k_apply_multi_morph<<<grid1D, block1D>>>(
      scene.d_tris,
      scene.d_baseTris,
      scene.d_shapeTris,
      scene.d_shapeWeights,
      scene.triCount,
      scene.shapeCount
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  };

  if (scene.morphReady) {
    apply_current_weights_to_gpu(appliedWeights);
  }

  std::cerr << "Controls: WASD move | UDP morph on 127.0.0.1:5005 | R reset | M MLT | F fullscreen | N denoise | E auto exposure | [ ] exposure | 0 reset exposure | 1..9 persistence | ESC quit\n";

  for(;;){
    int key = cv::waitKey(1) & 0xFF;

    if (key == 'e' || key == 'E') {
      autoExposure = !autoExposure;
      std::cerr << "Auto-exposure: " << (autoExposure ? "ON" : "OFF") << "\n";
    }

    if (key >= '1' && key <= '9') {
      int k = key - '0';
      basePersistAlpha = pov_alpha_from_key(k);
      P.persistAlpha = basePersistAlpha;
      std::cerr << "POV level " << k << " -> persistAlpha=" << P.persistAlpha << "\n";
    }

    if (key == 27) break;

    if (key == 'r' || key == 'R') {
      reset_accumulation();
      frameIndex = 0;
      dispEMA.setTo(cv::Scalar(0,0,0));
      morphSettleFrames = 0;
    }

    if (key == 'm' || key == 'M') {
      P.useMLT = 1 - P.useMLT;
      reset_accumulation();
      frameIndex = 0;
      dispEMA.setTo(cv::Scalar(0,0,0));
      morphSettleFrames = 0;
      std::cerr << "Metropolis mode: " << (P.useMLT ? "ON" : "OFF") << "\n";
    }

    if (key == 'f' || key == 'F') {
      fullscreen = !fullscreen;
      cv::setWindowProperty(WIN_NAME, cv::WND_PROP_FULLSCREEN,
                            fullscreen ? cv::WINDOW_FULLSCREEN : cv::WINDOW_NORMAL);
    }

    if (key == 'n' || key == 'N') {
      denoiseOn = !denoiseOn;
      dispEMA.setTo(cv::Scalar(0,0,0));
      std::cerr << "Denoise: " << (denoiseOn ? "ON" : "OFF") << "\n";
    }

    if (key == '[') {
      P.exposure = std::max(0.05f, P.exposure * 0.90f);
      std::cerr << "Exposure: " << P.exposure << "\n";
    }

    if (key == ']') {
      P.exposure = std::min(20.0f, P.exposure * 1.10f);
      std::cerr << "Exposure: " << P.exposure << "\n";
    }

    if (key == '0') {
      P.exposure = exposureDefault;
      std::cerr << "Exposure: " << P.exposure << " (reset)\n";
    }

    double nowT = (double)cv::getTickCount();
    float dt = float((nowT - lastT) / cv::getTickFrequency());
    lastT = nowT;
    dt = std::min(dt, 0.05f);

    float wishF = 0.0f, wishR = 0.0f;

#ifdef __linux__
    if (xkb.ok) {
      if (xkb.keyDown(XK_w) || xkb.keyDown(XK_W)) wishF += 1.0f;
      if (xkb.keyDown(XK_s) || xkb.keyDown(XK_S)) wishF -= 1.0f;
      if (xkb.keyDown(XK_d) || xkb.keyDown(XK_D)) wishR += 1.0f;
      if (xkb.keyDown(XK_a) || xkb.keyDown(XK_A)) wishR -= 1.0f;
    } else {
      if (key == 'w' || key == 'W') wishF += 1.0f;
      if (key == 's' || key == 'S') wishF -= 1.0f;
      if (key == 'd' || key == 'D') wishR += 1.0f;
      if (key == 'a' || key == 'A') wishR -= 1.0f;
    }

    if (scene.morphReady) {
      bool changed = false;
      morphRx.poll(changed);

      if (changed && !nearly_equal_vectors(appliedWeights, morphRx.latestWeights)) {
        appliedWeights = morphRx.latestWeights;
        apply_current_weights_to_gpu(appliedWeights);

        // Do not reset visible accumulation.
        // Only invalidate MLT chain state and temporarily flush EMA faster.
        reset_mlt_chains_only();
        morphSettleFrames = morphSettleFramesMax;
      }
    }
#else
    if (key == 'w' || key == 'W') wishF += 1.0f;
    if (key == 's' || key == 'S') wishF -= 1.0f;
    if (key == 'd' || key == 'D') wishR += 1.0f;
    if (key == 'a' || key == 'A') wishR -= 1.0f;
#endif

    float mag = sqrtf(wishF*wishF + wishR*wishR);
    if (mag > 1.0f) {
      wishF /= mag;
      wishR /= mag;
    }

    float3 fwd = cam.forward();
    float3 rgt = cam.right();

    float walkSpeed = 1.6f;
    float accel = 10.0f;
    float damping = 8.0f;

    float3 wishVel = (fwd * wishF + rgt * wishR) * walkSpeed;
    vel = vel + (wishVel - vel) * (1.0f - expf(-accel * dt));
    vel = vel * expf(-damping * dt);

    cam.pos = cam.pos + vel * dt;
    cam.pos.y = 0.0f;

    P.camPos = cam.pos;
    P.camFwd = cam.forward();
    P.camRgt = cam.right();
    P.camUp  = cam.up();
    P.fovY = cam.fovY * (float)M_PI / 180.0f;

    if (morphSettleFrames > 0) {
      P.persistAlpha = morphSettleAlpha;
      morphSettleFrames--;
    } else {
      P.persistAlpha = basePersistAlpha;
    }

    dim3 block(16,16);
    dim3 grid((W + block.x - 1)/block.x, (H + block.y - 1)/block.y);

    if (P.useMLT) {
      k_render_metro<<<grid, block>>>(scene, P, frameIndex, d_accum, d_out, d_seed, d_chainL);
    } else {
      k_render_pt<<<grid, block>>>(scene, P, frameIndex, d_accum, d_out, d_seed);
    }
    CUDA_CHECK(cudaGetLastError());

    uchar3* d_to_copy = d_out;

    if (denoiseOn && cheapDenoiseEvery > 0 && (int)(frameIndex % (uint64_t)cheapDenoiseEvery) == 0) {
      k_cheap_denoise_contrast<<<grid, block>>>(
        d_out,
        d_out_denoised,
        W,
        H,
        cheapDenoiseRadius,
        cheapContrastThreshold
      );
      CUDA_CHECK(cudaGetLastError());
      d_to_copy = d_out_denoised;
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hostFrame.data(), d_to_copy, sizeof(uchar3)*W*H, cudaMemcpyDeviceToHost));
    cv::Mat raw8u(H, W, CV_8UC3, hostFrame.data());

    if (autoExposure) {
      cv::Mat tmpF;
      raw8u.convertTo(tmpF, CV_32FC3, 1.0/255.0);

      std::vector<cv::Mat> ch(3);
      cv::split(tmpF, ch);

      cv::Mat luma = 0.0722f * ch[0] + 0.7152f * ch[1] + 0.2126f * ch[2];

      double meanL = cv::mean(luma)[0];
      float m = (float)meanL;

      const float eps = 1e-4f;
      float ratio = autoTargetLuma / (m + eps);
      float factor = powf(ratio, autoStrength);

      P.exposure = std::min(20.0f, std::max(0.05f, P.exposure * factor));
    }

    if (!denoiseOn) {
      cv::imshow(WIN_NAME, raw8u);
    } else {
      raw8u.convertTo(dispF, CV_32FC3, 1.0/255.0);
      cv::addWeighted(dispEMA, displayAlpha, dispF, (1.0f - displayAlpha), 0.0, dispEMA);
      dispEMA.convertTo(disp8u, CV_8UC3, 255.0);
      cv::imshow(WIN_NAME, disp8u);
    }

    frameIndex++;
  }

#ifdef __linux__
  morphRx.shutdown();
  xkb.shutdown();
#endif

  shutdown_renderer();
  free_scene(scene);
  return 0;
}
