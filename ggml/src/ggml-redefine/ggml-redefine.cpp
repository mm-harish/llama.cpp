#define CL_TARGET_OPENCL_VERSION GGML_REDEFINE_TARGET_VERSION
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

// suppress warnings in CL headers for GCC and Clang
#pragma GCC diagnostic ignored "-Woverlength-strings"
#ifdef __clang__
#pragma GCC diagnostic ignored "-Wgnu-anonymous-struct"
#endif

#include "ggml-redefine.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml.h"

#include <CL/cl.h>

#include <inttypes.h>
#include <string.h>

#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <string>
#include <vector>

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

#define CL_CHECK(err)                                                          \
  do {                                                                         \
    cl_int err_ = (err);                                                       \
    if (err_ != CL_SUCCESS) {                                                  \
      GGML_LOG_ERROR("ggml_redefine: %s error %d at %s:%d\n", #err, err_,      \
                     __FILE__, __LINE__);                                      \
      GGML_ASSERT(0);                                                          \
    }                                                                          \
  } while (0)

struct ggml_cl_version {
  cl_uint major = 0;
  cl_uint minor = 0;
};

// cl buffer wrapper
struct ggml_cl_buffer {
  cl_mem buffer;
  size_t size;

  ggml_cl_buffer() : buffer(nullptr), size(0) {}

  ~ggml_cl_buffer() {
    if (buffer) {
      CL_CHECK(clReleaseMemObject(buffer));
    }
  }

  void allocate(cl_context context, size_t new_size) {
    if (new_size > size) {
      size = new_size;
      if (buffer) {
        CL_CHECK(clReleaseMemObject(buffer));
      }
      cl_int err;
      CL_CHECK((
          buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &err),
          err));
    }
  }
};

// Buffer context for device buffers
struct ggml_backend_redefine_buffer_context {
  cl_mem buffer;

  ggml_backend_redefine_buffer_context(cl_mem buf) : buffer(buf) {}

  ~ggml_backend_redefine_buffer_context() {
    if (buffer) {
      CL_CHECK(clReleaseMemObject(buffer));
    }
  }
};

struct ggml_backend_redefine_context;

// backend device context
struct ggml_backend_redefine_device_context {
  cl_platform_id platform;
  std::string platform_name;

  cl_device_id device;
  std::string device_name;
  cl_device_type device_type;
  std::string device_version;

  // Initialized by ggml_cl2_init().
  ggml_backend_redefine_context *backend_ctx = nullptr;

  // Initialized by ggml_backend_redefine_device_get_buffer_type()
  ggml_backend_buffer_type buffer_type;

  cl_context context = nullptr;

  std::regex *opfilter = nullptr; // regex of ops to not claim
  std::string opfilter_str;       // regex string for opfilter
  size_t global_mem_size = 0;
};

// backend context
struct ggml_backend_redefine_context {
  int ref_count;

  cl_device_id device;
  std::string device_name;

  ggml_cl_version platform_version;
  ggml_cl_version opencl_c_version;

  // rest of the kernels are currently always loaded in alloc_buffer.
  bool kernels_loaded = false;

  std::string driver_version;

  cl_int alignment;
  size_t global_mem_size;
  size_t max_alloc_size;
  size_t max_workgroup_size;

  cl_context context;
  cl_command_queue queue;

  // prealloc buffers for transposing weights and activations
  ggml_cl_buffer prealloc_quant_trans;
  ggml_cl_buffer prealloc_scales_trans;
  ggml_cl_buffer prealloc_act_trans;

  // prealloc buffers for src0 and src1
  ggml_cl_buffer prealloc_src0;
  ggml_cl_buffer prealloc_src1;

  // prealloc buffers for MoE router table preprocess
  bool toggle_reorder = false;
  ggml_cl_buffer prealloc_post_router;
  ggml_cl_buffer prealloc_emap;
  ggml_cl_buffer prealloc_hist;
  ggml_cl_buffer prealloc_tile_offset;
  ggml_cl_buffer prealloc_total_tiles;
  ggml_cl_buffer prealloc_slot_counter;

  void enqueue_ndrange_kernel(cl_kernel kernel, cl_uint work_dim,
                              size_t *global_work_size, size_t *local_work_size,
                              const ggml_tensor *tensor) {

    GGML_UNUSED(tensor);
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, work_dim, NULL,
                                    global_work_size, local_work_size, 0, NULL,
                                    NULL));
  }

  void free() {
    clFinish(queue);

    ref_count--;
    // if (ref_count <= 0) {
    //   // CL_CHECK(clReleaseCommandQueue(queue));
    //   // CL_CHECK(clReleaseContext(context));
    //   // delete this;
    // }
  }
};

// forward declarations for backend API entry points
static ggml_backend_redefine_context *
ggml_redefine_init(ggml_backend_dev_t dev);

static void ggml_redefine_free(ggml_backend_t backend) {
  ggml_backend_redefine_context *ctx =
      (ggml_backend_redefine_context *)backend->context;
  ctx->free();
}

//------------------------------------------------------------------------------
// Backend API
//------------------------------------------------------------------------------

//
// backend
//
static const char *ggml_backend_redefine_name(ggml_backend_t backend) {
  GGML_UNUSED(backend);
  return "REDEFINE";
}

static void ggml_backend_redefine_free(ggml_backend_t backend) {
  ggml_redefine_free(backend);
}

static void ggml_backend_redefine_synchronize(ggml_backend_t backend) {
  auto *backend_ctx =
      static_cast<ggml_backend_redefine_context *>(backend->context);

  cl_event evt;
  CL_CHECK(clEnqueueBarrierWithWaitList(backend_ctx->queue, 0, nullptr, &evt));
  CL_CHECK(clWaitForEvents(1, &evt));
  CL_CHECK(clReleaseEvent(evt));
}

static ggml_status ggml_backend_redefine_graph_compute(ggml_backend_t backend,
                                                       ggml_cgraph *cgraph) {
  GGML_UNUSED(backend);
  GGML_UNUSED(cgraph);

  GGML_LOG_WARN("%s: not implemented yet\n", __func__);

  return GGML_STATUS_SUCCESS;
}

static void ggml_backend_redefine_set_tensor_async(ggml_backend_t backend,
                                                   struct ggml_tensor *tensor,
                                                   const void *data,
                                                   size_t offset, size_t size) {
  GGML_UNUSED(backend);
  GGML_UNUSED(tensor);
  GGML_UNUSED(data);
  GGML_UNUSED(offset);
  GGML_UNUSED(size);
  GGML_LOG_WARN("%s: not implemented yet\n", __func__);
}

static void
ggml_backend_redefine_get_tensor_async(ggml_backend_t backend,
                                       const struct ggml_tensor *tensor,
                                       void *data, size_t offset, size_t size) {
  GGML_UNUSED(backend);
  GGML_UNUSED(tensor);
  GGML_UNUSED(data);
  GGML_UNUSED(offset);
  GGML_UNUSED(size);
  GGML_LOG_WARN("%s: not implemented yet\n", __func__);
}

static void ggml_backend_redefine_set_tensor_2d_async(
    ggml_backend_t backend, struct ggml_tensor *tensor, const void *data,
    size_t offset, size_t size, size_t n_copies, size_t stride_tensor,
    size_t stride_data) {
  GGML_UNUSED(backend);
  GGML_UNUSED(tensor);
  GGML_UNUSED(data);
  GGML_UNUSED(offset);
  GGML_UNUSED(size);
  GGML_UNUSED(n_copies);
  GGML_UNUSED(stride_tensor);
  GGML_UNUSED(stride_data);
  GGML_LOG_WARN("%s: not implemented yet\n", __func__);
}

static void ggml_backend_redefine_get_tensor_2d_async(
    ggml_backend_t backend, const struct ggml_tensor *tensor, void *data,
    size_t offset, size_t size, size_t n_copies, size_t stride_tensor,
    size_t stride_data) {
  GGML_UNUSED(backend);
  GGML_UNUSED(tensor);
  GGML_UNUSED(data);
  GGML_UNUSED(offset);
  GGML_UNUSED(size);
  GGML_UNUSED(n_copies);
  GGML_UNUSED(stride_tensor);
  GGML_UNUSED(stride_data);
  GGML_LOG_WARN("%s: not implemented yet\n", __func__);
}

static bool ggml_backend_redefine_cpy_tensor_async(
    ggml_backend_t backend_src, ggml_backend_t backend_dst,
    const struct ggml_tensor *src, struct ggml_tensor *dst) {
  GGML_UNUSED(backend_src);
  GGML_UNUSED(backend_dst);
  GGML_UNUSED(src);
  GGML_UNUSED(dst);
  GGML_LOG_WARN("%s: not implemented yet\n", __func__);
  return false;
}

static ggml_backend_graph_plan_t
ggml_backend_redefine_graph_plan_create(ggml_backend_t backend,
                                        const struct ggml_cgraph *cgraph) {
  GGML_UNUSED(backend);
  GGML_UNUSED(cgraph);
  GGML_LOG_WARN("%s: not implemented yet\n", __func__);
  return NULL;
}

static void
ggml_backend_redefine_graph_plan_free(ggml_backend_t backend,
                                      ggml_backend_graph_plan_t plan) {
  GGML_UNUSED(backend);
  GGML_UNUSED(plan);
  GGML_LOG_WARN("%s: not implemented yet\n", __func__);
}

static void
ggml_backend_redefine_graph_plan_update(ggml_backend_t backend,
                                        ggml_backend_graph_plan_t plan,
                                        const struct ggml_cgraph *cgraph) {
  GGML_UNUSED(backend);
  GGML_UNUSED(plan);
  GGML_UNUSED(cgraph);
  GGML_LOG_WARN("%s: not implemented yet\n", __func__);
}

static ggml_status
ggml_backend_redefine_graph_plan_compute(ggml_backend_t backend,
                                         ggml_backend_graph_plan_t plan) {
  GGML_UNUSED(backend);
  GGML_UNUSED(plan);
  GGML_LOG_WARN("%s: not implemented yet\n", __func__);
  return GGML_STATUS_SUCCESS;
}

static void ggml_backend_redefine_event_record(ggml_backend_t backend,
                                               ggml_backend_event_t event) {
  GGML_UNUSED(backend);
  GGML_UNUSED(event);
  GGML_LOG_WARN("%s: not implemented yet\n", __func__);
}

static void ggml_backend_redefine_event_wait(ggml_backend_t backend,
                                             ggml_backend_event_t event) {
  GGML_UNUSED(backend);
  GGML_UNUSED(event);
  GGML_LOG_WARN("%s: not implemented yet\n", __func__);
}

static void ggml_backend_redefine_graph_optimize(ggml_backend_t backend,
                                                 struct ggml_cgraph *cgraph) {
  GGML_UNUSED(backend);
  GGML_UNUSED(cgraph);
  GGML_LOG_WARN("%s: not implemented yet\n", __func__);
}

static ggml_backend_i ggml_backend_redefine_i = {
    /* .get_name                = */ ggml_backend_redefine_name,
    /* .free                    = */ ggml_backend_redefine_free,
    /* .set_tensor_async        = */
    ggml_backend_redefine_set_tensor_async, /* ggml_backend_redefine_set_tensor_async
                                             */
                                            /* .get_tensor_async        = */
    ggml_backend_redefine_get_tensor_async, /* ggml_backend_redefine_get_tensor_async
                                             */
    /* .set_tensor_2d_async     = */ ggml_backend_redefine_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_redefine_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ggml_backend_redefine_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_redefine_synchronize,
    /* .graph_plan_create       = */ ggml_backend_redefine_graph_plan_create,
    /* .graph_plan_free         = */ ggml_backend_redefine_graph_plan_free,
    /* .graph_plan_update       = */ ggml_backend_redefine_graph_plan_update,
    /* .graph_plan_compute      = */ ggml_backend_redefine_graph_plan_compute,
    /* .graph_compute           = */ ggml_backend_redefine_graph_compute,
    /* .event_record            = */ ggml_backend_redefine_event_record,
    /* .event_wait              = */ ggml_backend_redefine_event_wait,
    /* .graph_optimize          = */ ggml_backend_redefine_graph_optimize,
};

// Backend registry
// All registered devices with a default device in the front.
static std::vector<ggml_backend_device> g_ggml_backend_redefine_devices;

// All device contexts associated with the devices above.
// The devices live as long as the process, so do the contexts.
static std::vector<std::unique_ptr<ggml_backend_redefine_device_context>>
    g_ggml_backend_redefine_dev_ctxs;

static const char *ggml_backend_redefine_reg_get_name(ggml_backend_reg_t reg) {
  return "REDEFINE";

  GGML_UNUSED(reg);
}

static size_t ggml_backend_redefine_reg_device_count(ggml_backend_reg_t reg) {
  return g_ggml_backend_redefine_devices.size();

  GGML_UNUSED(reg);
}

static ggml_backend_dev_t
ggml_backend_redefine_reg_device_get(ggml_backend_reg_t reg, size_t index) {
  GGML_ASSERT(index < ggml_backend_redefine_reg_device_count(reg));

  return &g_ggml_backend_redefine_devices[index];

  GGML_UNUSED(reg);
  GGML_UNUSED(index);
}

static struct ggml_backend_reg_i ggml_backend_redefine_reg_i = {
    /* .get_name         = */ ggml_backend_redefine_reg_get_name,
    /* .device_count     = */ ggml_backend_redefine_reg_device_count,
    /* .device_get       = */ ggml_backend_redefine_reg_device_get,
    /* .get_proc_address = */ NULL,
};

namespace /* anonymous */ {

std::string ggml_cl_get_platform_info_string(cl_platform_id platform,
                                             cl_platform_info param) {
  size_t size = 0;
  CL_CHECK(clGetPlatformInfo(platform, param, 0, nullptr, &size));

  std::vector<char> buffer(size);
  CL_CHECK(clGetPlatformInfo(platform, param, size, buffer.data(), nullptr));

  return std::string(buffer.data());
}

std::string ggml_cl_get_device_info_string(cl_device_id device,
                                           cl_device_info param) {
  size_t size = 0;
  CL_CHECK(clGetDeviceInfo(device, param, 0, nullptr, &size));

  std::vector<char> buffer(size);
  CL_CHECK(clGetDeviceInfo(device, param, size, buffer.data(), nullptr));

  return std::string(buffer.data());
}

bool ggml_redefine_platform_supported(const std::string &platform_name) {
  return platform_name.find("pocl") != std::string::npos ||
         platform_name.find("Portable Computing Language") != std::string::npos;
}

bool ggml_redefine_device_supported(const std::string &device_name) {
  return device_name.find("isasim") != std::string::npos ||
         device_name.find("ISA Simulator") != std::string::npos ||
         device_name.find("fsim") != std::string::npos ||
         device_name.find("Functional Simulator") != std::string::npos ||
         device_name.find("redefine") != std::string::npos;
}

const char *ggml_backend_redefine_device_get_name(ggml_backend_dev_t dev) {
  const auto *dev_ctx =
      static_cast<const ggml_backend_redefine_device_context *>(dev->context);
  return dev_ctx->device_name.c_str();
}

const char *
ggml_backend_redefine_device_get_description(ggml_backend_dev_t dev) {
  const auto *dev_ctx =
      static_cast<const ggml_backend_redefine_device_context *>(dev->context);
  return dev_ctx->device_version.c_str();
}

enum ggml_backend_dev_type
ggml_backend_redefine_device_get_type(ggml_backend_dev_t dev) {
  GGML_UNUSED(dev);
  return GGML_BACKEND_DEVICE_TYPE_GPU;
}

void ggml_backend_redefine_device_get_memory(ggml_backend_dev_t dev,
                                             size_t *free, size_t *total) {
  const auto *dev_ctx =
      static_cast<const ggml_backend_redefine_device_context *>(dev->context);
  if (total) {
    *total = dev_ctx->global_mem_size;
  }

  // OpenCL 1.x does not provide portable free-memory query per device.
  // Report total as available memory for scheduling heuristics.
  if (free) {
    *free = dev_ctx->global_mem_size;
  }
}

void ggml_backend_redefine_device_get_props(
    ggml_backend_dev_t dev, struct ggml_backend_dev_props *props) {
  props->name = ggml_backend_redefine_device_get_name(dev);
  props->description = ggml_backend_redefine_device_get_description(dev);
  props->type = ggml_backend_redefine_device_get_type(dev);
  ggml_backend_redefine_device_get_memory(dev, &props->memory_free,
                                          &props->memory_total);
  props->caps = ggml_backend_dev_caps{
      /* .async                 = */ false,
      /* .host_buffer           = */ false,
      /* .buffer_from_host_ptr  = */ false,
      /* .events                = */ false,
  };
}

//
// Buffer interface
//

void ggml_backend_redefine_buffer_free_buffer(ggml_backend_buffer_t buffer) {
  ggml_backend_redefine_buffer_context *ctx =
      (ggml_backend_redefine_buffer_context *)buffer->context;
  delete ctx;
}

void *ggml_backend_redefine_buffer_get_base(ggml_backend_buffer_t buffer) {
  ggml_backend_redefine_device_context *dev_ctx =
      (ggml_backend_redefine_device_context *)buffer->buft->device->context;
  // Return an offset address based on device alignment
  char *base = nullptr;
  return (void *)(base + dev_ctx->backend_ctx->alignment);
}

ggml_status
ggml_backend_redefine_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                         ggml_tensor *tensor) {
  GGML_UNUSED(buffer);

  if (tensor->view_src != nullptr) {
    GGML_ASSERT(tensor->view_src->buffer->buft == buffer->buft);
    tensor->extra = tensor->view_src->extra;
  } else {
    // Store offset information in extra
    GGML_UNUSED(buffer);
  }

  return GGML_STATUS_SUCCESS;
}

void ggml_backend_redefine_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                             ggml_tensor *tensor,
                                             const void *data, size_t offset,
                                             size_t size) {
  GGML_UNUSED(tensor);

  ggml_backend_redefine_device_context *dev_ctx =
      (ggml_backend_redefine_device_context *)buffer->buft->device->context;
  ggml_backend_redefine_context *backend_ctx = dev_ctx->backend_ctx;
  ggml_backend_redefine_buffer_context *buf_ctx =
      (ggml_backend_redefine_buffer_context *)buffer->context;

  cl_command_queue queue = backend_ctx->queue;

  CL_CHECK(clEnqueueWriteBuffer(queue, buf_ctx->buffer, CL_TRUE, offset, size,
                                data, 0, NULL, NULL));
}

void ggml_backend_redefine_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                             const ggml_tensor *tensor,
                                             void *data, size_t offset,
                                             size_t size) {
  GGML_UNUSED(tensor);

  ggml_backend_redefine_device_context *dev_ctx =
      (ggml_backend_redefine_device_context *)buffer->buft->device->context;
  ggml_backend_redefine_context *backend_ctx = dev_ctx->backend_ctx;
  ggml_backend_redefine_buffer_context *buf_ctx =
      (ggml_backend_redefine_buffer_context *)buffer->context;

  cl_command_queue queue = backend_ctx->queue;

  CL_CHECK(clEnqueueReadBuffer(queue, buf_ctx->buffer, CL_TRUE, offset, size,
                               data, 0, NULL, NULL));
}

void ggml_backend_redefine_memset_tensor(ggml_backend_buffer_t buffer,
                                         struct ggml_tensor *tensor,
                                         uint8_t value, size_t offset,
                                         size_t size) {
  GGML_UNUSED(buffer);
  GGML_UNUSED(tensor);
  GGML_UNUSED(value);
  GGML_UNUSED(offset);
  GGML_UNUSED(size);
  GGML_LOG_WARN("%s: not implemented yet\n", __func__);
}

void ggml_backend_redefine_set_tensor_2d(ggml_backend_buffer_t buffer,
                                         struct ggml_tensor *tensor,
                                         const void *data, size_t offset,
                                         size_t size, size_t n_copies,
                                         size_t stride_tensor,
                                         size_t stride_data) {
  GGML_UNUSED(buffer);
  GGML_UNUSED(tensor);
  GGML_UNUSED(data);
  GGML_UNUSED(offset);
  GGML_UNUSED(size);
  GGML_UNUSED(n_copies);
  GGML_UNUSED(stride_tensor);
  GGML_UNUSED(stride_data);
  GGML_LOG_WARN("%s: not implemented yet\n", __func__);
}

void ggml_backend_redefine_get_tensor_2d(ggml_backend_buffer_t buffer,
                                         const struct ggml_tensor *tensor,
                                         void *data, size_t offset, size_t size,
                                         size_t n_copies, size_t stride_tensor,
                                         size_t stride_data) {
  GGML_UNUSED(buffer);
  GGML_UNUSED(tensor);
  GGML_UNUSED(data);
  GGML_UNUSED(offset);
  GGML_UNUSED(size);
  GGML_UNUSED(n_copies);
  GGML_UNUSED(stride_tensor);
  GGML_UNUSED(stride_data);
  GGML_LOG_WARN("%s: not implemented yet\n", __func__);
}

bool ggml_backend_redefine_cpy_tensor(ggml_backend_buffer_t buffer,
                                      const struct ggml_tensor *src,
                                      struct ggml_tensor *dst) {
  GGML_UNUSED(buffer);
  GGML_UNUSED(src);
  GGML_UNUSED(dst);
  GGML_LOG_WARN("%s: not implemented yet\n", __func__);
  return false;
}

void ggml_backend_redefine_clear(ggml_backend_buffer_t buffer, uint8_t value) {
  GGML_UNUSED(buffer);
  GGML_UNUSED(value);
  GGML_LOG_WARN("%s: not implemented yet\n", __func__);
}

void ggml_backend_redefine_reset(ggml_backend_buffer_t buffer) {
  GGML_UNUSED(buffer);
  GGML_LOG_WARN("%s: not implemented yet\n", __func__);
}

ggml_backend_buffer_i ggml_backend_redefine_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_redefine_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_redefine_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_redefine_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_redefine_memset_tensor,
    /* .set_tensor      = */ ggml_backend_redefine_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_redefine_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_redefine_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_redefine_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_redefine_cpy_tensor,
    /* .clear           = */ ggml_backend_redefine_clear,
    /* .reset           = */ ggml_backend_redefine_reset,
};

//
// Buffer type interface
//

const char *ggml_backend_redefine_buffer_type_get_name(
    ggml_backend_buffer_type_t buffer_type) {
  GGML_UNUSED(buffer_type);
  return "REDEFINE";
}

ggml_backend_buffer_t ggml_backend_redefine_buffer_type_alloc_buffer(
    ggml_backend_buffer_type_t buffer_type, size_t size) {
  GGML_LOG_INFO("[REDEFINE] alloc_buffer size = %zu\n", size);
  ggml_backend_redefine_context *backend_ctx =
      ggml_redefine_init(buffer_type->device);

  // clCreateBuffer returns -61 for size 0
  size = std::max(size, (size_t)1);

  cl_int err;
  cl_mem mem =
      clCreateBuffer(backend_ctx->context, CL_MEM_READ_WRITE, size, NULL, &err);

  if (err != CL_SUCCESS) {
    GGML_LOG_INFO("%s: failed to allocate %.2f MiB\n", __func__,
                  size / 1024.0 / 1024.0);
    return nullptr;
  }

  ggml_backend_redefine_buffer_context *ctx =
      new ggml_backend_redefine_buffer_context(mem);

  return ggml_backend_buffer_init(
      buffer_type, ggml_backend_redefine_buffer_interface, ctx, size);
}

size_t ggml_backend_redefine_buffer_type_get_alignment(
    ggml_backend_buffer_type_t buffer_type) {
  ggml_backend_redefine_device_context *dev_ctx =
      (ggml_backend_redefine_device_context *)buffer_type->device->context;
  if (dev_ctx->backend_ctx) {
    return dev_ctx->backend_ctx->alignment;
  }
  return 128; // default alignment
}

size_t ggml_backend_redefine_buffer_type_get_max_size(
    ggml_backend_buffer_type_t buffer_type) {
  ggml_backend_redefine_device_context *dev_ctx =
      (ggml_backend_redefine_device_context *)buffer_type->device->context;
  if (dev_ctx->backend_ctx) {
    return dev_ctx->backend_ctx->max_alloc_size;
  }
  return dev_ctx->global_mem_size;
}

size_t align_up(size_t x, size_t a) { return (x + a - 1) & ~(a - 1); }

size_t ggml_backend_redefine_buffer_type_get_alloc_size(
    ggml_backend_buffer_type_t buft, const struct ggml_tensor *tensor) {

  GGML_UNUSED(buft);

  return align_up(ggml_nbytes(tensor), 64);
}

bool ggml_backend_redefine_buffer_type_is_host(
    ggml_backend_buffer_type_t buft) {
  GGML_UNUSED(buft);
  GGML_LOG_WARN("%s: not implemented yet\n", __func__);
  return false;
}

ggml_backend_buffer_type_i ggml_backend_redefine_buffer_type_i = {
    /* .get_name       = */ ggml_backend_redefine_buffer_type_get_name,
    /* .alloc_buffer   = */ ggml_backend_redefine_buffer_type_alloc_buffer,
    /* .get_alignment  = */ ggml_backend_redefine_buffer_type_get_alignment,
    /* .get_max_size   = */ ggml_backend_redefine_buffer_type_get_max_size,
    /* .get_alloc_size = */ ggml_backend_redefine_buffer_type_get_alloc_size,
    /* .is_host        = */ NULL,
};

//
// Device interface
//
static const char *
ggml_redefine_tensor_type_name(const struct ggml_tensor *tensor) {
  return tensor ? ggml_type_name(tensor->type) : "NULL";
}

static void ggml_redefine_log_op_info(const struct ggml_tensor *op,
                                      bool supported) {
  GGML_LOG_DEBUG(
      "ggml_redefine: %s op %s (%s) name=%s dst=%s src0=%s src1=%s src2=%s\n",
      supported ? "supports" : "rejects", ggml_op_name(op->op),
      ggml_op_desc(op), op->name, ggml_redefine_tensor_type_name(op),
      ggml_redefine_tensor_type_name(op->src[0]),
      ggml_redefine_tensor_type_name(op->src[1]),
      ggml_redefine_tensor_type_name(op->src[2]));
}

bool ggml_redefine_supports_op(ggml_backend_dev_t dev,
                               const struct ggml_tensor *op) {

  GGML_UNUSED(dev);
  return true;

  bool supported = false;

  switch (op->op) {
  case GGML_OP_NONE:
    supported = true;
    break;

  // tensor metadata ops
  case GGML_OP_RESHAPE:
  case GGML_OP_VIEW:
  case GGML_OP_PERMUTE:
  case GGML_OP_TRANSPOSE:
  case GGML_OP_CONT:
  case GGML_OP_DUP:
    supported = true;
    break;

  // GET_ROWS reads a row from a source tensor by row index.
  // It is used primarily by KV-cache access when the model needs to
  // fetch a specific key/value row for attention scoring.
  case GGML_OP_GET_ROWS:
    switch (op->src[0]->type) {
    case GGML_TYPE_F32:
    case GGML_TYPE_F16:
      supported = true;
      break;
    default:
      supported = false;
      break;
    }
    break;

  // SET_ROWS writes one or more rows into a destination tensor using an
  // index tensor. This is used during KV-cache updates to copy the newly
  // computed key/value rows back into the cached tensor layout.
  case GGML_OP_SET_ROWS:
    if (op->src[0]->type == GGML_TYPE_F32) {
      switch (op->type) {
      case GGML_TYPE_F16:
      case GGML_TYPE_F32:
        supported = (op->src[1]->type == GGML_TYPE_I64 ||
                     op->src[1]->type == GGML_TYPE_I32);
        break;
      default:
        supported = false;
        break;
      }
    } else {
      supported = false;
    }
    break;

  // transformer ops
  case GGML_OP_MUL_MAT:
    supported =
        op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32;
    break;

  case GGML_OP_ADD:
  case GGML_OP_MUL:
  case GGML_OP_SUB:
  case GGML_OP_DIV:
    supported = op->type == GGML_TYPE_F32;
    break;

  case GGML_OP_RMS_NORM:
  case GGML_OP_NORM:
    supported = true;
    break;

  case GGML_OP_ROPE:
    supported = op->src[0]->type == GGML_TYPE_F32;
    break;

  case GGML_OP_UNARY:
    switch (ggml_get_unary_op(op)) {
    case GGML_UNARY_OP_SILU:
    case GGML_UNARY_OP_GELU:
      supported = op->src[0]->type == GGML_TYPE_F32;
      break;
    default:
      supported = false;
      break;
    }
    break;

  case GGML_OP_FLASH_ATTN_EXT:
    supported = op->src[0]->type == GGML_TYPE_F32 &&
                op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32;
    break;

  default:
    supported = false;
    break;
  }
  supported = false;
  ggml_redefine_log_op_info(op, supported);
  return supported;
}

bool ggml_backend_redefine_device_supports_op(ggml_backend_dev_t dev,
                                              const struct ggml_tensor *op) {
  ggml_redefine_init(dev);
  return ggml_redefine_supports_op(dev, op);
}

bool ggml_backend_redefine_device_supports_buft(
    ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
  if (dev->iface.get_name != ggml_backend_redefine_device_get_name ||
      buft->iface.get_name != ggml_backend_redefine_buffer_type_get_name) {
    return false;
  }
  return true;
}

ggml_guid_t ggml_backend_redefine_guid() {
  static ggml_guid guid = {0xde, 0xe0, 0x70, 0xa2, 0x73, 0x4e, 0x4d, 0xbc,
                           0xb0, 0xc7, 0x4f, 0xd4, 0x6d, 0x4e, 0x90, 0xfe};
  return &guid;
}

void ggml_redefine_print_backend_info(
    ggml_backend_redefine_device_context *dev_ctx) {
  GGML_ASSERT(dev_ctx);
  GGML_ASSERT(dev_ctx->backend_ctx);

  auto *backend_ctx = dev_ctx->backend_ctx;

  GGML_LOG_INFO("ggml_redefine: OpenCL driver: %s\n",
                backend_ctx->driver_version.c_str());
  GGML_LOG_INFO("ggml_redefine: mem base addr align: %u\n",
                backend_ctx->alignment);
  GGML_LOG_INFO("ggml_redefine: global mem size: %zu MB\n",
                backend_ctx->global_mem_size / 1024 / 1024);
  GGML_LOG_INFO("ggml_redefine: max mem alloc size: %zu MB\n",
                backend_ctx->max_alloc_size / 1024 / 1024);
  GGML_LOG_INFO("ggml_redefine: max workgroup size: %zu\n",
                backend_ctx->max_workgroup_size);
}

ggml_backend_t ggml_backend_redefine_device_init(ggml_backend_dev_t dev,
                                                 const char *params) {
  ggml_backend_redefine_context *backend_ctx = ggml_redefine_init(dev);
  // Getting a new reference to the backend, increase ref_count
  backend_ctx->ref_count++;

  ggml_backend_t backend = new ggml_backend{
      /* .guid      = */ ggml_backend_redefine_guid(),
      /* .interface = */ ggml_backend_redefine_i,
      /* .device    = */ dev,
      /* .context   = */ backend_ctx,
  };

  ggml_backend_redefine_device_context *dev_ctx =
      (ggml_backend_redefine_device_context *)dev->context;
  ggml_redefine_print_backend_info(dev_ctx);
  return backend;

  GGML_UNUSED(params);
}

ggml_backend_buffer_type_t
ggml_backend_redefine_get_host_buffer_type(ggml_backend_dev_t dev) {
  GGML_UNUSED(dev);
  GGML_LOG_WARN("%s: not implemented yet\n", __func__);
  return NULL;
}

bool ggml_backend_redefine_offload_op(ggml_backend_dev_t dev,
                                      const struct ggml_tensor *op) {
  GGML_UNUSED(dev);
  GGML_UNUSED(op);
  GGML_LOG_WARN("%s: not implemented yet\n", __func__);
  return false;
}

ggml_backend_event_t ggml_backend_redefine_event_new(ggml_backend_dev_t dev) {
  GGML_UNUSED(dev);
  GGML_LOG_WARN("%s: not implemented yet\n", __func__);
  return NULL;
}

void ggml_backend_redefine_event_free(ggml_backend_dev_t dev,
                                      ggml_backend_event_t event) {
  GGML_UNUSED(dev);
  GGML_UNUSED(event);
  GGML_LOG_WARN("%s: not implemented yet\n", __func__);
}

void ggml_backend_redefine_event_synchronize(ggml_backend_dev_t dev,
                                             ggml_backend_event_t event) {
  GGML_UNUSED(dev);
  GGML_UNUSED(event);
  GGML_LOG_WARN("%s: not implemented yet\n", __func__);
}

ggml_backend_buffer_t ggml_backend_redefine_buffer_from_host_ptr(
    ggml_backend_dev_t dev, void *ptr, size_t size, size_t max_tensor_size) {
  GGML_UNUSED(dev);
  GGML_UNUSED(ptr);
  GGML_UNUSED(size);
  GGML_UNUSED(max_tensor_size);
  GGML_LOG_WARN("%s: not implemented yet\n", __func__);
  return NULL;
}

ggml_backend_buffer_type_t
ggml_backend_redefine_device_get_buffer_type(ggml_backend_dev_t dev) {
  auto *dev_ctx =
      static_cast<ggml_backend_redefine_device_context *>(dev->context);

  if (dev_ctx->buffer_type.iface.get_name == nullptr) {
    dev_ctx->buffer_type = ggml_backend_buffer_type{
        /* .iface   = */ ggml_backend_redefine_buffer_type_i,
        /* .device  = */ dev,
        /* .context = */ nullptr,
    };
  }

  return &dev_ctx->buffer_type;
}

struct ggml_backend_device_i ggml_backend_redefine_device_i = {
    /* .get_name             = */ ggml_backend_redefine_device_get_name,
    /* .get_description      = */ ggml_backend_redefine_device_get_description,
    /* .get_memory           = */ ggml_backend_redefine_device_get_memory,
    /* .get_type             = */ ggml_backend_redefine_device_get_type,
    /* .get_props            = */ ggml_backend_redefine_device_get_props,
    /* .init_backend         = */ ggml_backend_redefine_device_init,
    /* .get_buffer_type      = */ ggml_backend_redefine_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_redefine_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_redefine_device_supports_op,
    /* .supports_buft        = */ ggml_backend_redefine_device_supports_buft,
    /* .offload_op           = */ ggml_backend_redefine_offload_op,
    /* .event_new            = */ ggml_backend_redefine_event_new,
    /* .event_free           = */ ggml_backend_redefine_event_free,
    /* .event_synchronize    = */ ggml_backend_redefine_event_synchronize,
};
} // namespace

// Look for available and suitable devices (isasim, fsim, or redefine from
// pocl_platform).
static std::vector<ggml_backend_device>
ggml_redefine_probe_devices(ggml_backend_reg *reg) {
  std::vector<ggml_backend_device> found_devices;

  cl_uint n_platforms;
  CL_CHECK(clGetPlatformIDs(0, NULL, &n_platforms));
  if (n_platforms == 0) {
    GGML_LOG_INFO("ggml_redefine: no OpenCL platforms available\n");
    return found_devices;
  }

  std::vector<cl_platform_id> platform_ids(n_platforms);
  CL_CHECK(clGetPlatformIDs(n_platforms, platform_ids.data(), NULL));

  cl_context shared_context = nullptr;
  cl_platform_id selected_platform = nullptr;
  std::vector<cl_device_id> matching_devices;

  for (cl_platform_id platform : platform_ids) {
    std::string platform_name =
        ggml_cl_get_platform_info_string(platform, CL_PLATFORM_NAME);

    // Only look for pocl platform (Portable Computing Language)
    if (!ggml_redefine_platform_supported(platform_name)) {
      continue;
    }

    cl_uint n_devices;
    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &n_devices) !=
        CL_SUCCESS) {
      continue;
    }
    if (n_devices == 0) {
      continue;
    }

    std::vector<cl_device_id> device_ids(n_devices);
    CL_CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, n_devices,
                            device_ids.data(), NULL));

    for (cl_device_id device : device_ids) {
      std::string device_name =
          ggml_cl_get_device_info_string(device, CL_DEVICE_NAME);

      if (ggml_redefine_device_supported(device_name)) {
        matching_devices.push_back(device);
        if (!selected_platform) {
          selected_platform = platform;
        }
      }
    }

    if (!matching_devices.empty() && !selected_platform) {
      selected_platform = platform;
      break;
    }
  }

  if (matching_devices.empty()) {
    GGML_LOG_INFO("ggml_redefine: no redefine devices found.\n");
    return found_devices;
  }

  // Create shared context for all matching devices
  cl_int err;
  cl_context_properties properties[] = {(intptr_t)CL_CONTEXT_PLATFORM,
                                        (intptr_t)selected_platform, 0};
  CL_CHECK((shared_context =
                clCreateContext(properties, matching_devices.size(),
                                matching_devices.data(), NULL, NULL, &err),
            err));

  // Initialize each matching device
  for (cl_device_id device : matching_devices) {
    std::string device_name =
        ggml_cl_get_device_info_string(device, CL_DEVICE_NAME);
    std::string device_version =
        ggml_cl_get_device_info_string(device, CL_DEVICE_VERSION);

    cl_device_type device_type;
    CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(device_type),
                             &device_type, NULL));

    cl_ulong global_mem_size = 0;
    CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE,
                             sizeof(global_mem_size), &global_mem_size, NULL));

    GGML_LOG_INFO("ggml_redefine: device: '%s (%s)'\n", device_name.c_str(),
                  device_version.c_str());

    auto dev_ctx = std::unique_ptr<ggml_backend_redefine_device_context>(
        new ggml_backend_redefine_device_context{
            /*.platform         =*/selected_platform,
            /*.platform_name    =*/"pocl_platform",
            /*.device           =*/device,
            /*.device_name      =*/device_name,
            /*.device_type      =*/device_type,
            /*.device_version   =*/device_version,
            /*.backend_ctx      =*/nullptr,
            /*.buffer_type      =*/{},
            /*.context          =*/shared_context,
            /*.opfilter         =*/nullptr,
            /*.opfilter_str     =*/"",
            /*.global_mem_size  =*/(size_t)global_mem_size,
        });

    found_devices.push_back(ggml_backend_device{
        /* .iface   = */ ggml_backend_redefine_device_i,
        /* .reg     = */ reg,
        /* .context = */ dev_ctx.get(),
    });

    g_ggml_backend_redefine_dev_ctxs.push_back(std::move(dev_ctx));
  }

  if (found_devices.size()) {
    auto *dev_ctx = static_cast<ggml_backend_redefine_device_context *>(
        found_devices.front().context);
    GGML_LOG_INFO("ggml_redefine: default device: '%s (%s)'\n",
                  dev_ctx->device_name.c_str(),
                  dev_ctx->device_version.c_str());
  }

  return found_devices;
}

ggml_backend_reg_t ggml_backend_redefine_reg(void) {
  static std::mutex mutex;
  static ggml_backend_reg reg;
  static bool initialized = false;
  std::lock_guard<std::mutex> lock(mutex);

  if (initialized) {
    return &reg;
  }
  initialized = true;

  g_ggml_backend_redefine_devices = ggml_redefine_probe_devices(&reg);

  reg = ggml_backend_reg{
      /* .api_version = */ GGML_BACKEND_API_VERSION,
      /* .iface       = */ ggml_backend_redefine_reg_i,
      /* .context     = */ NULL,
  };

  return &reg;
}

static int ggml_backend_redefine_score(void) {
  // Return positive score if redefine backend is available
  return 1;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_redefine_score)
GGML_BACKEND_DL_IMPL(ggml_backend_redefine_reg)

struct ggml_cl_compiler_version {
  int major = -1;
  int minor = -1;
  int patch = -1;
};

static size_t align_to(size_t value, size_t to_alignment) {
  GGML_ASSERT(to_alignment && "Invalid alignment (must be non-zero)");
  GGML_ASSERT((to_alignment & (to_alignment - 1)) == 0 &&
              "to_alignment must be power-of-two");

  return ((value + to_alignment - 1) / to_alignment) * to_alignment;
}

// Parses a version string of form "XX.YY ". On an error returns ggml_cl_version
// with all zeroes.
static ggml_cl_version parse_cl_version(std::string_view str) {
  size_t major_str_begin = 0;
  size_t major_str_end = str.find('.', major_str_begin);
  if (major_str_end == std::string::npos) {
    return {};
  }

  size_t minor_str_begin = major_str_end + 1;
  size_t minor_str_end = str.find(' ', minor_str_begin);
  if (minor_str_end == std::string::npos) {
    return {};
  }

  cl_uint version_major;
  if (std::from_chars(str.data() + major_str_begin, str.data() + major_str_end,
                      version_major)
          .ec != std::errc{}) {
    return {};
  }

  cl_uint version_minor;
  if (std::from_chars(str.data() + minor_str_begin, str.data() + minor_str_end,
                      version_minor)
          .ec != std::errc{}) {
    return {};
  }
  return {version_major, version_minor};
}

// Returns OpenCL platform's version. On an error returns ggml_cl_version with
// all zeroes.
static ggml_cl_version get_opencl_platform_version(cl_platform_id platform) {
  size_t param_size;
  CL_CHECK(clGetPlatformInfo(platform, CL_PLATFORM_VERSION, 0, nullptr,
                             &param_size));
  std::unique_ptr<char[]> param_storage(new char[param_size]);
  CL_CHECK(clGetPlatformInfo(platform, CL_PLATFORM_VERSION, param_size,
                             param_storage.get(), nullptr));

  auto param_value = std::string_view(param_storage.get(), param_size);
  const std::string version_prefix =
      "OpenCL "; // Suffix: "XX.YY <platform-specific-info>"
  if (param_value.find(version_prefix) != 0) {
    return {};
  }
  param_value.remove_prefix(version_prefix.length());
  return parse_cl_version(param_value);
}

// Return a version to use in OpenCL C compilation. On an error returns
// ggml_cl_version with all zeroes.
static ggml_cl_version get_opencl_c_version(ggml_cl_version platform_version,
                                            cl_device_id device) {
  size_t param_size;

#if CL_TARGET_OPENCL_VERSION >= 300
  if (platform_version.major >= 3) {
    CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_ALL_VERSIONS, 0,
                             nullptr, &param_size));
    if (!param_size) {
      return {};
    }

    std::unique_ptr<cl_name_version[]> versions(
        new cl_name_version[param_size]);
    CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_ALL_VERSIONS,
                             param_size, versions.get(), nullptr));
    unsigned versions_count = param_size / sizeof(cl_name_version);

    cl_version version_max = 0;
    for (unsigned i = 0; i < versions_count; i++) {
      version_max = std::max<cl_version>(versions[i].version, version_max);
    }

    return {CL_VERSION_MAJOR(version_max), CL_VERSION_MINOR(version_max)};
  }
#else
  GGML_UNUSED(platform_version);
#endif // CL_TARGET_OPENCL_VERSION >= 300

  CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, 0, nullptr,
                           &param_size));
  if (!param_size) {
    return {};
  }

  std::unique_ptr<char[]> param_storage(new char[param_size]);
  CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, param_size,
                           param_storage.get(), nullptr));
  auto param_value = std::string_view(param_storage.get(), param_size);

  const std::string version_prefix =
      "OpenCL C "; // Suffix: "XX.YY <platform-specific-info>"
  if (param_value.find(version_prefix) != 0) {
    return {};
  }
  param_value.remove_prefix(version_prefix.length());

  return parse_cl_version(param_value);
}

// Initialize device if it is supported (returns nullptr if it is not).
static ggml_backend_redefine_context *
ggml_redefine_init(ggml_backend_dev_t dev) {
  GGML_ASSERT(dev);
  GGML_ASSERT(dev->context);

  ggml_backend_redefine_device_context *dev_ctx =
      (ggml_backend_redefine_device_context *)dev->context;
  GGML_ASSERT(dev_ctx->platform);
  GGML_ASSERT(dev_ctx->device);

  if (dev_ctx->backend_ctx) {
    return dev_ctx->backend_ctx;
  }

  auto backend_ctx = std::make_unique<ggml_backend_redefine_context>();
  backend_ctx->device = dev_ctx->device;

  // ref_count get increased in ggml_backend_redefine_device_init
  // This function is also used to retrieve backend context, so we don't want
  // to increase ref_count for each call. We only want to increase ref_count
  // when the associated device is initialized
  backend_ctx->ref_count = 0;

  // Populate backend device name
  backend_ctx->device_name = dev_ctx->device_name;

  // A local ref of cl_device_id for convenience
  cl_device_id device = backend_ctx->device;

  ggml_cl_version platform_version =
      get_opencl_platform_version(dev_ctx->platform);
  ggml_cl_version opencl_c_version =
      get_opencl_c_version(platform_version, device);

  backend_ctx->platform_version = platform_version;
  backend_ctx->opencl_c_version = opencl_c_version;

  // Check driver version
  size_t driver_version_str_size;
  clGetDeviceInfo(device, CL_DRIVER_VERSION, 0, NULL, &driver_version_str_size);
  char *driver_version = (char *)alloca(driver_version_str_size + 1);
  clGetDeviceInfo(device, CL_DRIVER_VERSION, driver_version_str_size,
                  driver_version, NULL);
  driver_version[driver_version_str_size] = '\0';
  backend_ctx->driver_version = driver_version;

  size_t ext_str_size;
  clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, NULL, &ext_str_size);
  char *ext_buffer = (char *)alloca(ext_str_size + 1);
  clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, ext_str_size, ext_buffer, NULL);
  ext_buffer[ext_str_size] = '\0'; // ensure it is null terminated

  cl_uint base_align_in_bits;
  CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_MEM_BASE_ADDR_ALIGN,
                           sizeof(cl_uint), &base_align_in_bits, NULL));
  GGML_ASSERT(base_align_in_bits % 8u == 0);
  backend_ctx->alignment = base_align_in_bits / 8u;

  backend_ctx->global_mem_size = dev_ctx->global_mem_size;

  CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(size_t),
                           &backend_ctx->max_alloc_size, NULL));

  cl_int err;

  // A local ref of cl_context for convenience
  cl_context context = backend_ctx->context = dev_ctx->context;
  cl_command_queue_properties command_queue_props = 0;
  CL_CHECK((backend_ctx->queue = clCreateCommandQueue(
                context, device, command_queue_props, &err),
            err));

  dev_ctx->backend_ctx = backend_ctx.release();
  return dev_ctx->backend_ctx;
}
