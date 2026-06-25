#ifndef GGML_REDEFINE_H
#define GGML_REDEFINE_H

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

//
// backend API
//
GGML_BACKEND_API ggml_backend_t ggml_backend_redefine_init(void);
GGML_BACKEND_API bool ggml_backend_is_redefine(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_redefine_buffer_type(void);
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_redefine_host_buffer_type(void);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_redefine_reg(void);

#ifdef  __cplusplus
}
#endif

#endif // GGML_REDEFINE_H
