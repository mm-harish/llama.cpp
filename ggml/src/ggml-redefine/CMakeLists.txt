find_package(OpenCL REQUIRED)
find_package(Python3 REQUIRED)

set(TARGET_NAME ggml-redefine)

ggml_add_backend_library(${TARGET_NAME}
                         ggml-redefine.cpp
                         ../../include/ggml-redefine.h)
target_link_libraries(${TARGET_NAME} PRIVATE ${OpenCL_LIBRARIES})
target_include_directories(${TARGET_NAME} PRIVATE ${OpenCL_INCLUDE_DIRS})

add_compile_definitions(GGML_OPENCL_TARGET_VERSION=${GGML_OPENCL_TARGET_VERSION})


function(ggml_redefine_add_kernel KNAME)
    set(KERN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/kernels/${KNAME}.c)
    message(STATUS "redefine: adding kernel ${KNAME}")
    configure_file(${KERN_SRC} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${KNAME}.c COPYONLY)
endfunction()

set(GGML_REDEFINE_KERNELS
)

foreach (K ${GGML_REDEFINE_KERNELS})
    ggml_redefine_add_kernel(${K})
endforeach()
