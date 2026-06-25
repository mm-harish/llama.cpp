# GGML REDEFINE Backend

The REDEFINE backend enables llama.cpp to use REDEFINE accelerators (Functional Simulator, ISA Simulator) via the PoCL OpenCL platform.

## Prerequisites

- Install REDEFINE SDK

## Building with REDEFINE Backend

### Option 1: Dynamic Backend Loading (Recommended)

Enable the REDEFINE backend as a dynamic module:

```bash
mkdir build-redefine
cd build-redefine
cmake /path/to/llama.cpp \
  -DGGML_REDEFINE=ON \
  -DGGML_BACKEND_DL=ON \
  -DGGML_OPENCL=OFF \
  -DGGML_NATIVE=OFF \
  -DCMAKE_BUILD_TYPE=Release

cmake --build . -j$(nproc)
```

**Flags:**
- `-DGGML_REDEFINE=ON`: Enable the REDEFINE backend
- `-DGGML_BACKEND_DL=ON`: Enable dynamic backend loading (creates standalone .so modules)
- `-DGGML_OPENCL=OFF`: Disable the main OpenCL backend to avoid conflicts
- `-DGGML_NATIVE=OFF`: Required when using `GGML_BACKEND_DL`

### Build Output

The build produces:
- `build-redefine/bin/libggml-redefine.so`: The REDEFINE backend module
- `build-redefine/bin/libggml.so`: Main GGML library
- `build-redefine/bin/llama-cli`: CLI tool with backend support

## Testing the Backend

### 1. Verify Device Detection

List available devices:

```bash
cd build-redefine
./bin/llama-cli --list-devices
```

**Expected output:**
```
Available devices:
  Functional Simulator: OpenCL 1.2 PoCL HSTR: fsim-x86_64-unknown-linux-gnu-alderlake (...)
  ISA Simulator: OpenCL 1.2 PoCL HSTR: isasim-riscv32-linux-elf-redefine-rv32 (...)
```

If no devices appear, check:
- PoCL installation: `clinfo` should show PoCL platforms
- Library paths: Ensure `libggml-redefine.so` is in the same directory as `libggml.so`
- OpenCL ICD: Verify OpenCL can find PoCL platforms

### 2. Using the Backend for Inference

Specify the device for a model:

```bash
./bin/llama-cli -m model.gguf --device "Functional Simulator" -p "Hello, world!"
```

Or select a device by index:

```bash
./bin/llama-cli -m model.gguf -dev 0 -p "Hello, world!"
```

### 3. Debug Logging

Enable detailed backend logging:

```bash
GGML_LOG_LEVEL=debug ./bin/llama-cli --list-devices
```

This will show:
- Platform detection
- Device enumeration
- Context creation
- Backend initialization

## Environment Variables

- `GGML_LOG_LEVEL`: Set to `debug` or `warn` for verbose output
- `GGML_BACKEND_PATH`: Explicitly specify path to `libggml-redefine.so` if not auto-discovered

## Architecture

The REDEFINE backend:
- **Uses OpenCL** for platform and device discovery (leverages PoCL)
- **Creates device contexts** for each REDEFINE accelerator
- **Implements the GGML backend interface** for graph execution
- **Dynamically loads** as a standalone module when `GGML_BACKEND_DL=ON`

### Key Files

- `ggml-redefine.cpp`: Main backend implementation
- `ggml-redefine.h`: Public API header
- `CMakeLists.txt`: Build configuration

## Current Limitations

- **Graph computation**: Currently returns success without executing (stub implementation)
- **Device memory**: Reported as 0 MiB (query not fully implemented)
- **Buffer management**: Minimal implementation
- **Operation support**: Device capability queries not fully implemented

These are placeholders for future implementation.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No devices detected | Run `clinfo` to verify PoCL/OpenCL setup; check library paths |
| `libggml-redefine.so` not found | Ensure `-DGGML_BACKEND_DL=ON` was used during build |
| Build fails with OpenCL headers | Install: `apt-get install opencl-headers` |
| Segmentation fault on device enumeration | Check that `libggml-redefine.so` and `libggml.so` are compatible versions |

## References

- [llama.cpp Backend System](https://github.com/ggml-org/llama.cpp/blob/master/docs/development/)
- [PoCL Documentation](https://pocl.readthedocs.io/)
- [OpenCL Specification](https://www.khronos.org/opencl/)
