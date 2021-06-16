
# cuda-oxide

`cuda-oxide` is a *safe* wrapper for [CUDA](https://en.wikipedia.org/wiki/CUDA). With `cuda-oxide` you can execute and coordinate CUDA kernels.

## Safety Philosophy

`cuda-oxide` *does not* offer any safety on the GPU-side of writing CUDA code. It doesn't compile Rust to PTX. `cuda-oxide` offers general CPU-level safety working with the CUDA library and best-availability safety for working with GPU buffers and objects.

Examples of things currently considered safe:
* Reading from an uninitialized GPU buffer into host memory
* Some invalid `libcuda` operations that will cause `libcuda` to stop accepting any API calls
* Setting various attributes that can have side effects for an entire device
* Writing to read-only device memory

## Supported Features
* Device Management
* Context Management
* Module Management
* JIT compilation of Modules
* Stream Management
* Kernel Execution
* Device Memory read/write

## Unsupported Features
* Memory Pools
* Unified Addressing
* Events & Stream Events
* Stream State Polling
* Stream Graph Capturing
* Stream Batch Memory Operations
* External Memory
* Multi-device helper (possible already, but not made easy)
* Graphs
* Textures & Surfaces
* OpenGL/VDPAU/EGL Interoperability

## Examples

See the `examples` directory for usage examples.