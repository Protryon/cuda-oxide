
# cuda-oxide

`cuda-oxide` is a *safe* wrapper for [CUDA](https://en.wikipedia.org/wiki/CUDA). With `cuda-oxide` you can execute and coordinate CUDA kernels.

## Safety Philosophy

`cuda-oxide` *does not* offer any safety on the GPU-side of writing CUDA code. It doesn't compile Rust to PTX. `cuda-oxide` offers general CPU-level safety working with the CUDA library and best-availability safety for working with GPU buffers and objects.

Examples of things currently considered safe:
* Reading from an uninitialized GPU buffer
* Reading from a GPU buffer currently being streamed and in an indeterminate state
* Passing garbage/invalid data to CUDA kernel parameters
* Some invalid `libcuda` operations that will cause `libcuda` to stop accepting any API calls
* Setting various attributes that can have side effects for an entire device
* Writing to read-only device memory

### Notable Exception

`cuda-oxide` doesn't parse compiled binaries to find a valid CUDA signature. It is possible to to load a kernel that will some number of parameters, pass in fewer from Rust, and get a segfault. We would need to write a Fatbin/etc parser to remedy this.

## Examples

See the `examples` directory for usage examples.