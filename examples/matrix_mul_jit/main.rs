use cuda_oxide::*;
use rand::{thread_rng, Rng};

const BLOCK_SIZE: u32 = 32;
const A_WIDTH: usize = BLOCK_SIZE as usize * 40;
const A_HEIGHT: usize = BLOCK_SIZE as usize * 60;
const B_WIDTH: usize = BLOCK_SIZE as usize * 40;
const B_HEIGHT: usize = BLOCK_SIZE as usize * 40;
const C_WIDTH: usize = B_WIDTH;
const C_HEIGHT: usize = A_HEIGHT;

fn matrix_bytes(input: &[f64]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 8) }
}

fn bytes_matrix(input: &[u8]) -> &[f64] {
    unsafe { std::slice::from_raw_parts(input.as_ptr() as *const f64, input.len() / 8) }
}

fn main() {
    Cuda::init().unwrap();
    let v = Cuda::version().unwrap();
    println!("Using CUDA {}", v);
    let device = Cuda::list_devices().unwrap();
    let device = device.first().unwrap();
    println!("using device: {}", device.name().unwrap());
    let device_compute = device.compute_capability().unwrap();
    println!("cuda device compute capability = {}", device_compute);

    let mut context = Context::new(device).unwrap();

    let handle = context.enter().unwrap();

    // normally this would be built by a build script, but examples in cargo don't seem to support this
    // nvcc matrixMul_kernel.cu -ptx
    let kernel = include_bytes!("./matrixMul_kernel.ptx");
    let linked_kernel = Linker::new(&handle, device_compute, LinkerOptions::default())
        .unwrap()
        .add("matrixMul_kernel.ptx", LinkerInputType::Ptx, &kernel[..])
        .unwrap();
    let module = linked_kernel.build_module().unwrap();

    let function = module.get_function("matrixMul_bs32_64bit").unwrap();

    let mut mat_a = vec![0.0; A_WIDTH * A_HEIGHT];
    let mut mat_b = vec![0.0; B_WIDTH * B_HEIGHT];

    for i in 0..mat_a.len() {
        mat_a[i] = thread_rng().gen_range(0.0..1.0);
    }
    for i in 0..mat_b.len() {
        mat_b[i] = thread_rng().gen_range(0.0..1.0);
    }

    let device_mat_a = DeviceBox::new(&handle, matrix_bytes(&mat_a[..])).unwrap();
    let device_mat_b = DeviceBox::new(&handle, matrix_bytes(&mat_b[..])).unwrap();

    let output = DeviceBox::alloc(&handle, C_WIDTH as u64 * C_HEIGHT as u64 * 8).unwrap();

    handle.context().synchronize().unwrap();

    let rea = device_mat_a.load().unwrap();
    assert_eq!(&rea[..], matrix_bytes(&mat_a[..]));

    let mut stream = Stream::new(&handle).unwrap();
    stream
        .launch(
            &function,
            (C_WIDTH as u32 / BLOCK_SIZE, C_HEIGHT as u32 / BLOCK_SIZE),
            (BLOCK_SIZE, BLOCK_SIZE),
            2 * BLOCK_SIZE * BLOCK_SIZE * 8,
            (
                &output,
                &device_mat_a,
                &device_mat_b,
                A_WIDTH as usize,
                B_WIDTH as usize,
            ),
        )
        .unwrap();

    stream.callback(|| println!("done")).unwrap();

    stream.sync().unwrap();

    let output = output.load().unwrap();
    let output = bytes_matrix(&output[..]);
    println!("{:?}", output);
}
