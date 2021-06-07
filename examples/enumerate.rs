use cuda_oxide::*;

fn main() {
    Cuda::init().unwrap();
    let v = Cuda::version().unwrap();
    println!("{:?}", v);
    for device in Cuda::list_devices().unwrap() {
        println!("name: {}", device.name().unwrap());
        println!("uuid: {}", device.uuid().unwrap());
        println!("memory size: {}", device.memory_size().unwrap());
        println!(
            "clock rate: {}",
            device
                .get_attribute(DeviceAttribute::MemoryClockRate)
                .unwrap()
        );
    }
}
