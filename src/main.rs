use vulkano::{
    instance::{Instance, InstanceCreateInfo},
    VulkanLibrary,
};

fn main() {
    let library = VulkanLibrary::new().expect("No local Vulkan library/DLL");
    let instance =
        Instance::new(library, InstanceCreateInfo::default()).expect("failed to create instance");

    for device in instance
        .enumerate_physical_devices()
        .expect("Could not enumerate devices")
    {
        println!("{}", device.properties().device_name);
    }

    let physical_devices = instance
        .enumerate_physical_devices()
        .expect("Could not enumerate devices")
        .next()
        .expect("no devices available");
}
