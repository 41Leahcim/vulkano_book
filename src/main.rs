use std::sync::Arc;

use vulkano::{
    instance::{Instance, InstanceCreateInfo},
    VulkanLibrary, device::physical::PhysicalDevice,
};

fn list_gpus(instance: Arc<Instance>){
    // List all available gpu's
    for device in instance
        .enumerate_physical_devices()
        .expect("Could not enumerate devices")
    {
        println!("{}: {:?}, {}", device.properties().device_name, device.properties().driver_name, device.api_version());
    }
}

fn list_queues(physical_device: Arc<PhysicalDevice>){
    for family in physical_device.queue_family_properties(){
        println!("Found a queue family with {:?} queue(s)", family.queue_count);
    }
}

fn main() {
    // Get a vulkan instance
    let library = VulkanLibrary::new().expect("No local Vulkan library/DLL");
    let instance =
        Instance::new(library, InstanceCreateInfo::default()).expect("failed to create instance");

    list_gpus(instance.clone());

    // Select the first gpu (we would want to allow the user to select one, in a real application)
    let physical_device = instance
        .enumerate_physical_devices()
        .expect("Could not enumerate devices")
        .next()
        .expect("no devices available");

    list_queues(physical_device.clone());
}
