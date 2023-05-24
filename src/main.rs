use std::sync::Arc;

use vulkano::{
    instance::{Instance, InstanceCreateInfo},
    VulkanLibrary, device::{physical::PhysicalDevice, QueueFlags, Device, DeviceCreateInfo, QueueCreateInfo}, memory::allocator::StandardMemoryAllocator,
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
        Instance::new(library, InstanceCreateInfo::default()).expect("Failed to create instance");

    list_gpus(instance.clone());

    // Select the first gpu (we would want to allow the user to select one, in a real application)
    let physical_device = instance
        .enumerate_physical_devices()
        .expect("Could not enumerate devices")
        .next()
        .expect("No devices available");

    // List the queues available on the selected device
    list_queues(physical_device.clone());

    // Select a queue that supports graphical operations
    let queue_family_index = physical_device
        .queue_family_properties()
        .iter()
        .enumerate()
        .position(|(_queue_family_index, queue_family_properties)|{
            queue_family_properties.queue_flags.contains(QueueFlags::GRAPHICS)
        })
        .expect("Couldn't find a graphical queue family") as u32;
    
    // Create a new Vulkan device, returning the device and an iterator over queues on that device
    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo{
            // Here we pass the desired queue family to use by index
            queue_create_infos: vec![QueueCreateInfo{
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        }
    ).expect("Failed to create device");

    // Select the first queue
    let queue = queues.next().unwrap();

    // Create a general purpose memory allocator
    let memory_allocator = StandardMemoryAllocator::new_default(device.clone());
}
