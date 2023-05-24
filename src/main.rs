use std::sync::Arc;

use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage},
    device::{
        physical::PhysicalDevice, Device, DeviceCreateInfo, Queue, QueueCreateInfo, QueueFlags,
    },
    instance::{Instance, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryUsage, StandardMemoryAllocator},
    VulkanLibrary,
};

fn list_gpus(instance: Arc<Instance>) {
    // List all available gpu's
    for device in instance
        .enumerate_physical_devices()
        .expect("Could not enumerate devices")
    {
        println!(
            "{}: {:?}, {}",
            device.properties().device_name,
            device.properties().driver_name,
            device.api_version()
        );
    }
}

fn list_queues(physical_device: Arc<PhysicalDevice>) {
    for family in physical_device.queue_family_properties() {
        println!(
            "Found a queue family with {:?} queue(s)",
            family.queue_count
        );
    }
}

fn initialization() -> (Arc<Device>, Arc<Queue>) {
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
        .position(|(_queue_family_index, queue_family_properties)| {
            queue_family_properties
                .queue_flags
                .contains(QueueFlags::GRAPHICS)
        })
        .expect("Couldn't find a graphical queue family") as u32;

    // Create a new Vulkan device, returning the device and an iterator over queues on that device
    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            // Here we pass the desired queue family to use by index
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
    .expect("Failed to create device");

    // Select the first queue
    let queue = queues.next().unwrap();

    (device, queue)
}

#[derive(Debug, BufferContents)]
#[repr(C)]
struct MyStruct {
    a: u32,
    b: u32,
}

fn main() {
    // Initialize Vulkan and store a reference to a device and it's first graphical queue
    let (device, queue) = initialization();

    // Create a general purpose memory allocator
    let memory_allocator = StandardMemoryAllocator::new_default(device.clone());

    // Create the source data and buffer
    let source_content = 0..64;
    let source = Buffer::from_iter(
        &memory_allocator,
        BufferCreateInfo{
            usage: BufferUsage::TRANSFER_SRC, // This data will be used as source for a data transfer
            ..Default::default()
        },
        AllocationCreateInfo{
            usage: MemoryUsage::Upload, // This will only upload data to the gpu
            ..Default::default()
        },
        source_content
    ).expect("Failed to create a source buffer.");

    // Create the destination data and buffer
    let destination_content = (0..64).map(|_| 0);
    let destination = Buffer::from_iter(
        &memory_allocator,
        BufferCreateInfo{
            usage: BufferUsage::TRANSFER_DST, // This will be the destination of the data transfer
            ..Default::default()
        },
        AllocationCreateInfo{
            usage: MemoryUsage::Download, // This will only download data from the gpu
            ..Default::default()
        },
        destination_content
    ).expect("Failed to create a destination buffer");
}
