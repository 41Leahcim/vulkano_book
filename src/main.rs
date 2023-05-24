use std::sync::Arc;

use vulkano::{
    device::{
        physical::PhysicalDevice, Device, DeviceCreateInfo, Queue, QueueCreateInfo, QueueFlags,
    },
    instance::{Instance, InstanceCreateInfo},
    memory::allocator::{StandardMemoryAllocator, GenericMemoryAllocator, FreeListAllocator},
    VulkanLibrary, image::{StorageImage, ImageDimensions}, format::Format,
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

fn get_queue_family_index(physical_device: Arc<PhysicalDevice>, selection: QueueFlags) -> u32{
    physical_device
        .queue_family_properties()
        .iter()
        .enumerate()
        .position(|(_queue_family_index, queue_family_properties)| {
            queue_family_properties
                .queue_flags
                .contains(selection)
        })
        .expect("Couldn't find a graphical queue family") as u32
}

fn initialization() -> (Arc<Device>, Arc<Queue>, u32) {
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
    let queue_family_index = get_queue_family_index(physical_device.clone(), QueueFlags::GRAPHICS);

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

    (device, queue, queue_family_index)
}

fn create_data(memory_allocator: &GenericMemoryAllocator<Arc<FreeListAllocator>>, queue: Arc<Queue>) -> Arc<StorageImage>{
    StorageImage::new(
        memory_allocator,
        ImageDimensions::Dim2d {
            width: 1024,
            height: 1024,
            array_layers: 1 // Images can be arrats of layers
        },
        Format::R8G8B8A8_UNORM,
        Some(queue.queue_family_index())
    ).unwrap()
}

fn main() {
    // Initialize Vulkan and store a reference to a device, a graphical queue family index, and the first queue of that queue family
    let (device, queue, queue_family_index) = initialization();

    // Create a general purpose memory allocator
    let memory_allocator = StandardMemoryAllocator::new_default(device.clone());

    let data_buffer = create_data(&memory_allocator, queue);
}
