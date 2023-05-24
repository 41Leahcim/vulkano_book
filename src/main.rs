use std::sync::Arc;

use vulkano::{
    device::{
        physical::PhysicalDevice, Device, DeviceCreateInfo, Queue, QueueCreateInfo, QueueFlags,
    },
    instance::{Instance, InstanceCreateInfo},
    memory::allocator::{StandardMemoryAllocator, GenericMemoryAllocator, FreeListAllocator, AllocationCreateInfo, MemoryUsage},
    VulkanLibrary, image::{StorageImage, ImageDimensions}, format::{Format, ClearColorValue}, command_buffer::{AutoCommandBufferBuilder, allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo}, CommandBufferUsage, ClearColorImageInfo, CopyImageToBufferInfo}, buffer::{Subbuffer, Buffer, BufferCreateInfo, BufferUsage}, sync::{self, GpuFuture},
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

    (device, queue)
}

fn create_image(memory_allocator: &GenericMemoryAllocator<Arc<FreeListAllocator>>, queue: Arc<Queue>) -> Arc<StorageImage>{
    StorageImage::new(
        memory_allocator,
        ImageDimensions::Dim2d {
            width: 1024,
            height: 1024,
            array_layers: 1 // Images can be arrats of layers
        },
        Format::R8G8B8A8_UNORM, // unsigned normalized 8-bit RGBA values
        Some(queue.queue_family_index())
    ).unwrap()
}

fn create_buffer(memory_allocator: &GenericMemoryAllocator<Arc<FreeListAllocator>>) -> Subbuffer<[u8]>{
    // Create a buffer
    Buffer::from_iter(
        memory_allocator,
        BufferCreateInfo{
            usage: BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo{
            usage: MemoryUsage::Download,
            ..Default::default()
        },
        (0..1024 * 1024 * 4).map(|_| 0u8)
    ).expect("Failed to create buffer")
}

fn main() {
    // Initialize Vulkan and store a reference to a device, a graphical queue family index, and the first queue of that queue family
    let (device, queue) = initialization();

    // Create a general purpose memory allocator
    let memory_allocator = StandardMemoryAllocator::new_default(device.clone());

    let image = create_image(&memory_allocator, queue.clone());
    let buf = create_buffer(&memory_allocator);

    // Create a command buffer allocator
    let command_buffer_allocator = StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default()
    );

    // Create a command buffer builder
    let mut builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit
    ).unwrap();

    builder
        .clear_color_image(
            ClearColorImageInfo{
                clear_value: ClearColorValue::Float([0.0, 0.0, 1.0, 1.0]), // normalized values (UNORM< SNORM< SRGB) are interpreted as floats
                ..ClearColorImageInfo::image(image.clone())
            }
        )
        .unwrap()
        .copy_image_to_buffer(
            CopyImageToBufferInfo::image_buffer(
                image.clone(), // image avlues are not interpreted as floating point values here, but as their actual type in memory
                buf.clone()
            )
        )
        .unwrap();

    let command_buffer =  builder.build().unwrap();
    
    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    future.wait(None).unwrap();
}
