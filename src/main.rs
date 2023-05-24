use std::sync::Arc;

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    device::{
        physical::PhysicalDevice, Device, DeviceCreateInfo, Queue, QueueCreateInfo, QueueFlags,
    },
    instance::{Instance, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryUsage, StandardMemoryAllocator, GenericMemoryAllocator, FreeListAllocator},
    VulkanLibrary, command_buffer::{allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo}, AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo}, sync::{self, GpuFuture},
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

fn create_data(memory_allocator: &GenericMemoryAllocator<Arc<FreeListAllocator>>) -> (Subbuffer<[i32]>, Subbuffer<[i32]>){
    // Create the source data and buffer
    let source_content = 0..1_000_000_000;
    let source = Buffer::from_iter(
        memory_allocator,
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
    let destination_content = (0..1_000_000_000).map(|_| 0);
    let destination = Buffer::from_iter(
        memory_allocator,
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
    (source, destination)
}

fn main() {
    // Initialize Vulkan and store a reference to a device, a graphical queue family index, and the first queue of that queue family
    let (device, queue, queue_family_index) = initialization();

    // Create a general purpose memory allocator
    let memory_allocator = StandardMemoryAllocator::new_default(device.clone());

    let (source, destination) = create_data(&memory_allocator);

    // Create a standard command buffer allocator, required to make the gpu perform operations
    let command_buffer_allocator = StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default()
    );

    // Create a primary command buffer builder which can only submit commands once
    let mut builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
        queue_family_index,
        CommandBufferUsage::OneTimeSubmit
    ).unwrap();

    // Set the operation(s) to perform, in this case we just want to move data from the source to the destination buffer
    builder
        .copy_buffer(CopyBufferInfo::buffers(source.clone(), destination.clone()))
        .unwrap();

    // Build the command buffer
    let command_buffer = builder.build().unwrap();

    // Execute the command buffer on the gpu
    // Fence will return a future letting the cpu know when the gpu is done
    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush() // Same as signal fence, and then flush
        .unwrap();

    // Wait for the gpu to be done with the calculations, don't limit the waiting time
    future.wait(None).unwrap();
    
    // Read the data to check whether the operation was successful
    let src_content = source.read().unwrap();
    let destination_content = destination.read().unwrap();
    assert_eq!(&*src_content, &*destination_content);
    println!("Everyting succeeded!");
}
