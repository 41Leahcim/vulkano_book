use std::sync::Arc;

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    device::{
        physical::PhysicalDevice, Device, DeviceCreateInfo, Queue, QueueCreateInfo, QueueFlags,
    },
    instance::{Instance, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryUsage, StandardMemoryAllocator, GenericMemoryAllocator, FreeListAllocator},
    VulkanLibrary, pipeline::{ComputePipeline, Pipeline, PipelineBindPoint}, descriptor_set::{allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet}, command_buffer::{allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo}, AutoCommandBufferBuilder, CommandBufferUsage}, sync::{self, GpuFuture},
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

fn create_data(memory_allocator: &GenericMemoryAllocator<Arc<FreeListAllocator>>) -> Subbuffer<[u32]>{
    // Create the source data and buffer
    let data_iter = 0..65536u32;
    Buffer::from_iter(
        memory_allocator,
        BufferCreateInfo{
            usage: BufferUsage::STORAGE_BUFFER, // This data will be stored for calculations performed with a compute shader
            ..Default::default()
        },
        AllocationCreateInfo{
            usage: MemoryUsage::Upload,
            ..Default::default()
        },
        data_iter
    ).expect("Failed to create buffer.")
}

mod cs{
    vulkano_shaders::shader!{
        ty: "compute",
        src: r"
// Version 4.60 of GLSL, set the GLSL version at the start of every shader
#version 460

// The number of invocations per dimension per work group (at least 32, at most 64)
// y and z are 1 for 1-dimensional arrays, but can be higher for multi-dimensional arrays
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

// A descriptor of a buffer with the name buf, which contains a uint (u32) array
layout(set = 0, binding = 0) buffer Data{
    uint data[];
} buf;

// Main function
void main(){
    // The index of the current invocation
    uint idx = gl_GlobalInvocationID.x;

    // Multiply the value at the current index with 12
    buf.data[idx] *= 12;
}
        "
    }
}

fn main() {
    // Initialize Vulkan and store a reference to a device, a graphical queue family index, and the first queue of that queue family
    let (device, queue, queue_family_index) = initialization();

    // Create a general purpose memory allocator
    let memory_allocator = StandardMemoryAllocator::new_default(device.clone());

    let data_buffer = create_data(&memory_allocator);

    // Load the shader into the device
    let shader = cs::load(device.clone())
        .expect("Failed to create shader module.");

    // Create a compute pipeline, an object that actually describes the compute operation to perform.
    let compute_pipeline = ComputePipeline::new(
        device.clone(),
        shader.entry_point("main").unwrap(),
        &(),
        None,
        |_|{}
    ).expect("Failed to create compute pipeline");

    // Create a standard descriptor set allocator
    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());

    // Retrieve the pipeline descriptor set layouts
    let pipeline_layout = compute_pipeline.layout();
    let descriptor_set_layouts = pipeline_layout.set_layouts();

    // Fetch the layout specific to the target pass
    let descriptor_set_layout_index = 0;
    let descriptor_set_layout = descriptor_set_layouts
        .get(descriptor_set_layout_index)
        .unwrap();
    
    // Create the actual descriptor set
    let descriptor_set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        descriptor_set_layout.clone(),
        [WriteDescriptorSet::buffer(0, data_buffer.clone())]
    )
    .unwrap();

    // Create the command buffer allocator
    let command_buffer_allocator = StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default()
    );

    // Create a command buffer builder
    let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
        queue_family_index,
        CommandBufferUsage::OneTimeSubmit
    ).unwrap();

    // Set the number of work groups per dimension (x, y, z)
    let work_group_counts = [1024, 1, 1];

    // Bind the compute pipeline and descriptor sets, finally dispatch it over work groups.
    command_buffer_builder
        .bind_pipeline_compute(compute_pipeline.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            compute_pipeline.layout().clone(),
            descriptor_set_layout_index as u32,
            descriptor_set
        )
        .dispatch(work_group_counts)
        .unwrap();

    // Build the command buffer
    let command_buffer = command_buffer_builder.build().unwrap();

    // Submit the command buffer
    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    // Wait for the future to complete
    future.wait(None).unwrap();

    // Check the output
    let content = data_buffer.read().unwrap();
    for (n, val) in content.iter().enumerate(){
        assert_eq!(*val, n as u32 * 12);
    }

    println!("Everything succeeded!");
}
