use std::sync::Arc;

use image::{ImageBuffer, Rgba};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
        AutoCommandBufferBuilder, CommandBufferUsage, CopyImageToBufferInfo,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{
        physical::PhysicalDevice, Device, DeviceCreateInfo, Queue, QueueCreateInfo, QueueFlags,
    },
    format::Format,
    image::{view::ImageView, ImageDimensions, StorageImage},
    instance::{Instance, InstanceCreateInfo},
    memory::allocator::{
        AllocationCreateInfo, FreeListAllocator, GenericMemoryAllocator, MemoryUsage,
        StandardMemoryAllocator,
    },
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    sync::{self, GpuFuture},
    VulkanLibrary,
};

const IMAGE_WIDTH: u32 = 1 << 14;
const IMAGE_HEIGHT: u32 = 1 << 14;
const WIDTH_INVOCATIONS: u32 = 8;
const HEIGHT_INVOCATIONS: u32 = 8;

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

fn get_queue_family_index(physical_device: Arc<PhysicalDevice>, selection: QueueFlags) -> u32 {
    physical_device
        .queue_family_properties()
        .iter()
        .enumerate()
        .position(|(_queue_family_index, queue_family_properties)| {
            queue_family_properties.queue_flags.contains(selection)
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

fn create_image(
    memory_allocator: &GenericMemoryAllocator<Arc<FreeListAllocator>>,
    queue: Arc<Queue>,
) -> Arc<StorageImage> {
    // Create an image
    StorageImage::new(
        memory_allocator,
        ImageDimensions::Dim2d {
            width: IMAGE_WIDTH,
            height: IMAGE_HEIGHT,
            array_layers: 1, // Images can be arrats of layers
        },
        Format::R8G8B8A8_UNORM, // unsigned normalized 8-bit RGBA values
        Some(queue.queue_family_index()),
    )
    .unwrap()
}

fn create_buffer(
    memory_allocator: &GenericMemoryAllocator<Arc<FreeListAllocator>>,
) -> Subbuffer<[u8]> {
    // Create a buffer
    Buffer::from_iter(
        memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::Download,
            ..Default::default()
        },
        (0..IMAGE_WIDTH * IMAGE_HEIGHT * 4).map(|_| 0u8),
    )
    .expect("Failed to create buffer")
}

mod image_shader {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
// Version 4.60 of GLSL, set the GLSL version at the start of every shader
#version 460

// The number of invocations per dimension per work group (at least 32, at most 64)
// z is 1 for 2-dimensional arrays, but can be higher for 3-dimensional arrays
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// A descriptor of an image2D with the name img, which has 4 channels of 8 bits per pixel
// Bound as the first binding of the first descriptor set
layout(set = 0, binding = 0, rgba8) uniform writeonly image2D img;

// Main function
void main(){
    // Normalize the coordinates
    vec2 norm_coordinates = (gl_GlobalInvocationID.xy + vec2(0.5)) / vec2(imageSize(img));

    // Calculate complex number that corresponds to the pixel of the image to modify
    vec2 c = (norm_coordinates - vec2(0.5)) * 2.0 - vec2(1.0, 0.0);

    // Search for the complex number in the mandelbrot set
    // It is in the mandelbrot set if z² + c diverges when iterated from z = 0 (z being a complex number)
    // It's diverging if length(z) > 4.0
    vec2 z = vec2(0.0, 0.0);
    float i;
    for(i = 0.0;i < 1.0;i += 0.005){
        z = vec2(
            z.x * z.x - z.y * z.y + c.x,
            z.y * z.x + z.x * z.y + c.y
        );
        if(length(z) > 4.0){
            break;
        }
    }

    // The closer c is to the set, the higher i will be, so use i for the current pixel
    vec4 to_write = vec4(vec3(i), 1.0);

    // Write to_write to the pixel with imageStore, to make sure correct type is written to the pixel
    imageStore(img, ivec2(gl_GlobalInvocationID.xy), to_write);
}
        "
    }
}

fn main() {
    // Initialize Vulkan and store a reference to a device, a graphical queue family index, and the first queue of that queue family
    let (device, queue) = initialization();

    // Create a general purpose memory allocator
    let memory_allocator = StandardMemoryAllocator::new_default(device.clone());

    // Create the image and buffer
    let image = create_image(&memory_allocator, queue.clone());
    let buf = create_buffer(&memory_allocator);

    // Create an image view
    let view = ImageView::new_default(image.clone()).unwrap();

    // Load the shader
    let shader = image_shader::load(device.clone()).expect("Failed to create shader module");

    // Create a compute pipeline
    let compute_pipeline = ComputePipeline::new(
        device.clone(),
        shader.entry_point("main").unwrap(),
        &(),
        None,
        |_| {},
    )
    .expect("Failed to create compute pipeline");

    // Create a standard descriptor allocator
    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());
    // Create a command buffer allocator
    let command_buffer_allocator = StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    );

    // Create a new descriptor set by adding the image view
    let layout = compute_pipeline.layout().set_layouts().get(0).unwrap();
    let set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        layout.clone(),
        [WriteDescriptorSet::image_view(0, view.clone())],
    )
    .unwrap();

    // Create a command buffer builder
    let mut builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    // Add commands to the builder to clear the image, and copy the image to the buffer
    builder
        .bind_pipeline_compute(compute_pipeline.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            compute_pipeline.layout().clone(),
            0,
            set,
        )
        .dispatch([IMAGE_WIDTH / WIDTH_INVOCATIONS, IMAGE_HEIGHT / HEIGHT_INVOCATIONS, 1])
        .unwrap()
        .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
            image.clone(), // image avlues are not interpreted as floating point values here, but as their actual type in memory
            buf.clone(),
        ))
        .unwrap();

    // Build the command buffer
    let command_buffer = builder.build().unwrap();

    // Create a future executing the commands, end with a fence signal, flush the commands
    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    // Wait for the future to be done
    future.wait(None).unwrap();

    // read the contents of the buffer into an image buffer, save the result as an image
    let buffer_content = buf.read().unwrap();
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(IMAGE_WIDTH, IMAGE_HEIGHT, &buffer_content[..]).unwrap();
    image.save("image.png").unwrap();
    println!("Everything succeeded!");
}
