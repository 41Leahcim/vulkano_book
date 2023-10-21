use std::sync::Arc;

use image::{ImageBuffer, Rgba};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage},
    command_buffer::{
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
        AutoCommandBufferBuilder, CommandBufferUsage, CopyImageToBufferInfo, RenderPassBeginInfo,
        SubpassContents,
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
    pipeline::{
        graphics::{
            input_assembly::InputAssemblyState,
            vertex_input::Vertex,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, Subpass},
    sync::{self, GpuFuture},
    VulkanLibrary,
};

const IMAGE_WIDTH: u32 = 1 << 14;
const IMAGE_HEIGHT: u32 = IMAGE_WIDTH;

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

/// A simple 2D Vertex (top-left = (-1, -1), bottom-right = (1, 1))
#[derive(BufferContents, Vertex, Clone, Copy)]
#[repr(C)]
struct Vertex2D {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}

struct Triangle {
    vertices: [Vertex2D; 3],
}

impl Triangle {
    pub fn to_vec(&self) -> Vec<Vertex2D> {
        self.vertices.to_vec()
    }
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
// Version 4.60 of GLSL, set the GLSL version at the start of every shader
#version 460

layout(location = 0) in vec2 position;

void main(){
    gl_Position = vec4(position, 0.0, 1.0);
}"
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
// Version 4.60 of GLSL, set the GLSL version at the start of every shader
#version 460
        
layout(location = 0) out vec4 f_color;
        
void main(){
    f_color = vec4(1.0, 0.0, 0.0, 1.0);
}"
    }
}

fn main() {
    // Initialize Vulkan and store a reference to a device, a graphical queue family index, and the first queue of that queue family
    let (device, queue) = initialization();

    // Create a general purpose memory allocator
    let memory_allocator = StandardMemoryAllocator::new_default(device.clone());

    // Create an image
    let image = create_image(&memory_allocator, queue.clone());

    // Create a triangle
    let triangle = Triangle {
        vertices: [
            Vertex2D {
                position: [-0.5, -0.5],
            },
            Vertex2D {
                position: [0.0, 0.5],
            },
            Vertex2D {
                position: [0.5, -0.25],
            },
        ],
    };

    // Create a vertex buffer, to define the shape to draw on the screen
    let vertex_buffer = Buffer::from_iter(
        &memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::Upload,
            ..Default::default()
        },
        triangle.to_vec(),
    )
    .unwrap();

    // Create a render pass
    let render_pass = vulkano::single_pass_renderpass!(
        // The GPU to use for drawing
        device.clone(),

        // Settings to pass through the pass (only 1 group in this case)
        attachments: {
            // A settings group to pass
            color: {
                load: Clear, // Clear the image when in a pass using this value
                // Store the output of the pass in the image, use DontCare if you don't want to store
                store: Store,
                format: Format::R8G8B8A8_UNORM, // RGBA use 8 bits each, not normalized
                samples: 1 // don't use multisampling
            }
        },

        // The pass
        pass: {
            color: [color], // Use the color settings
            depth_stencil: {} // Don't use a depth stencil
        }
    )
    .unwrap();

    // Create a view for the image
    let view = ImageView::new_default(image.clone()).unwrap();

    // Create a framebuffer
    let framebuffer = Framebuffer::new(
        render_pass.clone(),
        FramebufferCreateInfo {
            attachments: vec![view],
            ..Default::default()
        },
    )
    .unwrap();

    // Create an allocator for the command buffer
    let command_buffer_allocator = StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    );

    // Create a command buffer builder for the command buffer
    let mut builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    // Begin and immediately end a render pass
    builder
        .begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into())],
                ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
            },
            SubpassContents::Inline,
        )
        .unwrap()
        .end_render_pass()
        .unwrap();

    // Load the vertex and fragment shaders
    let vs = vs::load(device.clone()).expect("Failed to create shader module");
    let fs = fs::load(device.clone()).expect("Failed to create shader module");

    // Create a viewport
    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [IMAGE_WIDTH as f32, IMAGE_HEIGHT as f32],
        depth_range: 0.0..1.0,
    };

    // Create a graphics pipeline
    let pipeline = GraphicsPipeline::start()
        // Load vertices per vertex
        .vertex_input_state(Vertex2D::per_vertex())
        // The vertex shader starts at main, and doesn't have any specialization constants.
        // A vulkan shader could have multiple entry points, so the entry point has to be set.
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        // Indicate the type of primitives, default is a list of triangles
        .input_assembly_state(InputAssemblyState::new())
        // Set the fixed viewport
        // Use only one fixed viewport, better performance but viewport can't change
        .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
        // Same as the vertex input, but this for the fragment input
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        // This graphics pipeline object concerns the first pass of the render pass.
        .render_pass(Subpass::from(render_pass, 0).unwrap())
        // Build the pipeline, now that everything is specified
        .build(device.clone())
        .unwrap();

    // Create a buffer for storing the image on RAM
    let buffer = Buffer::from_iter(
        &memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::Download,
            ..Default::default()
        },
        (0..IMAGE_WIDTH * IMAGE_HEIGHT * 8).map(|_| 0u8),
    )
    .expect("Failed to create buffer");

    // Create a new command buffer builder
    let mut builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    // Start a new render pass
    builder
        .begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into())],
                ..RenderPassBeginInfo::framebuffer(framebuffer)
            },
            SubpassContents::Inline,
        )
        .unwrap()
        // Bind the graphics pipeline
        .bind_pipeline_graphics(pipeline)
        // Bind the vertex buffer
        .bind_vertex_buffers(0, vertex_buffer)
        // Draw the triangle on the screen
        // Calling draw for each object is easier than calling it once for everything
        .draw(3, 1, 0, 0)
        .unwrap()
        // End the render pass
        .end_render_pass()
        .unwrap()
        // Copy the image to a buffer on RAM
        .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(image, buffer.clone()))
        .unwrap();

    // Build the command buffer
    let command_buffer = builder.build().unwrap();

    // Synchronize the GPU and CPU
    sync::now(device)
        // Prepare the queue and command buffer to execute
        .then_execute(queue, command_buffer)
        .unwrap()
        // Receive a signal when done, send the commands to execute
        .then_signal_fence_and_flush()
        .unwrap()
        // Wait until the commands have been executed
        .wait(None)
        .unwrap();

    // Read the buffer contents
    let buffer_content = buffer.read().unwrap();

    // Turn the buffer contents into an image
    let image =
        ImageBuffer::<Rgba<u8>, _>::from_raw(IMAGE_WIDTH, IMAGE_HEIGHT, &buffer_content[..])
            .unwrap();

    // Save the image
    image.save("image.png").unwrap();

    println!("Everything succeeded!");
}
