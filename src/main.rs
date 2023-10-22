use std::{sync::Arc, time::Instant};

use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
        RenderPassBeginInfo, SubpassContents,
    },
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
    },
    image::{view::ImageView, ImageUsage, SwapchainImage},
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
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    shader::ShaderModule,
    swapchain::{
        AcquireError, Surface, Swapchain, SwapchainCreateInfo, SwapchainCreationError,
        SwapchainPresentInfo,
    },
    sync::{self, future::FenceSignalFuture, FlushError, GpuFuture},
    VulkanLibrary,
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

fn list_gpus(instance: &Arc<Instance>) {
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

fn list_queues(physical_device: &Arc<PhysicalDevice>) {
    for family in physical_device.queue_family_properties() {
        println!(
            "Found a queue family with {:?} queue(s)",
            family.queue_count
        );
    }
}

fn select_physical_device(
    instance: &Arc<Instance>,
    surface: &Arc<Surface>,
    device_extensions: &DeviceExtensions,
) -> (Arc<PhysicalDevice>, u32) {
    instance
        .enumerate_physical_devices()
        .expect("Could not enumerate devices")
        .filter(|p| p.supported_extensions().contains(device_extensions))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                // Find the first suitable queue family.
                // If none is found, `None` is returned to `filter_map`,
                // which disqualifies this physical device.
                .position(|(i, q)| {
                    q.queue_flags.contains(QueueFlags::GRAPHICS)
                        && p.surface_support(i as u32, surface).unwrap_or(false)
                })
                .map(|q| (p, q as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            _ => 4,
        })
        .expect("No device available")
}

#[allow(clippy::type_complexity)]
fn initialization() -> (
    Arc<PhysicalDevice>,
    Arc<Device>,
    Arc<Queue>,
    GenericMemoryAllocator<Arc<FreeListAllocator>>,
    EventLoop<()>,
    Arc<Surface>,
) {
    // Get a vulkan instance
    let library = VulkanLibrary::new().expect("No local Vulkan library/DLL");
    let required_extensions = vulkano_win::required_extensions(&library);
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            enabled_extensions: required_extensions,
            ..Default::default()
        },
    )
    .expect("Failed to create instance");

    // Create an event loop and a surface (cross-platform window abstraction)
    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    // List the GPUs with Vulkan support
    list_gpus(&instance);

    // Set the required extensions for the application
    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    };

    // Select the first gpu with the required supported extensions
    let (physical_device, queue_family_index) =
        select_physical_device(&instance, &surface, &device_extensions);

    // List the queues available on the selected device
    list_queues(&physical_device);

    // Create a new Vulkan device, returning the device and an iterator over queues on that device
    let (device, mut queues) = Device::new(
        physical_device.clone(),
        DeviceCreateInfo {
            // Here we pass the desired queue family to use by index
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            enabled_extensions: device_extensions,
            ..Default::default()
        },
    )
    .expect("Failed to create device");

    // Select the first queue
    let queue = queues.next().unwrap();

    // Create a general purpose memory allocator
    let memory_allocator = StandardMemoryAllocator::new_default(device.clone());

    (
        physical_device,
        device,
        queue,
        memory_allocator,
        event_loop,
        surface,
    )
}

/// A simple 2D Vertex (top-left = (-1, -1), bottom-right = (1, 1))
#[derive(BufferContents, Vertex, Clone, Copy)]
#[repr(C)]
struct Vertex2D {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}

impl Vertex2D {
    pub const fn new(x: f32, y: f32) -> Self {
        Self { position: [x, y] }
    }
}

struct Triangle {
    vertices: [Vertex2D; 3],
}

impl Triangle {
    pub fn new(vertices: impl Into<[Vertex2D; 3]>) -> Self {
        Self {
            vertices: vertices.into(),
        }
    }

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

fn get_render_pass(device: Arc<Device>, swapchain: &Arc<Swapchain>) -> Arc<RenderPass> {
    vulkano::single_pass_renderpass!(
        // The GPU to use for drawing
        device.clone(),

        // Settings to pass through the pass (only 1 group in this case)
        attachments: {
            // A settings group to pass
            color: {
                load: Clear, // Clear the image when in a pass using this value
                // Store the output of the pass in the image, use DontCare if you don't want to store
                store: Store,
                format: swapchain.image_format(), // Set the format the same as the swapchain
                samples: 1 // don't use multisampling
            }
        },

        // The pass
        pass: {
            color: [color], // Use the color settings
            depth_stencil: {} // Don't use a depth stencil
        }
    )
    .unwrap()
}

fn get_frame_buffers(
    images: &[Arc<SwapchainImage>],
    render_pass: &Arc<RenderPass>,
) -> Vec<Arc<Framebuffer>> {
    images
        .iter()
        .map(|image| {
            // Create a view for the image
            let view = ImageView::new_default(image.clone()).unwrap();

            // Create a framebuffer
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect()
}

fn get_pipeline(
    device: Arc<Device>,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    render_pass: Arc<RenderPass>,
    viewport: Viewport,
) -> Arc<GraphicsPipeline> {
    GraphicsPipeline::start()
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
        .unwrap()
}

fn get_command_buffers(
    command_buffer_allocator: &StandardCommandBufferAllocator,
    queue: &Arc<Queue>,
    pipeline: &Arc<GraphicsPipeline>,
    framebuffers: &[Arc<Framebuffer>],
    vertex_buffer: &Subbuffer<[Vertex2D]>,
) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
    framebuffers
        .iter()
        .map(|framebuffer| {
            // Create a new command buffer builder
            let mut builder = AutoCommandBufferBuilder::primary(
                command_buffer_allocator,
                queue.queue_family_index(),
                CommandBufferUsage::MultipleSubmit,
            )
            .unwrap();

            // Start a new render pass
            builder
                .begin_render_pass(
                    RenderPassBeginInfo {
                        clear_values: vec![Some([0.1, 0.1, 0.1, 1.0].into())],
                        ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                    },
                    SubpassContents::Inline,
                )
                .unwrap()
                // Bind the graphics pipeline
                .bind_pipeline_graphics(pipeline.clone())
                // Bind the vertex buffer
                .bind_vertex_buffers(0, vertex_buffer.clone())
                // Draw the triangle on the screen
                // Calling draw for each object is easier than calling it once for everything
                .draw(vertex_buffer.len() as u32, 1, 0, 0)
                .unwrap()
                // End the render pass
                .end_render_pass()
                .unwrap();
            Arc::new(builder.build().unwrap())
        })
        .collect()
}

fn main() {
    // Initialize Vulkan and store a reference to a device, a graphical queue family index, and the first queue of that queue family
    let (physical_device, device, queue, memory_allocator, event_loop, surface) = initialization();

    // Use the surface to create an actual window
    let window = surface
        .object()
        .unwrap()
        .clone()
        .downcast::<Window>()
        .unwrap();

    let caps = physical_device
        .surface_capabilities(&surface, Default::default())
        .expect("Failed to get surface capabilities");

    let dimensions = window.inner_size();
    let composite_alpha = caps.supported_composite_alpha.into_iter().next().unwrap();
    let image_format = Some(
        physical_device
            .surface_formats(&surface, Default::default())
            .unwrap()[0]
            .0,
    );

    let (mut swapchain, images) = Swapchain::new(
        device.clone(),
        surface.clone(),
        SwapchainCreateInfo {
            min_image_count: caps.min_image_count + 1, // How many buffers to use in the swapchain
            image_format,
            image_extent: dimensions.into(),
            image_usage: ImageUsage::COLOR_ATTACHMENT, // What the images are going to be used for
            composite_alpha,
            ..Default::default()
        },
    )
    .unwrap();

    // Create a triangle
    let triangle = Triangle::new([
        Vertex2D::new(-0.5, -0.5),
        Vertex2D::new(0.0, 0.5),
        Vertex2D::new(0.5, -0.25),
    ]);

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
    let render_pass = get_render_pass(device.clone(), &swapchain);

    // Create an allocator for the command buffer
    let command_buffer_allocator = StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    );

    // Load the vertex and fragment shaders
    let vs = vs::load(device.clone()).expect("Failed to create shader module");
    let fs = fs::load(device.clone()).expect("Failed to create shader module");

    // Create a viewport
    #[allow(clippy::cast_precision_loss)]
    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: window.inner_size().into(),
        depth_range: 0.0..1.0,
    };

    let framebuffers = get_frame_buffers(&images, &render_pass);

    // Create a graphics pipeline
    let pipeline = get_pipeline(
        device.clone(),
        vs.clone(),
        fs.clone(),
        render_pass.clone(),
        viewport.clone(),
    );

    let mut command_buffers = get_command_buffers(
        &command_buffer_allocator,
        &queue,
        &pipeline,
        &framebuffers,
        &vertex_buffer,
    );

    let mut window_resized = false;
    let mut recreate_swapchain = false;
    let frames_in_flight = images.len();
    let mut fences: Vec<Option<Arc<FenceSignalFuture<_>>>> = vec![None; frames_in_flight];
    let mut previous_fence_i = 0;
    let mut last_frame = Instant::now();

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => *control_flow = ControlFlow::Exit,
        Event::WindowEvent {
            event: WindowEvent::Resized(_),
            ..
        } => window_resized = true,
        Event::MainEventsCleared => {
            if recreate_swapchain || window_resized {
                recreate_swapchain = false;

                let new_dimensions = window.inner_size();

                let (new_swapchain, new_images) = match swapchain.recreate(SwapchainCreateInfo {
                    // Here, "image_extend" will correspond to the window dimensions
                    image_extent: new_dimensions.into(),
                    ..swapchain.create_info()
                }) {
                    Ok(r) => r,
                    // This error tends to happen when the user is manually resizing the window.
                    // Simple restarting the loop is the easiest way to fix this issue.
                    Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
                    Err(e) => panic!("Failed to recreate swapchain: {e}"),
                };
                swapchain = new_swapchain;

                let new_framebuffers = get_frame_buffers(&new_images, &render_pass);

                if window_resized {
                    window_resized = false;

                    viewport.dimensions = new_dimensions.into();

                    let new_pipeline = get_pipeline(
                        device.clone(),
                        vs.clone(),
                        fs.clone(),
                        render_pass.clone(),
                        viewport.clone(),
                    );

                    command_buffers = get_command_buffers(
                        &command_buffer_allocator,
                        &queue,
                        &new_pipeline,
                        &new_framebuffers,
                        &vertex_buffer,
                    );
                }
            }

            // Take the image index on which should be drawn, aswell as the future representing the
            // moment when the GPU will gain access to the image.
            // If no image is available (which hapens when the draw command is submitted
            // too quickly), then the function will block until there is. Second parameter if for
            // the timeout.
            let (image_i, suboptimal, acquire_future) =
                match vulkano::swapchain::acquire_next_image(swapchain.clone(), None) {
                    Ok(r) => r,
                    Err(AcquireError::OutOfDate) => {
                        recreate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("Failed to acquire next image: {e}"),
                };

            // The acquire function may be suboptimal where the swapchain will still work, but may
            // not get properly displayed. If that happens, the swapchain should be recreated.
            if suboptimal {
                recreate_swapchain = true;
            }

            // Wait for the fence related to this image to finish.
            // Normally this would be the oldest fence, that most likely has already finished
            if let Some(image_fence) = &fences[image_i as usize] {
                image_fence.wait(None).unwrap();
            }

            let previous_future = match fences[previous_fence_i as usize].clone() {
                // Use the existing fence signal
                Some(fence) => fence.boxed(),

                // Create a NowFuture
                None => {
                    let mut now = sync::now(device.clone());
                    now.cleanup_finished();
                    now.boxed()
                }
            };

            // Create the future to submit to the GPU
            let future = previous_future
                .join(acquire_future)
                .then_execute(queue.clone(), command_buffers[image_i as usize].clone())
                .unwrap()
                .then_swapchain_present(
                    queue.clone(),
                    SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_i),
                )
                .then_signal_fence_and_flush();

            fences[image_i as usize] = match future {
                #[allow(clippy::arc_with_non_send_sync)]
                Ok(value) => Some(Arc::new(value)),
                Err(FlushError::OutOfDate) => {
                    recreate_swapchain = true;
                    None
                }
                Err(e) => {
                    println!("Failed to flush future: {e}");
                    None
                }
            };

            previous_fence_i = image_i;

            let now = Instant::now();
            println!(
                "{} fps",
                (1.0 / (now - last_frame).as_secs_f64()).round() as i64
            );
            last_frame = now;
        }
        _ => (),
    });
}
