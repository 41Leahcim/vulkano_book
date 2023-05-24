use vulkano::{VulkanLibrary, instance::{Instance, InstanceCreateInfo}};

fn main() {
    let library = VulkanLibrary::new().expect("No local Vulkan library/DLL");
    let instance = Instance::new(library, InstanceCreateInfo::default())
        .expect("failed to create instance");
    
}
