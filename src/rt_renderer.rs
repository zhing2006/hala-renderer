use std::rc::Rc;
use std::cell::RefCell;
use std::path::Path;
use std::io::Write;

use hala_gfx::{
  HalaGPURequirements,
  HalaContext,
};

use crate::error::HalaRendererError;
use crate::scene::{
  cpu,
  gpu,
  loader,
};

/// The type of the environment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HalaEnvType(u8);
impl HalaEnvType {
  pub const SKY: Self = Self(0);
  pub const MAP: Self = Self(1);

  pub fn from_u8(value: u8) -> Self {
    match value {
      0 => Self::SKY,
      1 => Self::MAP,
      _ => panic!("Invalid light type."),
    }
  }

  pub fn to_u8(&self) -> u8 {
    self.0
  }
}


#[repr(C, align(4))]
#[derive(Debug, Clone, Copy)]
pub struct HalaGlobalUniform {
  pub ground_color: glam::Vec4,
  pub sky_color: glam::Vec4,
  pub resolution: glam::Vec2,
  pub max_depth: u32,
  pub rr_depth: u32,
  pub frame_index: u32,
  pub camera_index: u32,
  pub env_type: u32,
  pub env_map_width: u32,
  pub env_map_height: u32,
  pub env_total_sum: f32,
  pub env_rotation: f32,
  pub env_intensity: f32,
  pub enable_tonemap: u32,
  pub enable_aces: u32,
  pub use_simple_aces: u32,
  pub num_of_lights: u32,
}

/// The renderer.
pub struct HalaRenderer {
  pub name: String,
  pub width: u32,
  pub height: u32,
  pub max_depth: u32,
  pub rr_depth: u32,
  pub enable_tonemap: bool,
  pub enable_aces: bool,
  pub use_simple_aces: bool,
  pub max_frames: u64,

  pub context: std::mem::ManuallyDrop<HalaContext>,
  pub is_ray_tracing: bool,

  pub(crate) graphics_command_buffers: std::mem::ManuallyDrop<hala_gfx::HalaCommandBufferSet>,
  pub(crate) transfer_command_buffers: std::mem::ManuallyDrop<hala_gfx::HalaCommandBufferSet>,
  pub(crate) transfer_staging_buffer: std::mem::ManuallyDrop<hala_gfx::HalaBuffer>,

  pub(crate) descriptor_pool: std::mem::ManuallyDrop<Rc<RefCell<hala_gfx::HalaDescriptorPool>>>,
  pub(crate) static_descriptor_set: std::mem::ManuallyDrop<hala_gfx::HalaDescriptorSet>,
  pub(crate) dynamic_descriptor_set: std::mem::ManuallyDrop<hala_gfx::HalaDescriptorSet>,
  pub(crate) global_uniform_buffer: std::mem::ManuallyDrop<hala_gfx::HalaBuffer>,
  pub(crate) final_image: std::mem::ManuallyDrop<hala_gfx::HalaImage>,
  pub(crate) final_image_binding_index: u32,
  pub(crate) accum_image: std::mem::ManuallyDrop<hala_gfx::HalaImage>,
  pub(crate) accum_image_binding_index: u32,
  pub(crate) albedo_image: std::mem::ManuallyDrop<hala_gfx::HalaImage>,
  pub(crate) albedo_image_binding_index: u32,
  pub(crate) normal_image: std::mem::ManuallyDrop<hala_gfx::HalaImage>,
  pub(crate) normal_image_binding_index: u32,

  pub(crate) raygen_shaders: Vec<hala_gfx::HalaShader>,
  pub(crate) miss_shaders: Vec<hala_gfx::HalaShader>,
  pub(crate) hit_shaders: Vec<(Option<hala_gfx::HalaShader>, Option<hala_gfx::HalaShader>, Option<hala_gfx::HalaShader>)>,
  pub(crate) callable_shaders: Vec<hala_gfx::HalaShader>,
  pub(crate) pipeline: Option<hala_gfx::HalaRayTracingPipeline>,
  pub(crate) sbt: Option<hala_gfx::HalaShaderBindingTable>,

  pub(crate) blue_noise_image: Option<hala_gfx::HalaImage>,
  pub(crate) scene_in_gpu: Option<gpu::HalaScene>,

  pub(crate) envmap: Option<crate::envmap::EnvMap>,
  pub(crate) env_rotation: f32,
  pub(crate) env_ground_color: glam::Vec4,
  pub(crate) env_sky_color: glam::Vec4,
  pub(crate) env_intensity: f32,

  pub(crate) textures_descriptor_set: Option<hala_gfx::HalaDescriptorSet>,

  pub(crate) host_accessible_buffer: std::mem::ManuallyDrop<hala_gfx::HalaBuffer>,

  // Render data.
  image_index: usize,
  is_device_lost: bool,

  // Statistic.
  total_frames: u64,
  last_stat_time: std::time::Instant,
  elapsed_time: std::time::Duration,
  total_gpu_nanoseconds: u128,
  total_gpu_frames: u64,
}

/// The Drop implementation of the renderer.
impl Drop for HalaRenderer {

  fn drop(&mut self) {
    unsafe {
      std::mem::ManuallyDrop::drop(&mut self.host_accessible_buffer);
      self.textures_descriptor_set = None;
      self.envmap = None;
      self.scene_in_gpu = None;
      self.blue_noise_image = None;
      self.sbt = None;
      self.pipeline = None;
      self.raygen_shaders.clear();
      self.miss_shaders.clear();
      self.hit_shaders.clear();
      self.callable_shaders.clear();
      std::mem::ManuallyDrop::drop(&mut self.normal_image);
      std::mem::ManuallyDrop::drop(&mut self.albedo_image);
      std::mem::ManuallyDrop::drop(&mut self.accum_image);
      std::mem::ManuallyDrop::drop(&mut self.final_image);
      std::mem::ManuallyDrop::drop(&mut self.global_uniform_buffer);
      std::mem::ManuallyDrop::drop(&mut self.dynamic_descriptor_set);
      std::mem::ManuallyDrop::drop(&mut self.static_descriptor_set);
      std::mem::ManuallyDrop::drop(&mut self.descriptor_pool);
      std::mem::ManuallyDrop::drop(&mut self.transfer_staging_buffer);
      std::mem::ManuallyDrop::drop(&mut self.transfer_command_buffers);
      std::mem::ManuallyDrop::drop(&mut self.graphics_command_buffers);
      std::mem::ManuallyDrop::drop(&mut self.context);
    }
    log::debug!("A HalaRenderer \"{}\" is dropped.", self.name);
  }

}

/// The implementation of the renderer.
impl HalaRenderer {

  /// Create a new renderer.
  /// param name: The name of the renderer.
  /// param gpu_req: The GPU requirements of the renderer.
  /// param window: The window of the renderer.
  /// param is_ray_tracing: Whether the renderer is for ray tracing.
  /// param max_depth: The max depth of the ray tracing.
  /// param rr_depth: The Russian Roulette depth of the ray tracing.
  /// return: The renderer.
  #[allow(clippy::too_many_arguments)]
  pub fn new(
    name: &str,
    gpu_req: &HalaGPURequirements,
    window: &winit::window::Window,
    is_ray_tracing: bool,
    max_depth: u32,
    rr_depth: u32,
    enable_tonemap: bool,
    enable_aces: bool,
    use_simple_aces: bool,
    max_frames: u64,
  ) -> Result<Self, HalaRendererError> {
    let context = HalaContext::new(name, gpu_req, window)?;
    let width = gpu_req.width;
    let height = gpu_req.height;

    // Craete command buffers.
    let graphics_command_buffers = hala_gfx::HalaCommandBufferSet::new(
      Rc::clone(&context.logical_device),
      Rc::clone(&context.pools),
      hala_gfx::HalaCommandBufferType::GRAPHICS,
      hala_gfx::HalaCommandBufferLevel::PRIMARY,
      context.swapchain.num_of_images,
      "main_graphics.cmd_buffer",
    )?;
    let transfer_command_buffers = hala_gfx::HalaCommandBufferSet::new(
      Rc::clone(&context.logical_device),
      Rc::clone(&context.pools),
      hala_gfx::HalaCommandBufferType::TRANSFER,
      hala_gfx::HalaCommandBufferLevel::PRIMARY,
      context.swapchain.num_of_images,
      "main_transfer.cmd_buffer",
    )?;
    let transfer_staging_buffer = hala_gfx::HalaBuffer::new(
      Rc::clone(&context.logical_device),
      256 * 1024 * 1024, // 4096 * 4096 * RGBA32F = 256MB
      hala_gfx::HalaBufferUsageFlags::TRANSFER_SRC,
      hala_gfx::HalaMemoryLocation::CpuToGpu,
      "transfer_staging.buffer",
    )?;

    // Create descriptors.
    let descriptor_pool = Rc::new(RefCell::new(hala_gfx::HalaDescriptorPool::new(
      Rc::clone(&context.logical_device),
      &[
        (
          hala_gfx::HalaDescriptorType::ACCELERATION_STRUCTURE,
          4,
        ),
        (
          hala_gfx::HalaDescriptorType::STORAGE_IMAGE,
          8,
        ),
        (
          hala_gfx::HalaDescriptorType::UNIFORM_BUFFER,
          16,
        ),
        (
          hala_gfx::HalaDescriptorType::STORAGE_BUFFER,
          32,
        ),
        (
          hala_gfx::HalaDescriptorType::SAMPLED_IMAGE,
          64,
        ),
        (
          hala_gfx::HalaDescriptorType::SAMPLER,
          16,
        ),
        (
          hala_gfx::HalaDescriptorType::COMBINED_IMAGE_SAMPLER,
          256,
        ),
      ],
      512,
      "main.descriptor_pool"
    )?));

    let static_descriptor_set = hala_gfx::HalaDescriptorSet::new_static(
      Rc::clone(&context.logical_device),
      Rc::clone(&descriptor_pool),
      hala_gfx::HalaDescriptorSetLayout::new(
        Rc::clone(&context.logical_device),
        &[
          ( // Acceleration structure.
            0,
            hala_gfx::HalaDescriptorType::ACCELERATION_STRUCTURE,
            1,
            hala_gfx::HalaShaderStageFlags::RAYGEN | hala_gfx::HalaShaderStageFlags::CLOSEST_HIT,
            hala_gfx::HalaDescriptorBindingFlags::PARTIALLY_BOUND
          ),
          ( // Final image.
            1,
            hala_gfx::HalaDescriptorType::STORAGE_IMAGE,
            1,
            hala_gfx::HalaShaderStageFlags::RAYGEN,
            hala_gfx::HalaDescriptorBindingFlags::PARTIALLY_BOUND
          ),
          ( // Accum image.
            2,
            hala_gfx::HalaDescriptorType::STORAGE_IMAGE,
            1,
            hala_gfx::HalaShaderStageFlags::RAYGEN | hala_gfx::HalaShaderStageFlags::CALLABLE,
            hala_gfx::HalaDescriptorBindingFlags::PARTIALLY_BOUND
          ),
          ( // Albedo image.
            3,
            hala_gfx::HalaDescriptorType::STORAGE_IMAGE,
            1,
            hala_gfx::HalaShaderStageFlags::RAYGEN | hala_gfx::HalaShaderStageFlags::CALLABLE,
            hala_gfx::HalaDescriptorBindingFlags::PARTIALLY_BOUND
          ),
          ( // Normal image.
            4,
            hala_gfx::HalaDescriptorType::STORAGE_IMAGE,
            1,
            hala_gfx::HalaShaderStageFlags::RAYGEN | hala_gfx::HalaShaderStageFlags::CALLABLE,
            hala_gfx::HalaDescriptorBindingFlags::PARTIALLY_BOUND
          ),
          ( // Blue noise image.
            5,
            hala_gfx::HalaDescriptorType::SAMPLED_IMAGE,
            1,
            hala_gfx::HalaShaderStageFlags::RAYGEN | hala_gfx::HalaShaderStageFlags::CALLABLE,
            hala_gfx::HalaDescriptorBindingFlags::PARTIALLY_BOUND
          ),
          ( // Environment map(skybox).
            6,
            hala_gfx::HalaDescriptorType::COMBINED_IMAGE_SAMPLER,
            1,
            hala_gfx::HalaShaderStageFlags::RAYGEN | hala_gfx::HalaShaderStageFlags::CALLABLE,
            hala_gfx::HalaDescriptorBindingFlags::PARTIALLY_BOUND
          ),
          ( // Environment map(marginal distribution, conditional distribution)
            7,
            hala_gfx::HalaDescriptorType::SAMPLED_IMAGE,
            2,
            hala_gfx::HalaShaderStageFlags::RAYGEN | hala_gfx::HalaShaderStageFlags::CALLABLE,
            hala_gfx::HalaDescriptorBindingFlags::PARTIALLY_BOUND
          ),
          ( // Environment distribution sampler.
            8,
            hala_gfx::HalaDescriptorType::SAMPLER,
            1,
            hala_gfx::HalaShaderStageFlags::RAYGEN | hala_gfx::HalaShaderStageFlags::CALLABLE,
            hala_gfx::HalaDescriptorBindingFlags::PARTIALLY_BOUND
          )
        ],
        "main_static.descriptor_set_layout",
      )?,
      1,
      0,
      "main_static.descriptor_set",
    )?;

    let dynamic_descriptor_set = hala_gfx::HalaDescriptorSet::new(
      Rc::clone(&context.logical_device),
      Rc::clone(&descriptor_pool),
      hala_gfx::HalaDescriptorSetLayout::new(
        Rc::clone(&context.logical_device),
        &[
          ( // Main uniform buffer.
            0,
            hala_gfx::HalaDescriptorType::UNIFORM_BUFFER,
            1,
            hala_gfx::HalaShaderStageFlags::RAYGEN | hala_gfx::HalaShaderStageFlags::CLOSEST_HIT | hala_gfx::HalaShaderStageFlags::CALLABLE,
            hala_gfx::HalaDescriptorBindingFlags::PARTIALLY_BOUND
          ),
          ( // Camera uniform buffer.
            1,
            hala_gfx::HalaDescriptorType::UNIFORM_BUFFER,
            1,
            hala_gfx::HalaShaderStageFlags::RAYGEN | hala_gfx::HalaShaderStageFlags::CALLABLE,
            hala_gfx::HalaDescriptorBindingFlags::PARTIALLY_BOUND
          ),
          ( // Light uniform buffer.
            2,
            hala_gfx::HalaDescriptorType::UNIFORM_BUFFER,
            1,
            hala_gfx::HalaShaderStageFlags::RAYGEN | hala_gfx::HalaShaderStageFlags::INTERSECTION | hala_gfx::HalaShaderStageFlags::CALLABLE,
            hala_gfx::HalaDescriptorBindingFlags::PARTIALLY_BOUND
          ),
          ( // Material storage buffer.
            3,
            hala_gfx::HalaDescriptorType::STORAGE_BUFFER,
            1,
            hala_gfx::HalaShaderStageFlags::RAYGEN | hala_gfx::HalaShaderStageFlags::CALLABLE,
            hala_gfx::HalaDescriptorBindingFlags::PARTIALLY_BOUND
          ),
          ( // Primitive storage buffer.
            4,
            hala_gfx::HalaDescriptorType::STORAGE_BUFFER,
            1,
            hala_gfx::HalaShaderStageFlags::RAYGEN | hala_gfx::HalaShaderStageFlags::CLOSEST_HIT,
            hala_gfx::HalaDescriptorBindingFlags::PARTIALLY_BOUND
          ),
        ],
        "main_dynamic.descriptor_set_layout",
      )?,
      context.swapchain.num_of_images,
      0,
      "main_dynamic.descriptor_set",
    )?;

    // Create global uniform buffer.
    let uniform_buffer = hala_gfx::HalaBuffer::new(
      Rc::clone(&context.logical_device),
      std::mem::size_of::<HalaGlobalUniform>() as u64,
      hala_gfx::HalaBufferUsageFlags::UNIFORM_BUFFER,
      hala_gfx::HalaMemoryLocation::CpuToGpu,
      "global.uniform_buffer",
    )?;

    // Create storage image.
    let (
      final_image,
      accum_image,
      albedo_image,
      normal_image,
      host_accessible_buffer,
    ) = Self::create_storage_images(&context)?;

    // Return the renderer.
    log::debug!("A HalaRenderer \"{}\"[{} x {}] is created.", name, width, height);
    Ok(Self {
      name: name.to_string(),
      width,
      height,
      max_depth,
      rr_depth,
      enable_tonemap,
      enable_aces,
      use_simple_aces,
      max_frames: if max_frames == 0 { std::u64::MAX } else { max_frames },

      context: std::mem::ManuallyDrop::new(context),
      is_ray_tracing,
      graphics_command_buffers: std::mem::ManuallyDrop::new(graphics_command_buffers),
      transfer_command_buffers: std::mem::ManuallyDrop::new(transfer_command_buffers),
      transfer_staging_buffer: std::mem::ManuallyDrop::new(transfer_staging_buffer),
      descriptor_pool: std::mem::ManuallyDrop::new(descriptor_pool),
      static_descriptor_set: std::mem::ManuallyDrop::new(static_descriptor_set),
      dynamic_descriptor_set: std::mem::ManuallyDrop::new(dynamic_descriptor_set),
      global_uniform_buffer: std::mem::ManuallyDrop::new(uniform_buffer),
      final_image: std::mem::ManuallyDrop::new(final_image),
      final_image_binding_index: 0,
      accum_image: std::mem::ManuallyDrop::new(accum_image),
      accum_image_binding_index: 0,
      albedo_image: std::mem::ManuallyDrop::new(albedo_image),
      albedo_image_binding_index: 0,
      normal_image: std::mem::ManuallyDrop::new(normal_image),
      normal_image_binding_index: 0,
      raygen_shaders: Vec::new(),
      miss_shaders: Vec::new(),
      hit_shaders: Vec::new(),
      callable_shaders: Vec::new(),
      pipeline: None,
      sbt: None,
      blue_noise_image: None,
      scene_in_gpu: None,
      envmap: None,
      env_rotation: 0.0,
      env_ground_color: glam::Vec4::new(1.0, 1.0, 1.0, 1.0),
      env_sky_color: glam::Vec4::new(0.5, 0.7, 1.0, 1.0),
      env_intensity: 1.0,

      textures_descriptor_set: None,

      host_accessible_buffer: std::mem::ManuallyDrop::new(host_accessible_buffer),

      image_index: 0,
      is_device_lost: false,

      total_frames: 0,
      last_stat_time: std::time::Instant::now(),
      elapsed_time: std::time::Duration::from_secs(0),
      total_gpu_nanoseconds: 0,
      total_gpu_frames: 0,
    })
  }

  /// Create storage images.
  /// param context: The context.
  /// return: The result(final_image, accum_image, albedo_image, normal_image).
  fn create_storage_images(context: &hala_gfx::HalaContext)
    -> Result<(hala_gfx::HalaImage, hala_gfx::HalaImage, hala_gfx::HalaImage, hala_gfx::HalaImage, hala_gfx::HalaBuffer), HalaRendererError>
  {
    let final_image = hala_gfx::HalaImage::new_2d(
      Rc::clone(&context.logical_device),
      hala_gfx::HalaImageUsageFlags::STORAGE | hala_gfx::HalaImageUsageFlags::TRANSFER_SRC,
      context.swapchain.desc.format.into(),
      context.gpu_req.width,
      context.gpu_req.height,
      1,
      1,
      hala_gfx::HalaMemoryLocation::GpuOnly,
      "final.image",
    )?;
    let accum_image = hala_gfx::HalaImage::new_2d(
      Rc::clone(&context.logical_device),
      hala_gfx::HalaImageUsageFlags::STORAGE | hala_gfx::HalaImageUsageFlags::TRANSFER_SRC,
      hala_gfx::HalaFormat::R32G32B32A32_SFLOAT,
      context.gpu_req.width,
      context.gpu_req.height,
      1,
      1,
      hala_gfx::HalaMemoryLocation::GpuOnly,
      "accum.image",
    )?;
    let albedo_image = hala_gfx::HalaImage::new_2d(
      Rc::clone(&context.logical_device),
      hala_gfx::HalaImageUsageFlags::STORAGE | hala_gfx::HalaImageUsageFlags::TRANSFER_SRC,
      hala_gfx::HalaFormat::R32G32B32A32_SFLOAT,
      context.gpu_req.width,
      context.gpu_req.height,
      1,
      1,
      hala_gfx::HalaMemoryLocation::GpuOnly,
      "albedo.image",
    )?;
    let normal_image = hala_gfx::HalaImage::new_2d(
      Rc::clone(&context.logical_device),
      hala_gfx::HalaImageUsageFlags::STORAGE | hala_gfx::HalaImageUsageFlags::TRANSFER_SRC,
      hala_gfx::HalaFormat::R32G32B32A32_SFLOAT,
      context.gpu_req.width,
      context.gpu_req.height,
      1,
      1,
      hala_gfx::HalaMemoryLocation::GpuOnly,
      "normal.image",
    )?;

    let host_accessible_buffer = hala_gfx::HalaBuffer::new(
      Rc::clone(&context.logical_device),
      4 * 4 * context.gpu_req.width as u64 * context.gpu_req.height as u64, // 4 * float32 * width * height
      hala_gfx::HalaBufferUsageFlags::TRANSFER_DST,
      hala_gfx::HalaMemoryLocation::GpuToCpu,
      "host_accessible.buffer",
    )?;

    // Transfer the final image layout to GENERAL.
    {
      let command_buffers = hala_gfx::HalaCommandBufferSet::new(
        Rc::clone(&context.logical_device),
        Rc::clone(&context.short_time_pools),
        hala_gfx::HalaCommandBufferType::GRAPHICS,
        hala_gfx::HalaCommandBufferLevel::PRIMARY,
        1,
        "one_time.command_buffers",
      )?;

      command_buffers.begin(0, hala_gfx::HalaCommandBufferUsageFlags::ONE_TIME_SUBMIT)?;

      let images = [final_image.raw, accum_image.raw, albedo_image.raw, normal_image.raw];
      for image in images.into_iter() {
        command_buffers.set_image_barriers(
          0,
          &[hala_gfx::HalaImageBarrierInfo {
            image,
            old_layout: hala_gfx::HalaImageLayout::UNDEFINED,
            new_layout: hala_gfx::HalaImageLayout::GENERAL,
            src_access_mask: hala_gfx::HalaAccessFlags2::NONE,
            dst_access_mask: hala_gfx::HalaAccessFlags2::NONE,
            src_stage_mask: hala_gfx::HalaPipelineStageFlags2::NONE,
            dst_stage_mask: hala_gfx::HalaPipelineStageFlags2::ALL_COMMANDS,
            aspect_mask: hala_gfx::HalaImageAspectFlags::COLOR,
            ..Default::default()
          }],
        );
      }

      command_buffers.end(0)?;

      context.logical_device.borrow().graphics_submit(
        &command_buffers,
        0,
        0,
      )?;

      context.logical_device.borrow().graphics_wait(0)?;
    }

    Ok((final_image, accum_image, albedo_image, normal_image, host_accessible_buffer))
  }

  /// Push a general shader to the renderer.
  /// param code: The compiled shader code.
  /// param stage: The shader stage.
  /// param rt_group_type: The ray tracing shader group type.
  /// param debug_name: The debug name.
  /// return The result.
  pub fn push_general_shader(
    &mut self,
    code: &[u8],
    stage: hala_gfx::HalaShaderStageFlags,
    rt_group_type: hala_gfx::HalaRayTracingShaderGroupType,
    debug_name: &str) -> Result<(), HalaRendererError>
  {
    let shader = hala_gfx::HalaShader::new(
      Rc::clone(&self.context.logical_device),
      code,
      stage,
      rt_group_type,
      debug_name,
    )?;
    match stage {
      hala_gfx::HalaShaderStageFlags::RAYGEN => {
        self.raygen_shaders.push(shader);
      },
      hala_gfx::HalaShaderStageFlags::MISS => {
        self.miss_shaders.push(shader);
      },
      hala_gfx::HalaShaderStageFlags::CALLABLE => {
        self.callable_shaders.push(shader);
      },
      _ => {
        return Err(HalaRendererError::new("The shader is not general shader!", None));
      }
    }

    Ok(())
  }

  /// Push a general shader to the renderer with file.
  /// param code: The compiled shader code.
  /// param stage: The shader stage.
  /// param rt_group_type: The ray tracing shader group type.
  /// param debug_name: The debug name.
  /// return The result.
  pub fn push_general_shader_with_file(
    &mut self,
    file_path: &str,
    stage: hala_gfx::HalaShaderStageFlags,
    rt_group_type: hala_gfx::HalaRayTracingShaderGroupType,
    debug_name: &str) -> Result<(), HalaRendererError>
  {
    let shader = hala_gfx::HalaShader::with_file(
      Rc::clone(&self.context.logical_device),
      file_path,
      stage,
      rt_group_type,
      debug_name,
    )?;
    match stage {
      hala_gfx::HalaShaderStageFlags::RAYGEN => {
        self.raygen_shaders.push(shader);
      },
      hala_gfx::HalaShaderStageFlags::MISS => {
        self.miss_shaders.push(shader);
      },
      hala_gfx::HalaShaderStageFlags::CALLABLE => {
        self.callable_shaders.push(shader);
      },
      _ => {
        return Err(HalaRendererError::new("The shader is not general shader!", None));
      }
    }

    Ok(())
  }

  /// Push a hit shaders to the renderer.
  /// param closest_code: The compiled closest hit shader code.
  /// param any_code: The compiled any hit shader code.
  /// param intersection_code: The compiled intersection shader code.
  /// param debug_name: The debug name.
  /// return The result.
  pub fn push_hit_shaders(
    &mut self,
    closest_code: Option<&[u8]>,
    any_code: Option<&[u8]>,
    intersection_code: Option<&[u8]>,
    debug_name: &str) -> Result<(), HalaRendererError>
  {
    let closest_shader = match closest_code {
      Some(code) => Some(hala_gfx::HalaShader::new(
        Rc::clone(&self.context.logical_device),
        code,
        hala_gfx::HalaShaderStageFlags::CLOSEST_HIT,
        match intersection_code {
          Some(_) => hala_gfx::HalaRayTracingShaderGroupType::TRIANGLES_HIT_GROUP,
          None => hala_gfx::HalaRayTracingShaderGroupType::PROCEDURAL_HIT_GROUP,
        },
        &format!("{}.rchit", debug_name),
      )?),
      None => None,
    };
    let any_shader = match any_code {
      Some(code) => Some(hala_gfx::HalaShader::new(
        Rc::clone(&self.context.logical_device),
        code,
        hala_gfx::HalaShaderStageFlags::ANY_HIT,
        match intersection_code {
          Some(_) => hala_gfx::HalaRayTracingShaderGroupType::TRIANGLES_HIT_GROUP,
          None => hala_gfx::HalaRayTracingShaderGroupType::PROCEDURAL_HIT_GROUP,
        },
        &format!("{}.rahit", debug_name),
      )?),
      None => None,
    };
    let intersection_shader = match intersection_code {
      Some(code) => Some(hala_gfx::HalaShader::new(
        Rc::clone(&self.context.logical_device),
        code,
        hala_gfx::HalaShaderStageFlags::INTERSECTION,
        hala_gfx::HalaRayTracingShaderGroupType::PROCEDURAL_HIT_GROUP,
        &format!("{}.rint", debug_name),
      )?),
      None => None,
    };

    self.hit_shaders.push((closest_shader, any_shader, intersection_shader));

    Ok(())
  }

  /// Push a hit shader to the renderer with file.
  /// param closest_file_path: The closest hit shader file path.
  /// param any_file_path: The any hit shader file path.
  /// param intersection_file_path: The intersection shader file path.
  /// param debug_name: The debug name.
  /// return The result.
  pub fn push_hit_shaders_with_file(
    &mut self,
    closest_file_path: Option<&str>,
    any_file_path: Option<&str>,
    intersection_file_path: Option<&str>,
    debug_name: &str) -> Result<(), HalaRendererError>
  {
    // closest_file_path, any_file_path and intersection_file_path can not be all none.
    if closest_file_path.is_none() && any_file_path.is_none() && intersection_file_path.is_none() {
      return Err(HalaRendererError::new("All hit shaders are none!", None));
    }

    let closest_shader = match closest_file_path {
      Some(file_path) => Some(hala_gfx::HalaShader::with_file(
        Rc::clone(&self.context.logical_device),
        file_path,
        hala_gfx::HalaShaderStageFlags::CLOSEST_HIT,
        match intersection_file_path {
          Some(_) => hala_gfx::HalaRayTracingShaderGroupType::TRIANGLES_HIT_GROUP,
          None => hala_gfx::HalaRayTracingShaderGroupType::PROCEDURAL_HIT_GROUP,
        },
        &format!("{}.rchit.spv", debug_name),
      )?),
      None => None,
    };
    let any_shader = match any_file_path {
      Some(file_path) => Some(hala_gfx::HalaShader::with_file(
        Rc::clone(&self.context.logical_device),
        file_path,
        hala_gfx::HalaShaderStageFlags::ANY_HIT,
        match intersection_file_path {
          Some(_) => hala_gfx::HalaRayTracingShaderGroupType::TRIANGLES_HIT_GROUP,
          None => hala_gfx::HalaRayTracingShaderGroupType::PROCEDURAL_HIT_GROUP,
        },
        &format!("{}.rahit.spv", debug_name),
      )?),
      None => None,
    };
    let intersection_shader = match intersection_file_path {
      Some(file_path) => Some(hala_gfx::HalaShader::with_file(
        Rc::clone(&self.context.logical_device),
        file_path,
        hala_gfx::HalaShaderStageFlags::INTERSECTION,
        hala_gfx::HalaRayTracingShaderGroupType::PROCEDURAL_HIT_GROUP,
        &format!("{}.rint.spv", debug_name),
      )?),
      None => None,
    };

    self.hit_shaders.push((closest_shader, any_shader, intersection_shader));

    Ok(())
  }

  /// Load blue noise texture.
  /// param path: The path of the blue noise texture.
  /// return: The result.
  pub fn load_blue_noise_texture<P: AsRef<Path>>(&mut self, path: P) -> Result<(), HalaRendererError> {
    let path = path.as_ref();
    let file_name = path.file_stem().ok_or(HalaRendererError::new("The file name is none!", None))?;

    let tex_in_cpu = cpu::image_data::HalaImageData::new_with_file(path)?;

    // Create the blue noise image.
    let image = hala_gfx::HalaImage::new_2d(
      Rc::clone(&self.context.logical_device),
      hala_gfx::HalaImageUsageFlags::SAMPLED | hala_gfx::HalaImageUsageFlags::TRANSFER_DST,
      tex_in_cpu.format,
      tex_in_cpu.width,
      tex_in_cpu.height,
      1,
      1,
      hala_gfx::HalaMemoryLocation::GpuOnly,
      &format!("texture_{}.image", file_name.to_string_lossy())
    )?;
    let data = match tex_in_cpu.data_type {
      cpu::image_data::HalaImageDataType::ByteData(data) => data,
      cpu::image_data::HalaImageDataType::FloatData(data) => {
        let mut byte_data = Vec::with_capacity(data.len() * 4);
        for f in data {
          byte_data.extend_from_slice(&f.to_ne_bytes());
        }
        byte_data
      },
    };
    image.update_gpu_memory_with_buffer(
      data.as_slice(),
      hala_gfx::HalaPipelineStageFlags2::RAY_TRACING_SHADER,
      &self.transfer_staging_buffer,
      &self.transfer_command_buffers)?;
    self.blue_noise_image = Some(image);

    Ok(())
  }

  /// Set the scene to be rendered.
  /// param scene_in_cpu: The scene in the CPU.
  /// return: The result.
  pub fn set_scene(&mut self, scene_in_cpu: &cpu::HalaScene) -> Result<(), HalaRendererError> {
    // Release the old scene in the GPU.
    self.scene_in_gpu = None;

    // Upload the new scene to the GPU.
    let mut scene_in_gpu = loader::HalaSceneGPUUploader::upload(
      &self.context,
      &self.transfer_command_buffers,
      scene_in_cpu)?;
    if self.is_ray_tracing {
      loader::HalaSceneGPUUploader::additively_upload_for_ray_tracing(
        &self.context,
        &self.graphics_command_buffers,
        &self.transfer_command_buffers,
        scene_in_cpu, &mut scene_in_gpu)?;
    }
    self.scene_in_gpu = Some(scene_in_gpu);

    Ok(())
  }

  /// Set the environment map.
  /// param path: The path of the environment map.
  /// param rotation: The rotation of the environment map.
  /// return: The result.
  pub fn set_envmap<P: AsRef<Path>>(&mut self, path: P, rotation: f32) -> Result<(), HalaRendererError> {
    self.envmap = Some(crate::envmap::EnvMap::new_with_file(
      path,
      &self.context,
      &self.transfer_staging_buffer,
      &self.transfer_command_buffers,
    )?);
    self.env_rotation = rotation;

    Ok(())
  }

  /// Set the ground color.
  /// param color: The color.
  pub fn set_ground_color(&mut self, color: glam::Vec4) {
    self.env_ground_color = color;
  }

  /// Set the sky color.
  /// param color: The color.
  pub fn set_sky_color(&mut self, color: glam::Vec4) {
    self.env_sky_color = color;
  }

  /// Set the intensity of the environment.
  /// param intensity: The intensity.
  pub fn set_env_intensity(&mut self, intensity: f32) {
    self.env_intensity = intensity;
  }

  /// Commit all GPU resources.
  pub fn commit(&mut self) -> Result<(), HalaRendererError> {
    // Create texture descriptor set.
    let textures_descriptor_set = hala_gfx::HalaDescriptorSet::new_static(
      Rc::clone(&self.context.logical_device),
      Rc::clone(&self.descriptor_pool),
      hala_gfx::HalaDescriptorSetLayout::new(
        Rc::clone(&self.context.logical_device),
        &[
          ( // All textures in the scene.
            0,
            hala_gfx::HalaDescriptorType::COMBINED_IMAGE_SAMPLER,
            self.scene_in_gpu.as_ref().ok_or(HalaRendererError::new("The scene in GPU is none!", None))?
              .textures.len() as u32,
            hala_gfx::HalaShaderStageFlags::RAYGEN | hala_gfx::HalaShaderStageFlags::CALLABLE,
            hala_gfx::HalaDescriptorBindingFlags::PARTIALLY_BOUND
          ),
        ],
        "textures.descriptor_set_layout",
      )?,
      1,
      0,
      "textures.descriptor_set",
    )?;

    let textures: &Vec<_> = self.scene_in_gpu.as_ref().ok_or(HalaRendererError::new("The scene in GPU is none!", None))?.textures.as_ref();
    let samplers: &Vec<_> = self.scene_in_gpu.as_ref().ok_or(HalaRendererError::new("The scene in GPU is none!", None))?.samplers.as_ref();
    let images: &Vec<_> = self.scene_in_gpu.as_ref().ok_or(HalaRendererError::new("The scene in GPU is none!", None))?.images.as_ref();
    let mut combined_textures = Vec::new();
    for (sampler_index, image_index) in textures.iter().enumerate() {
      let sampler = samplers.get(sampler_index).ok_or(HalaRendererError::new("The sampler is none!", None))?;
      let image = images.get(*image_index as usize).ok_or(HalaRendererError::new("The image is none!", None))?;
      combined_textures.push((image, sampler));
    }
    if !combined_textures.is_empty() {
      textures_descriptor_set.update_combined_image_samplers(
        0,
        0,
        combined_textures.as_slice(),
      );
    }

    // If we have cache file at ./out/pipeline_cache.bin, we can load it.
    let pipeline_cache = if std::path::Path::new("./out/pipeline_cache.bin").exists() {
      log::debug!("Load pipeline cache from file: ./out/pipeline_cache.bin");
      hala_gfx::HalaPipelineCache::with_cache_file(
        Rc::clone(&self.context.logical_device),
        "./out/pipeline_cache.bin",
      )?
    } else {
      log::debug!("Create a new pipeline cache.");
      hala_gfx::HalaPipelineCache::new(
        Rc::clone(&self.context.logical_device),
      )?
    };

    // Create pipeline.
    let pipeline = hala_gfx::HalaRayTracingPipeline::new(
      Rc::clone(&self.context.logical_device),
      &[&self.static_descriptor_set.layout, &self.dynamic_descriptor_set.layout, &textures_descriptor_set.layout],
      self.raygen_shaders.as_slice(),
      self.miss_shaders.as_slice(),
      self.hit_shaders.as_slice(),
      self.callable_shaders.as_slice(),
      2,
      Some(&pipeline_cache),
      false,
      "main.pipeline",
    )?;

    // Save pipeline cache.
    pipeline_cache.save("./out/pipeline_cache.bin")?;

    // Create shader binding table.
    let sbt = hala_gfx::HalaShaderBindingTable::new(
      Rc::clone(&self.context.logical_device),
      self.raygen_shaders.as_slice(),
      self.miss_shaders.as_slice(),
      self.hit_shaders.as_slice(),
      self.callable_shaders.as_slice(),
      &pipeline,
      &self.transfer_staging_buffer,
      &self.transfer_command_buffers,
      "main.sbt",
    )?;

    self.textures_descriptor_set = Some(textures_descriptor_set);
    self.pipeline = Some(pipeline);
    self.sbt = Some(sbt);

    // Update static descriptor set.
    let mut static_binding_index = 0;
    self.static_descriptor_set.update_acceleration_structures(
      0,
      static_binding_index,
      &[
        self.scene_in_gpu.as_ref().ok_or(HalaRendererError::new("The scene in GPU is none!", None))?
          .tplas.as_ref().ok_or(HalaRendererError::new("The top level acceleration structure is none!", None))?
      ],
    );
    static_binding_index += 1;

    self.final_image_binding_index = static_binding_index;
    self.static_descriptor_set.update_storage_images(
      0,
      self.final_image_binding_index,
      std::slice::from_ref(self.final_image.as_ref()),
    );
    static_binding_index += 1;

    self.accum_image_binding_index = static_binding_index;
    self.static_descriptor_set.update_storage_images(
      0,
      self.accum_image_binding_index,
      std::slice::from_ref(&self.accum_image.as_ref()),
    );
    static_binding_index += 1;
    self.albedo_image_binding_index = static_binding_index;
    self.static_descriptor_set.update_storage_images(
      0,
      self.albedo_image_binding_index,
      std::slice::from_ref(&self.albedo_image.as_ref()),
    );
    static_binding_index += 1;
    self.normal_image_binding_index = static_binding_index;
    self.static_descriptor_set.update_storage_images(
      0,
      self.normal_image_binding_index,
      std::slice::from_ref(&self.normal_image.as_ref()),
    );
    static_binding_index += 1;

    let blue_noise_image = self.blue_noise_image.as_ref().ok_or(HalaRendererError::new("The blue noise image is none!", None))?;
    self.static_descriptor_set.update_sampled_images(
      0,
      static_binding_index,
      std::slice::from_ref(blue_noise_image),
    );
    static_binding_index += 1;

    if let Some(envmap) = self.envmap.as_ref() {
      self.static_descriptor_set.update_combined_image_samplers(
        0,
        static_binding_index,
        &[(&envmap.image, &envmap.sampler)],
      );
      static_binding_index += 1;
      self.static_descriptor_set.update_sampled_images(
        0,
        static_binding_index,
        &[&envmap.marginal_distribution_image, &envmap.conditional_distribution_image],
      );
      static_binding_index += 1;
      self.static_descriptor_set.update_sampler(
        0,
        static_binding_index,
        &[&envmap.distribution_sampler],
      );
      // static_binding_index += 1;
    }

    // Update dynamic descriptor set.
    for index in 0..self.context.swapchain.num_of_images {
      self.dynamic_descriptor_set.update_uniform_buffers(
        index,
        0,
        &[self.global_uniform_buffer.as_ref()],
      );
      self.dynamic_descriptor_set.update_uniform_buffers(
        index,
        1,
        &[&self.scene_in_gpu.as_ref().ok_or(HalaRendererError::new("The scene in GPU is none!", None))?.cameras],
      );
      self.dynamic_descriptor_set.update_uniform_buffers(
        index,
        2,
        &[&self.scene_in_gpu.as_ref().ok_or(HalaRendererError::new("The scene in GPU is none!", None))?.lights],
      );
      self.dynamic_descriptor_set.update_storage_buffers(
        index,
        3,
        &[&self.scene_in_gpu.as_ref().ok_or(HalaRendererError::new("The scene in GPU is none!", None))?.materials],
      );
      self.dynamic_descriptor_set.update_storage_buffers(
        index,
        4,
        &[self.scene_in_gpu.as_ref().ok_or(HalaRendererError::new("The scene in GPU is none!", None))?
          .primitives.as_ref().ok_or(HalaRendererError::new("The primitives in GPU is none!", None))?],
      );
    }

    Ok(())
  }

  /// Update the renderer.
  /// param delta_time: The delta time.
  /// return: The result.
  pub fn update(&mut self, _delta_time: f64, width: u32, height: u32) -> Result<(), HalaRendererError> {
    self.check_and_restore_device(width, height)?;

    // Statistic.
    if self.total_frames > self.context.swapchain.num_of_images as u64 {
      let gpu_time = self.context.get_gpu_frame_time()?;
      let gpu_time_nanos = gpu_time.as_nanos();
      self.total_gpu_nanoseconds += gpu_time_nanos;
      self.total_gpu_frames += 1;

      let now = std::time::Instant::now();
      let interval = now - self.last_stat_time;
      self.elapsed_time += interval;
      if self.elapsed_time > std::time::Duration::from_secs(1) {
        let elapsed_time_nanos = self.elapsed_time.as_nanos();
        log::info!(
          "FPS: {}, GPU Time: {:.4}ms, CPU Time: {:.4}ms, Total Frames: {}",
          self.total_gpu_frames * elapsed_time_nanos as u64 / 1000000000,
          self.total_gpu_nanoseconds as f64 / self.total_gpu_frames as f64 / 1000000.0,
          elapsed_time_nanos as f64 / self.total_gpu_frames as f64 / 1000000.0,
          self.total_frames + 1,
        );
        self.total_gpu_nanoseconds = 0;
        self.total_gpu_frames = 0;
        self.elapsed_time -= std::time::Duration::from_secs(1);
      }
      self.last_stat_time = now;
    }
    self.total_frames += 1;

    // Skip the update if the total frames is greater than the max frames.
    if self.total_frames > self.max_frames {
      return Ok(());
    }

    // Update global uniform buffer.
    let (use_hdri, env_total_sum, env_map_width, env_map_height) = match self.envmap.as_ref() {
      Some(envmap) => (true, envmap.total_luminance, envmap.image.extent.width, envmap.image.extent.height),
      None => (false, 0f32, 0, 0),
    };
    let num_of_lights = if let Some(scene_in_gpu) = self.scene_in_gpu.as_ref() {
      scene_in_gpu.light_data.len() as u32
    } else {
      0
    };
    self.global_uniform_buffer.update_memory(&[HalaGlobalUniform {
      ground_color: self.env_ground_color,
      sky_color: self.env_sky_color,
      resolution: glam::Vec2::new(self.width as f32, self.height as f32),
      max_depth: self.max_depth,
      rr_depth: self.rr_depth,
      frame_index: (self.total_frames - 1) as u32,
      camera_index: 0,
      env_type: if use_hdri { HalaEnvType::MAP.to_u8() as u32 } else { HalaEnvType::SKY.to_u8() as u32 },
      env_map_width,
      env_map_height,
      env_total_sum,
      env_rotation: self.env_rotation / 360f32,
      env_intensity: self.env_intensity,
      enable_tonemap: self.enable_tonemap as u32,
      enable_aces: self.enable_aces as u32,
      use_simple_aces: self.use_simple_aces as u32,
      num_of_lights,
    }])?;

    // Update the renderer.
    self.image_index = self.context.prepare_frame()?;
    self.context.record_graphics_command_buffer(
      self.image_index,
      &self.graphics_command_buffers,
      None,
      |_, _| {
        Ok(())
      },
      Some(&self.final_image),
      |index, command_buffers| {
        let _pipline = self.pipeline.as_ref().ok_or(hala_gfx::HalaGfxError::new("The pipeline is none!", None))?;
        let _sbt = self.sbt.as_ref().ok_or(hala_gfx::HalaGfxError::new("The shader binding table is none!", None))?;

        command_buffers.bind_ray_tracing_pipeline(index, _pipline);
        command_buffers.bind_ray_tracing_descriptor_sets(
          index,
          _pipline,
          0,
          &[
            self.static_descriptor_set.as_ref(),
            self.dynamic_descriptor_set.as_ref(),
            self.textures_descriptor_set.as_ref().ok_or(hala_gfx::HalaGfxError::new("The textures descriptor set is none!", None))?,
          ],
          &[],
        );
        command_buffers.trace_rays(
          index,
          _sbt,
          self.width,
          self.height,
          1,
        );

        Ok(true)
      },
    )?;

    Ok(())
  }

  /// Render the renderer.
  /// return: The result.
  pub fn render(&mut self) -> Result<(), HalaRendererError> {
    // Skip the rendering and wait to reset the device on the next frame update.
    if self.is_device_lost {
      return Ok(());
    }

    // Skip the update if the total frames is greater than the max frames.
    if self.total_frames > self.max_frames {
      return Ok(());
    }

    // Render the renderer.
    match self.context.submit_and_present_frame(self.image_index, &self.graphics_command_buffers) {
      Ok(_) => (),
      Err(err) => {
        if err.is_device_lost() {
          log::warn!("The device is lost!");
          self.is_device_lost = true;
        } else {
          return Err(err.into());
        }
      }
    }

    Ok(())
  }

  /// Wait the renderer idle.
  /// return: The result.
  pub fn wait_idle(&self) -> Result<(), HalaRendererError> {
    self.context.logical_device.borrow().wait_idle()?;

    Ok(())
  }

  /// Save the images to the file.
  /// param path: The output path of the image.
  /// return: The result.
  pub fn save_images<P: AsRef<Path>>(&self, path: P) -> Result<(), HalaRendererError> {
    if self.is_device_lost {
      // Skip the saving and wait to reset the device on the next frame update.
      log::warn!("The device is lost! Please wait to reset the device and try again.");
      return Ok(());
    }

    let path = path.as_ref();
    let filename = path.file_stem().ok_or(HalaRendererError::new("The file name is none!", None))?;
    let color_image_path = path.with_file_name(format!("{}_color.pfm", filename.to_string_lossy()));
    let albedo_image_path = path.with_file_name(format!("{}_albedo.pfm", filename.to_string_lossy()));
    let normal_image_path = path.with_file_name(format!("{}_normal.pfm", filename.to_string_lossy()));

    let save_image_2_file = |image: &hala_gfx::HalaImage, path: &Path, is_color: bool| -> Result<(), HalaRendererError> {
      let mut pixels = vec![0f32; 4 * self.width as usize * self.height as usize];

      self.wait_idle()?;
      self.context.logical_device.borrow().transfer_execute_and_submit(
        &self.transfer_command_buffers,
        0,
        |_logical_device, command_buffers, index| {
          command_buffers.copy_image_2_buffer(
            index,
            image,
            hala_gfx::HalaImageLayout::GENERAL,
            &self.host_accessible_buffer);
        },
        0)?;
      self.host_accessible_buffer.download_memory(pixels.as_mut_slice())?;

      if is_color {
        let luminance = |c: glam::Vec3| -> f32 {
          0.212671 * c.x + 0.715160 * c.y + 0.072169 * c.z
        };
        let rrt_odt_fit = |v: glam::Vec3| -> glam::Vec3 {
          let a = v * (v + 0.0245786) - 0.000090537;
          let b = v * (0.983729 * v + 0.432951) + 0.238081;
          a / b
        };
        let aces_fitted = |color: glam::Vec3| -> glam::Vec3 {
          const ACES_INPUT_MATRIX: glam::Mat3 = glam::Mat3::from_cols(
            glam::Vec3::new(0.59719, 0.07600, 0.02840),
            glam::Vec3::new(0.35458, 0.90834, 0.13383),
            glam::Vec3::new(0.04823, 0.01566, 0.83777)
          );
          const ACES_OUTPUT_MATRIX: glam::Mat3 = glam::Mat3::from_cols(
            glam::Vec3::new(1.60475, -0.10208, -0.00327),
            glam::Vec3::new(-0.53108, 1.10813, -0.07276),
            glam::Vec3::new(-0.07367, -0.00605, 1.07602)
          );
          let mut color = ACES_INPUT_MATRIX * color;
          color = rrt_odt_fit(color);
          color = ACES_OUTPUT_MATRIX * color;
          color = color.clamp(glam::Vec3::ZERO, glam::Vec3::ONE);
          color
        };
        let aces = |c: glam::Vec3| -> glam::Vec3 {
          const A: f32 = 2.51;
          const B: f32 = 0.03;
          const Y: f32 = 2.43;
          const D: f32 = 0.59;
          const E: f32 = 0.14;

          let r = (c * (A * c + B)) / (c * (Y * c + D) + E);
          r.clamp(glam::Vec3::ZERO, glam::Vec3::ONE)
        };
        let tonemap = |c: glam::Vec3, limit: f32| -> glam::Vec3 {
          c * 1.0 / (1.0 + luminance(c) / limit)
        };

        // Convert the color image to sRGB.
        for pixel in pixels.chunks_exact_mut(4) {
          let color = glam::Vec3::new(pixel[0], pixel[1], pixel[2]);
          let color = if self.enable_tonemap {
            if self.enable_aces {
              if self.use_simple_aces {
                aces(color)
              } else {
                aces_fitted(color)
              }
            } else {
              tonemap(color, 1.5)
            }
          } else {
            color
          };
          pixel[0] = color.x;
          pixel[1] = color.y;
          pixel[2] = color.z;
        }
      }

      let image_file = std::fs::File::create(path)
        .map_err(|err| HalaRendererError::new(&format!("Failed to create the image file: {:?}", path), Some(Box::new(err))))?;
      let mut writer = std::io::BufWriter::new(image_file);
      writeln!(&mut writer, "PF\n{} {}\n-1.0", image.extent.width, image.extent.height)
        .map_err(|err| HalaRendererError::new(&format!("Failed to write the image file: {:?}", path), Some(Box::new(err))))?;
      for row in pixels.chunks_exact(4 * image.extent.width as usize).rev() {
        for pixel in row.chunks_exact(4) {
          writer.write_all(&pixel[0].to_le_bytes())
            .map_err(|err| HalaRendererError::new(&format!("Failed to write the image file: {:?}", path), Some(Box::new(err))))?;
          writer.write_all(&pixel[1].to_le_bytes())
            .map_err(|err| HalaRendererError::new(&format!("Failed to write the image file: {:?}", path), Some(Box::new(err))))?;
          writer.write_all(&pixel[2].to_le_bytes())
            .map_err(|err| HalaRendererError::new(&format!("Failed to write the image file: {:?}", path), Some(Box::new(err))))?;
        }
      }
      writer.flush()
        .map_err(|err| HalaRendererError::new(&format!("Failed to flush the image file: {:?}", path), Some(Box::new(err))))?;

      Ok(())
    };

    log::debug!("Begin to save the color image...");
    save_image_2_file(&self.accum_image, &color_image_path, true)?;
    log::info!("Save the color image to file: {:?}", color_image_path);

    log::debug!("Begin to save the albedo image...");
    save_image_2_file(&self.albedo_image, &albedo_image_path, false)?;
    log::info!("Save the albedo image to file: {:?}", albedo_image_path);

    log::debug!("Begin to save the normal image...");
    save_image_2_file(&self.normal_image, &normal_image_path, false)?;
    log::info!("Save the normal image to file: {:?}", normal_image_path);

    Ok(())
  }

  /// Check and restore the device.
  /// param width: The width of the swapchain.
  /// param height: The height of the swapchain.
  /// return: The result.
  fn check_and_restore_device(&mut self, width: u32, height: u32) -> Result<(), HalaRendererError> {
    if self.is_device_lost {
      self.context.reset_swapchain(width, height)?;

      self.width = width;
      self.height = height;
      unsafe {
        std::mem::ManuallyDrop::drop(&mut self.host_accessible_buffer);
        std::mem::ManuallyDrop::drop(&mut self.normal_image);
        std::mem::ManuallyDrop::drop(&mut self.albedo_image);
        std::mem::ManuallyDrop::drop(&mut self.accum_image);
        std::mem::ManuallyDrop::drop(&mut self.final_image);
      }
      let (
        final_image,
        accum_image,
        albedo_image,
        normal_image,
        host_accessible_buffer,
      ) = Self::create_storage_images(&self.context)?;
      self.final_image = std::mem::ManuallyDrop::new(final_image);
      self.accum_image = std::mem::ManuallyDrop::new(accum_image);
      self.albedo_image = std::mem::ManuallyDrop::new(albedo_image);
      self.normal_image = std::mem::ManuallyDrop::new(normal_image);
      self.host_accessible_buffer = std::mem::ManuallyDrop::new(host_accessible_buffer);

      self.static_descriptor_set.update_storage_images(
        0,
        self.final_image_binding_index,
        std::slice::from_ref(self.final_image.as_ref()),
      );
      self.static_descriptor_set.update_storage_images(
        0,
        self.accum_image_binding_index,
        std::slice::from_ref(&self.accum_image.as_ref()),
      );
      self.static_descriptor_set.update_storage_images(
        0,
        self.albedo_image_binding_index,
        std::slice::from_ref(&self.albedo_image.as_ref()),
      );
      self.static_descriptor_set.update_storage_images(
        0,
        self.normal_image_binding_index,
        std::slice::from_ref(&self.normal_image.as_ref()),
      );

      self.total_frames = 0;
      self.last_stat_time = std::time::Instant::now();
      self.elapsed_time = std::time::Duration::from_secs(0);
      self.total_gpu_nanoseconds = 0;
      self.total_gpu_frames = 0;

      self.is_device_lost = false;
    }

    Ok(())
  }

}