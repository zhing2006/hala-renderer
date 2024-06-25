use std::rc::Rc;
use std::cell::RefCell;

use hala_gfx::{
  HalaGPURequirements,
  HalaContext,
};

use crate::error::HalaRendererError;

#[repr(C, align(4))]
#[derive(Debug, Clone, Copy)]
pub struct HalaGlobalUniform {
  // The view matrix.
  pub v_mtx: glam::Mat4,
  // The projection matrix.
  pub p_mtx: glam::Mat4,
  // The view-projection matrix.
  pub vp_mtx: glam::Mat4,
}

#[repr(C, align(4))]
#[derive(Debug, Clone, Copy)]
pub struct HalaObjectUniform {
  // The model matrix.
  pub m_mtx: glam::Mat4,
  // The inverse model matrix.
  pub i_m_mtx: glam::Mat4,
  // The model-view matrix.
  pub mv_mtx: glam::Mat4,
  // The transposed model-view matrix.
  pub t_mv_mtx: glam::Mat4,
  // The inverse transposed model-view matrix.
  pub it_mv_mtx: glam::Mat4,
  // The model-view-projection matrix.
  pub mvp_mtx: glam::Mat4,
}

/// The renderer.
pub struct HalaRenderer {

  pub name: String,
  pub width: u32,
  pub height: u32,

  pub context: std::mem::ManuallyDrop<Rc<RefCell<HalaContext>>>,

  pub(crate) graphics_command_buffers: std::mem::ManuallyDrop<hala_gfx::HalaCommandBufferSet>,
  pub(crate) transfer_command_buffers: std::mem::ManuallyDrop<hala_gfx::HalaCommandBufferSet>,
  pub(crate) transfer_staging_buffer: std::mem::ManuallyDrop<hala_gfx::HalaBuffer>,

  pub(crate) descriptor_pool: std::mem::ManuallyDrop<Rc<RefCell<hala_gfx::HalaDescriptorPool>>>,
  pub(crate) static_descriptor_set: std::mem::ManuallyDrop<hala_gfx::HalaDescriptorSet>,
  pub(crate) dynamic_descriptor_set: std::mem::ManuallyDrop<hala_gfx::HalaDescriptorSet>,
  pub(crate) global_uniform_buffer: std::mem::ManuallyDrop<hala_gfx::HalaBuffer>,
  pub(crate) object_uniform_buffer: std::mem::ManuallyDrop<hala_gfx::HalaBuffer>,

  // Render data.
  pub image_index: usize,
  pub is_device_lost: bool,

  // Statistic.
  pub total_frames: u64,
  pub last_stat_time: std::time::Instant,
  pub elapsed_time: std::time::Duration,
  pub total_gpu_nanoseconds: u128,
  pub total_gpu_frames: u64,

}

/// The Drop implementation of the renderer.
impl Drop for HalaRenderer {

  fn drop(&mut self) {
    unsafe {
      std::mem::ManuallyDrop::drop(&mut self.object_uniform_buffer);
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
  /// return: The renderer.
  #[allow(clippy::too_many_arguments)]
  pub fn new(
    name: &str,
    gpu_req: &HalaGPURequirements,
    window: &winit::window::Window,
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
          ( // Global uniform buffer.
            0,
            hala_gfx::HalaDescriptorType::UNIFORM_BUFFER,
            1,
            hala_gfx::HalaShaderStageFlags::VERTEX | hala_gfx::HalaShaderStageFlags::FRAGMENT,
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
          ( // Object uniform buffer.
            0,
            hala_gfx::HalaDescriptorType::UNIFORM_BUFFER,
            1,
            hala_gfx::HalaShaderStageFlags::VERTEX | hala_gfx::HalaShaderStageFlags::FRAGMENT,
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
    let global_uniform_buffer = hala_gfx::HalaBuffer::new(
      Rc::clone(&context.logical_device),
      std::mem::size_of::<HalaGlobalUniform>() as u64,
      hala_gfx::HalaBufferUsageFlags::UNIFORM_BUFFER,
      hala_gfx::HalaMemoryLocation::CpuToGpu,
      "global.uniform_buffer",
    )?;

    // Create object uniform buffer.
    let object_uniform_buffer = hala_gfx::HalaBuffer::new(
      Rc::clone(&context.logical_device),
      std::mem::size_of::<HalaObjectUniform>() as u64,
      hala_gfx::HalaBufferUsageFlags::UNIFORM_BUFFER,
      hala_gfx::HalaMemoryLocation::CpuToGpu,
      "object.uniform_buffer",
    )?;

    // Return the renderer.
    log::debug!("A HalaRenderer \"{}\"[{} x {}] is created.", name, width, height);
    Ok(Self {
      name: name.to_string(),
      width,
      height,
      context: std::mem::ManuallyDrop::new(Rc::new(RefCell::new(context))),

      graphics_command_buffers: std::mem::ManuallyDrop::new(graphics_command_buffers),
      transfer_command_buffers: std::mem::ManuallyDrop::new(transfer_command_buffers),
      transfer_staging_buffer: std::mem::ManuallyDrop::new(transfer_staging_buffer),
      descriptor_pool: std::mem::ManuallyDrop::new(descriptor_pool),
      static_descriptor_set: std::mem::ManuallyDrop::new(static_descriptor_set),
      dynamic_descriptor_set: std::mem::ManuallyDrop::new(dynamic_descriptor_set),
      global_uniform_buffer: std::mem::ManuallyDrop::new(global_uniform_buffer),
      object_uniform_buffer: std::mem::ManuallyDrop::new(object_uniform_buffer),

      image_index: 0,
      is_device_lost: false,

      total_frames: 0,
      last_stat_time: std::time::Instant::now(),
      elapsed_time: std::time::Duration::from_secs(0),
      total_gpu_nanoseconds: 0,
      total_gpu_frames: 0,
    })
  }

  /// Commit all GPU resources.
  pub fn commit(&mut self) -> Result<(), HalaRendererError> {
    Ok(())
  }

  /// Wait the renderer idle.
  /// return: The result.
  pub fn wait_idle(&self) -> Result<(), HalaRendererError> {
    let context = self.context.borrow();
    context.logical_device.borrow().wait_idle()?;

    Ok(())
  }

  /// Check and restore the device.
  /// param width: The width of the swapchain.
  /// param height: The height of the swapchain.
  /// return: The result.
  fn check_and_restore_device(&mut self, width: u32, height: u32) -> Result<(), HalaRendererError> {
    let mut context = self.context.borrow_mut();

    if self.is_device_lost {
      context.reset_swapchain(width, height)?;

      self.width = width;
      self.height = height;

      // TODO: Update resources.

      self.total_frames = 0;
      self.last_stat_time = std::time::Instant::now();
      self.elapsed_time = std::time::Duration::from_secs(0);
      self.total_gpu_nanoseconds = 0;
      self.total_gpu_frames = 0;

      self.is_device_lost = false;
    }

    Ok(())
  }

  /// Update the renderer.
  /// param delta_time: The delta time.
  /// param width: The width of the window.
  /// param height: The height of the window.
  /// param ui_fn: The draw UI function.
  /// return: The result.
  pub fn update<F>(&mut self, _delta_time: f64, width: u32, height: u32, ui_fn: F) -> Result<(), HalaRendererError>
    where F: FnOnce(usize, &hala_gfx::HalaCommandBufferSet) -> Result<(), hala_gfx::HalaGfxError>
  {
    self.check_and_restore_device(width, height)?;
    let context = self.context.borrow();

    // Statistic.
    if self.total_frames > context.swapchain.num_of_images as u64 {
      let gpu_time = context.get_gpu_frame_time()?;
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

    // TODO: Update resources.

    // Update the renderer.
    self.image_index = context.prepare_frame()?;
    context.record_graphics_command_buffer(
      self.image_index,
      &self.graphics_command_buffers,
      Some(([25.0 / 255.0, 118.0 / 255.0, 210.0 / 255.0, 1.0], 1.0, 0)),
      |index, command_buffers| {
        ui_fn(index, command_buffers)?;

        Ok(())
      },
      None,
      |_, _| {
        Ok(false)
      },
    )?;

    Ok(())
  }

  /// Render the renderer.
  /// return: The result.
  pub fn render(&mut self) -> Result<(), HalaRendererError> {
    let mut context = self.context.borrow_mut();

    // Skip the rendering and wait to reset the device on the next frame update.
    if self.is_device_lost {
      return Ok(());
    }

    // Render the renderer.
    match context.submit_and_present_frame(self.image_index, &self.graphics_command_buffers) {
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

}