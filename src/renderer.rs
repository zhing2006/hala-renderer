use std::rc::Rc;
use std::cell::RefCell;

use anyhow::Result;

use hala_gfx::HalaContext;

use crate::error::HalaRendererError;

/// The renderer informaton.
pub struct HalaRendererInfo {
  pub name: String,
  pub width: u32,
  pub height: u32,
}

/// The renderer information implementation.
impl HalaRendererInfo {

  /// Create a new renderer information.
  /// param name: The renderer name.
  /// param width: The renderer width.
  /// param height: The renderer height.
  /// return: The renderer information.
  pub fn new(name: &str, width: u32, height: u32) -> Self {
    Self {
      name: name.to_string(),
      width,
      height,
    }
  }

}

/// The renderer resources.
pub struct HalaRendererResources {
  pub context: std::mem::ManuallyDrop<Rc<RefCell<HalaContext>>>,

  pub graphics_command_buffers: std::mem::ManuallyDrop<hala_gfx::HalaCommandBufferSet>,
  pub compute_command_buffers: std::mem::ManuallyDrop<hala_gfx::HalaCommandBufferSet>,
  pub transfer_command_buffers: std::mem::ManuallyDrop<hala_gfx::HalaCommandBufferSet>,
  pub transfer_staging_buffer: std::mem::ManuallyDrop<hala_gfx::HalaBuffer>,

  pub descriptor_pool: std::mem::ManuallyDrop<Rc<RefCell<hala_gfx::HalaDescriptorPool>>>,
}

/// The renderer resources drop implementation.
impl Drop for HalaRendererResources {

  fn drop(&mut self) {
    unsafe {
      std::mem::ManuallyDrop::drop(&mut self.graphics_command_buffers);
      std::mem::ManuallyDrop::drop(&mut self.compute_command_buffers);
      std::mem::ManuallyDrop::drop(&mut self.transfer_command_buffers);
      std::mem::ManuallyDrop::drop(&mut self.transfer_staging_buffer);
      std::mem::ManuallyDrop::drop(&mut self.descriptor_pool);
      std::mem::ManuallyDrop::drop(&mut self.context);
    }
  }

}

/// The renderer resources implementation.
impl HalaRendererResources {

  pub fn new(
    name: &str,
    gpu_req: &hala_gfx::HalaGPURequirements,
    window: &winit::window::Window,
    descriptor_sizes: &[(hala_gfx::HalaDescriptorType, usize)],
  ) -> Result<Self, HalaRendererError> {
    let context = HalaContext::new(name, gpu_req, window)?;

    // Craete command buffers.
    let graphics_command_buffers = hala_gfx::HalaCommandBufferSet::new(
      Rc::clone(&context.logical_device),
      Rc::clone(&context.pools),
      hala_gfx::HalaCommandBufferType::GRAPHICS,
      hala_gfx::HalaCommandBufferLevel::PRIMARY,
      context.swapchain.num_of_images,
      "main_graphics.cmd_buffer",
    )?;
    let compute_command_buffers = hala_gfx::HalaCommandBufferSet::new(
      Rc::clone(&context.logical_device),
      Rc::clone(&context.pools),
      hala_gfx::HalaCommandBufferType::COMPUTE,
      hala_gfx::HalaCommandBufferLevel::PRIMARY,
      context.swapchain.num_of_images,
      "main_compute.cmd_buffer",
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
      descriptor_sizes,
      512,
      "main.descriptor_pool"
    )?));

    Ok(
      Self {
        context: std::mem::ManuallyDrop::new(Rc::new(RefCell::new(context))),

        graphics_command_buffers: std::mem::ManuallyDrop::new(graphics_command_buffers),
        compute_command_buffers: std::mem::ManuallyDrop::new(compute_command_buffers),
        transfer_command_buffers: std::mem::ManuallyDrop::new(transfer_command_buffers),
        transfer_staging_buffer: std::mem::ManuallyDrop::new(transfer_staging_buffer),

        descriptor_pool: std::mem::ManuallyDrop::new(descriptor_pool),
      }
    )
  }

}

/// The renderer data.
#[derive(Default)]
pub struct HalaRendererData {
  pub image_index: usize,
  pub is_device_lost: bool,
}

/// The renderer data implementation.
impl HalaRendererData {

  /// Create a new renderer data.
  /// return: The renderer data.
  pub fn new() -> Self {
    Self::default()
  }

}


/// The renderer statistics.
pub struct HalaRendererStatistics {
  pub total_frames: u64,
  pub last_stat_time: std::time::Instant,
  pub elapsed_time: std::time::Duration,
  pub total_gpu_nanoseconds: u128,
  pub total_gpu_frames: u64,
}

/// The renderer statistics default implementation.
impl Default for HalaRendererStatistics {

  fn default() -> Self {
    Self {
      total_frames: 0,
      last_stat_time: std::time::Instant::now(),
      elapsed_time: std::time::Duration::new(0, 0),
      total_gpu_nanoseconds: 0,
      total_gpu_frames: 0,
    }
  }

}

/// The renderer statistics implementation.
impl HalaRendererStatistics {

  /// Create a new renderer statistics.
  /// return: The renderer statistics.
  pub fn new() -> Self {
    Self::default()
  }

  /// Reset the renderer statistics.
  pub fn reset(&mut self) {
    self.total_frames = 0;
    self.last_stat_time = std::time::Instant::now();
    self.elapsed_time = std::time::Duration::new(0, 0);
    self.total_gpu_nanoseconds = 0;
    self.total_gpu_frames = 0;
  }

  /// Set the GPU time.
  /// param gpu_time: The GPU time.
  pub fn set_gpu_time(&mut self, gpu_time: &std::time::Duration) {
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

  /// Increase the total frames.
  pub fn inc_total_frames(&mut self) {
    self.total_frames += 1;
  }

}

/// The renderer trait.
pub trait HalaRendererTrait {

  fn info(&self) -> &HalaRendererInfo;
  fn info_mut(&mut self) -> &mut HalaRendererInfo;
  fn resources(&self) -> &HalaRendererResources;
  fn resources_mut(&mut self) -> &mut HalaRendererResources;
  fn data(&self) -> &HalaRendererData;
  fn data_mut(&mut self) -> &mut HalaRendererData;
  fn statistics(&self) -> &HalaRendererStatistics;
  fn statistics_mut(&mut self) -> &mut HalaRendererStatistics;

  fn get_descriptor_sizes() -> Vec<(hala_gfx::HalaDescriptorType, usize)>;

  /// Commit all GPU resources.
  /// return: The result.
  fn commit(&mut self) -> Result<(), HalaRendererError>;

  /// Check and restore the device.
  /// param width: The width of the swapchain.
  /// param height: The height of the swapchain.
  /// return: The result.
  fn check_and_restore_device(&mut self, width: u32, height: u32) -> Result<(), HalaRendererError> {
    self.check_and_restore_swapchain(width, height)
  }
  fn check_and_restore_swapchain(&mut self, width: u32, height: u32) -> Result<(), HalaRendererError> {
    if self.data().is_device_lost {
      self.resources().context.borrow_mut().reset_swapchain(width, height)?;

      self.info_mut().width = width;
      self.info_mut().height = height;

      self.statistics_mut().reset();

      self.data_mut().is_device_lost = false;
    }

    Ok(())
  }

  /// Wait the renderer idle.
  /// return: The result.
  fn wait_idle(&self) -> Result<(), HalaRendererError> {
    let context = self.resources().context.borrow();
    context.logical_device.borrow().wait_idle()?;

    Ok(())
  }

  /// Update the renderer.
  /// param delta_time: The delta time.
  /// param width: The width of the window.
  /// param height: The height of the window.
  /// param ui_fn: The draw UI function.
  /// return: The result.
  fn update<F>(&mut self, _delta_time: f64, width: u32, height: u32, ui_fn: F) -> Result<(), HalaRendererError>
    where F: FnOnce(usize, &hala_gfx::HalaCommandBufferSet) -> Result<(), hala_gfx::HalaGfxError>;
  fn pre_update(&mut self, width: u32, height: u32) -> Result<(), HalaRendererError> {
    self.check_and_restore_device(width, height)?;

    // Get a new image index.
    let image_index = self.resources().context.borrow().prepare_frame()?;
    self.data_mut().image_index = image_index;

    // This image is finished. So we can get statistic data safely.
    if self.statistics_mut().total_frames > self.resources().context.borrow().swapchain.num_of_images as u64 {
      let gpu_time = self.resources().context.borrow().get_gpu_frame_time(self.data().image_index)?;
      self.statistics_mut().set_gpu_time(&gpu_time);
    }
    self.statistics_mut().inc_total_frames();

    Ok(())
  }

  /// Render the renderer.
  /// return: The result.
  fn render(&mut self) -> Result<(), HalaRendererError> {
    // Skip the rendering and wait to reset the device on the next frame update.
    if self.data().is_device_lost {
      return Ok(());
    }

    let result = {
      let mut context = self.resources().context.borrow_mut();

      // Render the renderer.
      context.submit_and_present_frame(self.data().image_index, &self.resources().graphics_command_buffers)
    };

    match result {
      Ok(_) => (),
      Err(err) => {
        if err.is_device_lost() {
          log::warn!("The device is lost!");
          self.data_mut().is_device_lost = true;
        } else {
          return Err(err.into());
        }
      }
    }

    Ok(())
  }

}