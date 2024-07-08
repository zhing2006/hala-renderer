use std::rc::Rc;
use std::cell::RefCell;

use serde::{Serialize, Deserialize};

use hala_gfx::{
  HalaCommandBufferSet,
  HalaComputePipeline,
  HalaDescriptorSet,
  HalaDescriptorSetLayout,
  HalaLogicalDevice,
  HalaPipelineCache,
  HalaRayTracingShaderGroupType,
  HalaShader,
  HalaShaderStageFlags,
  HalaBuffer,
};

use crate::error::HalaRendererError;
use crate::shader_cache::HalaShaderCache;

/// The compute program description.
#[derive(Serialize, Deserialize)]
pub struct HalaComputeProgramDesc {
  pub shader_file_path: String,
}

/// The compute program.
pub struct HalaComputeProgram {
  #[allow(dead_code)]
  shader: Rc<RefCell<HalaShader>>,
  pipeline: HalaComputePipeline,
}

/// The compute program implementation.
impl HalaComputeProgram {

  /// Create a new compute program.
  /// param logical_device: The logical device.
  /// param descriptor_set_layouts: The descriptor set layouts.
  /// param desc: The compute program description.
  /// param pipeline_cache: The pipeline cache.
  /// param name: The debug name.
  /// return: The compute program.
  pub fn new<DSL>(
    logical_device: Rc<RefCell<HalaLogicalDevice>>,
    descriptor_set_layouts: &[DSL],
    desc: &HalaComputeProgramDesc,
    pipeline_cache: Option<&HalaPipelineCache>,
    debug_name: &str,
  ) -> Result<Self, HalaRendererError>
  where
    DSL: AsRef<HalaDescriptorSetLayout>,
  {
    let shader_cache = HalaShaderCache::get_instance();
    let mut shader_cache = shader_cache.borrow_mut();

    let shader = shader_cache.load(
      logical_device.clone(),
      &desc.shader_file_path,
      HalaShaderStageFlags::COMPUTE,
      HalaRayTracingShaderGroupType::GENERAL,
      &format!("{}.comp.spv", debug_name),
    )?;
    let pipeline = HalaComputePipeline::new(
      logical_device.clone(),
      descriptor_set_layouts,
      shader.borrow().as_ref(),
      pipeline_cache,
      &format!("{}.compute_pipeline", debug_name),
    )?;

    Ok(Self { shader, pipeline })
  }

  /// Get the compute pipeline.
  /// return: The compute pipeline.
  pub fn get_pso(&self) -> &HalaComputePipeline {
    &self.pipeline
  }

  /// Bind the compute program.
  /// param index: The index.
  /// param command_buffers: The command buffers.
  /// param descriptor_sets: The descriptor sets.
  pub fn bind<DS>(&self, index: usize, command_buffers: &HalaCommandBufferSet, descriptor_sets: &[DS])
  where
    DS: AsRef<HalaDescriptorSet>
  {
    command_buffers.bind_compute_pipeline(index, &self.pipeline);
    if !descriptor_sets.is_empty() {
      command_buffers.bind_compute_descriptor_sets(
        index,
        &self.pipeline,
        0,
        descriptor_sets,
        &[],
      );
    }
  }

  /// Dispatch the compute program.
  /// param index: The index.
  /// param command_buffer_set: The command buffer set.
  /// param group_count_x: The group count x.
  /// param group_count_y: The group count y.
  /// param group_count_z: The group count z.
  pub fn dispatch(
    &self,
    index: usize,
    command_buffer_set: &HalaCommandBufferSet,
    group_count_x: u32,
    group_count_y: u32,
    group_count_z: u32,
  ) {
    command_buffer_set.dispatch(index, group_count_x, group_count_y, group_count_z);
  }

  /// Dispatch the compute program with indirect.
  /// param index: The index.
  /// param command_buffer_set: The command buffer set.
  /// param buffer: The buffer.
  /// param offset: The offset.
  pub fn dispatch_indirect(
    &self,
    index: usize,
    command_buffer_set: &HalaCommandBufferSet,
    buffer: &HalaBuffer,
    offset: u64,
  ) {
    command_buffer_set.dispatch_indirect(index, buffer, offset);
  }

}