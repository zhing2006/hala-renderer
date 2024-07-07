use std::rc::Rc;
use std::cell::RefCell;
use std::path::Path;

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

/// The compute program description.
#[derive(Serialize, Deserialize)]
pub struct HalaComputeProgramDesc {
  pub name: String,
  pub shader_file_path: String,
}

/// The compute program.
pub struct HalaComputeProgram {
  #[allow(dead_code)]
  shader: HalaShader,
  pipeline: HalaComputePipeline,
}

/// The compute program implementation.
impl HalaComputeProgram {

  /// Create a new compute program.
  /// param logical_device: The logical device.
  /// param descriptor_set_layouts: The descriptor set layouts.
  /// param desc: The compute program description.
  /// param pipeline_cache: The pipeline cache.
  /// param debug_name: The debug name.
  /// return: The compute program.
  pub fn new<P, DSL>(
    logical_device: Rc<RefCell<HalaLogicalDevice>>,
    shader_dir: P,
    descriptor_set_layouts: &[DSL],
    desc: &HalaComputeProgramDesc,
    pipeline_cache: Option<&HalaPipelineCache>,
  ) -> Result<Self, HalaRendererError>
  where
    P: AsRef<Path>,
    DSL: AsRef<HalaDescriptorSetLayout>,
  {
    let shader = HalaShader::with_file(
      logical_device.clone(),
      &format!("{}/{}", shader_dir.as_ref().to_string_lossy(), desc.shader_file_path),
      HalaShaderStageFlags::COMPUTE,
      HalaRayTracingShaderGroupType::GENERAL,
      &format!("{}.comp.spv", desc.name),
    )?;
    let pipeline = HalaComputePipeline::new(
      logical_device.clone(),
      descriptor_set_layouts,
      &shader,
      pipeline_cache,
      &format!("{}.compute_pipeline", desc.name),
    )?;

    Ok(Self { shader, pipeline })
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
    command_buffers.bind_compute_descriptor_sets(
      index,
      &self.pipeline,
      0,
      descriptor_sets,
      &[],
    );
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