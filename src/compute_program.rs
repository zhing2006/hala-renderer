use std::rc::Rc;
use std::cell::RefCell;

use serde::{Serialize, Deserialize};

use hala_gfx::{
  HalaCommandBufferSet,
  HalaComputePipeline,
  HalaDescriptorSet,
  HalaDescriptorSetLayout,
  HalaDescriptorType,
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
  pub push_constant_size: u32,
  #[serde(default)]
  pub bindings: Vec<HalaDescriptorType>,
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
    let push_constant_ranges = if desc.push_constant_size > 0 {
      &[
        hala_gfx::HalaPushConstantRange {
          stage_flags: HalaShaderStageFlags::COMPUTE,
          offset: 0,
          size: desc.push_constant_size,
        },
      ]
    } else {
      &[] as &[hala_gfx::HalaPushConstantRange]
    };
    let pipeline = HalaComputePipeline::new(
      logical_device.clone(),
      descriptor_set_layouts,
      push_constant_ranges,
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

  /// Push constants.
  /// param index: The index of the command buffer.
  /// param command_buffers: The command buffers.
  /// param offset: The offset.
  /// param data: The data.
  pub fn push_constants(&self, index: usize, command_buffers: &HalaCommandBufferSet, offset: u32, data: &[u8]) {
    let shader_stage = HalaShaderStageFlags::COMPUTE;
    command_buffers.push_constants(index, self.pipeline.layout, shader_stage, offset, data);
  }

  /// Push constants f32.
  /// param index: The index of the command buffer.
  /// param command_buffers: The command buffers.
  /// param offset: The offset.
  /// param data: The data.
  pub fn push_constants_f32(&self, index: usize, command_buffers: &HalaCommandBufferSet, offset: u32, data: &[f32]) {
    let shader_stage = HalaShaderStageFlags::COMPUTE;
    command_buffers.push_constants_f32(index, self.pipeline.layout, shader_stage, offset, data);
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