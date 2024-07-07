use std::rc::Rc;
use std::cell::RefCell;
use std::path::Path;

use serde::{Deserialize, Serialize};

use hala_gfx::{
  HalaCommandBufferSet,
  HalaRayTracingPipeline,
  HalaDescriptorSetLayout,
  HalaDescriptorSet,
  HalaLogicalDevice,
  HalaPipelineCache,
  HalaShader,
  HalaShaderStageFlags,
  HalaRayTracingShaderGroupType,
  HalaBuffer,
  HalaShaderBindingTable,
};

use crate::error::HalaRendererError;

/// The raytracing hit shader description.
#[derive(Serialize, Deserialize)]
pub struct HalaRayTracingHitShaderDesc {
  pub closest_hit_shader_file_path: Option<String>,
  pub any_hit_shader_file_path: Option<String>,
  pub intersection_shader_file_path: Option<String>,
}

/// The raytracing program description.
#[derive(Serialize, Deserialize)]
pub struct HalaRayTracingProgramDesc {
  pub name: String,
  pub raygen_shader_file_paths: Vec<String>,
  pub miss_shader_file_paths: Vec<String>,
  pub hit_shader_file_paths: Vec<HalaRayTracingHitShaderDesc>,
  pub callable_shader_file_paths: Vec<String>,
  pub ray_recursion_depth: u32,
}

/// The raytracing program.
pub struct HalaRayTracingProgram {
  pub raygen_shaders: Vec<HalaShader>,
  pub miss_shaders: Vec<HalaShader>,
  pub hit_shaders: Vec<(Option<HalaShader>, Option<HalaShader>, Option<HalaShader>)>,
  pub callable_shaders: Vec<HalaShader>,
  pub pipeline: HalaRayTracingPipeline,
  pub sbt: HalaShaderBindingTable,
}

/// The implementation of the raytracing program.
impl HalaRayTracingProgram {

  pub fn new<P, DSL>(
    logical_device: Rc<RefCell<HalaLogicalDevice>>,
    shader_dir: P,
    descriptor_set_layouts: &[DSL],
    desc: &HalaRayTracingProgramDesc,
    pipeline_cache: Option<&HalaPipelineCache>,
    staging_buffer: &HalaBuffer,
    transfer_command_buffers: &HalaCommandBufferSet,
  ) -> Result<Self, HalaRendererError>
  where
    P: AsRef<Path>,
    DSL: AsRef<HalaDescriptorSetLayout>,
  {
    let mut raygen_shaders = Vec::new();
    for raygen_shader_file_path in &desc.raygen_shader_file_paths {
      let shader = HalaShader::with_file(
        logical_device.clone(),
        &format!("{}/{}", shader_dir.as_ref().to_string_lossy(), raygen_shader_file_path),
        HalaShaderStageFlags::RAYGEN,
        HalaRayTracingShaderGroupType::GENERAL,
        &format!("{}.rgen.spv", desc.name),
      )?;
      raygen_shaders.push(shader);
    }
    let mut miss_shaders = Vec::new();
    for miss_shader_file_path in &desc.miss_shader_file_paths {
      let shader = HalaShader::with_file(
        logical_device.clone(),
        &format!("{}/{}", shader_dir.as_ref().to_string_lossy(), miss_shader_file_path),
        HalaShaderStageFlags::MISS,
        HalaRayTracingShaderGroupType::GENERAL,
        &format!("{}.miss.spv", desc.name),
      )?;
      miss_shaders.push(shader);
    }
    let mut hit_shaders = Vec::new();
    for hit_shader_desc in &desc.hit_shader_file_paths {
      let closest_hit_shader = match hit_shader_desc.closest_hit_shader_file_path {
        Some(ref closest_hit_shader_file_path) => {
          Some(HalaShader::with_file(
            logical_device.clone(),
            &format!("{}/{}", shader_dir.as_ref().to_string_lossy(), closest_hit_shader_file_path),
            HalaShaderStageFlags::CLOSEST_HIT,
            HalaRayTracingShaderGroupType::TRIANGLES_HIT_GROUP,
            &format!("{}.chit.spv", desc.name),
          )?)
        },
        None => None,
      };
      let any_hit_shader = match hit_shader_desc.any_hit_shader_file_path {
        Some(ref any_hit_shader_file_path) => {
          Some(HalaShader::with_file(
            logical_device.clone(),
            &format!("{}/{}", shader_dir.as_ref().to_string_lossy(), any_hit_shader_file_path),
            HalaShaderStageFlags::ANY_HIT,
            HalaRayTracingShaderGroupType::TRIANGLES_HIT_GROUP,
            &format!("{}.ahit.spv", desc.name),
          )?)
        },
        None => None,
      };
      let intersection_shader = match hit_shader_desc.intersection_shader_file_path {
        Some(ref intersection_shader_file_path) => {
          Some(HalaShader::with_file(
            logical_device.clone(),
            &format!("{}/{}", shader_dir.as_ref().to_string_lossy(), intersection_shader_file_path),
            HalaShaderStageFlags::INTERSECTION,
            HalaRayTracingShaderGroupType::PROCEDURAL_HIT_GROUP,
            &format!("{}.isec.spv", desc.name),
          )?)
        },
        None => None,
      };
      hit_shaders.push((closest_hit_shader, any_hit_shader, intersection_shader));
    }
    let mut callable_shaders = Vec::new();
    for callable_shader_file_path in &desc.callable_shader_file_paths {
      let shader = HalaShader::with_file(
        logical_device.clone(),
        &format!("{}/{}", shader_dir.as_ref().to_string_lossy(), callable_shader_file_path),
        HalaShaderStageFlags::CALLABLE,
        HalaRayTracingShaderGroupType::GENERAL,
        &format!("{}.call.spv", desc.name),
      )?;
      callable_shaders.push(shader);
    }
    let pipeline = HalaRayTracingPipeline::new(
      logical_device.clone(),
      descriptor_set_layouts,
      &raygen_shaders,
      &miss_shaders,
      &hit_shaders,
      &callable_shaders,
      desc.ray_recursion_depth,
      pipeline_cache,
      false,
      &format!("{}.raytracing_pipeline", desc.name),
    )?;
    let sbt = HalaShaderBindingTable::new(
      logical_device.clone(),
      &raygen_shaders,
      &miss_shaders,
      &hit_shaders,
      &callable_shaders,
      &pipeline,
      staging_buffer,
      transfer_command_buffers,
      &format!("{}.sbt", desc.name),
    )?;

    Ok(Self {
      raygen_shaders,
      miss_shaders,
      hit_shaders,
      callable_shaders,
      pipeline,
      sbt,
    })
  }

  /// Bind the raytracing program.
  /// param index: The index.
  /// param command_buffers: The command buffers.
  /// param descriptor_sets: The descriptor sets.
  pub fn bind<DS>(&self, index: usize, command_buffers: &HalaCommandBufferSet, descriptor_sets: &[DS])
  where
    DS: AsRef<HalaDescriptorSet>
  {
    command_buffers.bind_ray_tracing_pipeline(index, &self.pipeline);
    command_buffers.bind_ray_tracing_descriptor_sets(
      index,
      &self.pipeline,
      0,
      descriptor_sets,
      &[],
    );
  }

  /// Trace rays.
  /// param index: The index.
  /// param command_buffers: The command buffers.
  /// param width: The width.
  /// param height: The height.
  /// param depth: The depth.
  pub fn trace_rays(&self, index: usize, command_buffers: &HalaCommandBufferSet, width: u32, height: u32, depth: u32) {
    command_buffers.trace_rays(index, &self.sbt, width, height, depth);
  }

}