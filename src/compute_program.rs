use std::rc::Rc;
use std::cell::RefCell;
use std::path::Path;
use std::fmt;

use serde::{Serialize, Deserialize, Deserializer};
use serde::de::{self, Unexpected, Visitor};

use hala_gfx::{
  HalaCommandBufferSet,
  HalaComputePipeline,
  HalaDescriptorSetLayout,
  HalaDescriptorSet,
  HalaLogicalDevice,
  HalaPipelineCache,
  HalaShader,
  HalaShaderStageFlags,
  HalaRayTracingShaderGroupType,
};

use crate::error::HalaRendererError;

/// The compute program description.
#[derive(Serialize, Deserialize)]
pub struct HalaComputeProgramDesc {
  pub name: String,
  pub shader_file_path: String,
  #[serde(deserialize_with = "shader_stage_flags_deserialize")]
  pub shader_stage: HalaShaderStageFlags,
  #[serde(deserialize_with = "ray_tracing_shader_group_type_deserialize")]
  pub rt_group: HalaRayTracingShaderGroupType,
}

/// Deserialize the shader stage flags.
fn shader_stage_flags_deserialize<'de, D>(deserializer: D) -> Result<HalaShaderStageFlags, D::Error>
where
  D: Deserializer<'de>,
{
  struct HalaStringToShaderStageFlagsVisitor;

  impl<'de> Visitor<'de> for HalaStringToShaderStageFlagsVisitor {
    type Value = HalaShaderStageFlags;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
      formatter.write_str("a string 'A' or 'B'")
    }

    fn visit_str<E>(self, value: &str) -> Result<HalaShaderStageFlags, E>
    where
      E: de::Error,
    {
      match value {
        "VERTEX" => Ok(HalaShaderStageFlags::VERTEX),
        "vertex" => Ok(HalaShaderStageFlags::VERTEX),
        "TESSELLATION_CONTROL" => Ok(HalaShaderStageFlags::TESSELLATION_CONTROL),
        "tessellation_control" => Ok(HalaShaderStageFlags::TESSELLATION_CONTROL),
        "TESSELLATION_EVALUATION" => Ok(HalaShaderStageFlags::TESSELLATION_EVALUATION),
        "tessellation_evaluation" => Ok(HalaShaderStageFlags::TESSELLATION_EVALUATION),
        "GEOMETRY" => Ok(HalaShaderStageFlags::GEOMETRY),
        "geometry" => Ok(HalaShaderStageFlags::GEOMETRY),
        "FRAGMENT" => Ok(HalaShaderStageFlags::FRAGMENT),
        "fragment" => Ok(HalaShaderStageFlags::FRAGMENT),
        "COMPUTE" => Ok(HalaShaderStageFlags::COMPUTE),
        "compute" => Ok(HalaShaderStageFlags::COMPUTE),
        "ALL_GRAPHICS" => Ok(HalaShaderStageFlags::ALL_GRAPHICS),
        "all_graphics" => Ok(HalaShaderStageFlags::ALL_GRAPHICS),
        "ALL" => Ok(HalaShaderStageFlags::ALL),
        "all" => Ok(HalaShaderStageFlags::ALL),
        "RAYGEN" => Ok(HalaShaderStageFlags::RAYGEN),
        "raygen" => Ok(HalaShaderStageFlags::RAYGEN),
        "ANY_HIT" => Ok(HalaShaderStageFlags::ANY_HIT),
        "any_hit" => Ok(HalaShaderStageFlags::ANY_HIT),
        "CLOSEST_HIT" => Ok(HalaShaderStageFlags::CLOSEST_HIT),
        "closest_hit" => Ok(HalaShaderStageFlags::CLOSEST_HIT),
        "MISS" => Ok(HalaShaderStageFlags::MISS),
        "miss" => Ok(HalaShaderStageFlags::MISS),
        "INTERSECTION" => Ok(HalaShaderStageFlags::INTERSECTION),
        "intersection" => Ok(HalaShaderStageFlags::INTERSECTION),
        "CALLABLE" => Ok(HalaShaderStageFlags::CALLABLE),
        "callable" => Ok(HalaShaderStageFlags::CALLABLE),
        "TASK" => Ok(HalaShaderStageFlags::TASK),
        "task" => Ok(HalaShaderStageFlags::TASK),
        "MESH" => Ok(HalaShaderStageFlags::MESH),
        "mesh" => Ok(HalaShaderStageFlags::MESH),
        _ => Err(E::invalid_value(Unexpected::Str(value), &self)),
      }
    }
  }

  deserializer.deserialize_str(HalaStringToShaderStageFlagsVisitor)
}

/// Deserialize the ray tracing shader group type.
fn ray_tracing_shader_group_type_deserialize<'de, D>(
  deserializer: D,
) -> Result<HalaRayTracingShaderGroupType, D::Error>
where
  D: Deserializer<'de>,
{
  struct HalaStringToRayTracingShaderGroupTypeVisitor;

  impl<'de> Visitor<'de> for HalaStringToRayTracingShaderGroupTypeVisitor {
    type Value = HalaRayTracingShaderGroupType;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
      formatter.write_str("a string 'A' or 'B'")
    }

    fn visit_str<E>(self, value: &str) -> Result<HalaRayTracingShaderGroupType, E>
    where
      E: de::Error,
    {
      match value {
        "DEFAULT" => Ok(HalaRayTracingShaderGroupType::default()),
        "default" => Ok(HalaRayTracingShaderGroupType::default()),
        "GENERAL" => Ok(HalaRayTracingShaderGroupType::GENERAL),
        "general" => Ok(HalaRayTracingShaderGroupType::GENERAL),
        "TRIANGLES_HIT_GROUP" => Ok(HalaRayTracingShaderGroupType::TRIANGLES_HIT_GROUP),
        "triangles_hit_group" => Ok(HalaRayTracingShaderGroupType::TRIANGLES_HIT_GROUP),
        "PROCEDURAL_HIT_GROUP" => Ok(HalaRayTracingShaderGroupType::PROCEDURAL_HIT_GROUP),
        "procedural_hit_group" => Ok(HalaRayTracingShaderGroupType::PROCEDURAL_HIT_GROUP),
        _ => Err(E::invalid_value(Unexpected::Str(value), &self)),
      }
    }
  }

  deserializer.deserialize_str(HalaStringToRayTracingShaderGroupTypeVisitor)
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
      desc.shader_stage,
      desc.rt_group,
      &format!("{}.comp", desc.name),
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

  pub fn dispatch(&self,
    index: usize,
    command_buffer_set: &HalaCommandBufferSet,
    group_count_x: u32,
    group_count_y: u32,
    group_count_z: u32,
  ) {
    command_buffer_set.dispatch(index, group_count_x, group_count_y, group_count_z);
  }

}