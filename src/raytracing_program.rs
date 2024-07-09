use std::rc::Rc;
use std::cell::RefCell;

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
use crate::shader_cache::HalaShaderCache;

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
  pub raygen_shader_file_paths: Vec<String>,
  pub miss_shader_file_paths: Vec<String>,
  pub hit_shader_file_paths: Vec<HalaRayTracingHitShaderDesc>,
  pub callable_shader_file_paths: Vec<String>,
  pub push_constant_size: u32,
  pub ray_recursion_depth: u32,
}

/// The raytracing program.
pub struct HalaRayTracingProgram {
  pub raygen_shaders: Vec<Rc<RefCell<HalaShader>>>,
  pub miss_shaders: Vec<Rc<RefCell<HalaShader>>>,
  pub hit_shaders: Vec<(Option<Rc<RefCell<HalaShader>>>, Option<Rc<RefCell<HalaShader>>>, Option<Rc<RefCell<HalaShader>>>)>,
  pub callable_shaders: Vec<Rc<RefCell<HalaShader>>>,
  pub pipeline: HalaRayTracingPipeline,
  pub sbt: HalaShaderBindingTable,
}

/// The implementation of the raytracing program.
impl HalaRayTracingProgram {

  /// Create a new raytracing program.
  /// param logical_device: The logical device.
  /// param descriptor_set_layouts: The descriptor set layouts.
  /// param desc: The description.
  /// param pipeline_cache: The pipeline cache.
  /// param staging_buffer: The staging buffer.
  /// param transfer_command_buffers: The transfer command buffers.
  /// param debug_name: The debug name.
  /// return: The raytracing program.
  pub fn new<DSL>(
    logical_device: Rc<RefCell<HalaLogicalDevice>>,
    descriptor_set_layouts: &[DSL],
    desc: &HalaRayTracingProgramDesc,
    pipeline_cache: Option<&HalaPipelineCache>,
    staging_buffer: &HalaBuffer,
    transfer_command_buffers: &HalaCommandBufferSet,
    debug_name: &str,
  ) -> Result<Self, HalaRendererError>
  where
    DSL: AsRef<HalaDescriptorSetLayout>,
  {
    let mut raygen_shaders = Vec::new();
    for raygen_shader_file_path in &desc.raygen_shader_file_paths {
      let shader = HalaShaderCache::get_instance().borrow_mut().load(
        logical_device.clone(),
        raygen_shader_file_path,
        HalaShaderStageFlags::RAYGEN,
        HalaRayTracingShaderGroupType::GENERAL,
        &format!("{}.rgen.spv", debug_name),
      )?;
      raygen_shaders.push(shader);
    }
    let mut miss_shaders = Vec::new();
    for miss_shader_file_path in &desc.miss_shader_file_paths {
      let shader = HalaShaderCache::get_instance().borrow_mut().load(
        logical_device.clone(),
        miss_shader_file_path,
        HalaShaderStageFlags::MISS,
        HalaRayTracingShaderGroupType::GENERAL,
        &format!("{}.miss.spv", debug_name),
      )?;
      miss_shaders.push(shader);
    }
    let mut hit_shaders = Vec::new();
    for hit_shader_desc in &desc.hit_shader_file_paths {
      let closest_hit_shader = match hit_shader_desc.closest_hit_shader_file_path {
        Some(ref closest_hit_shader_file_path) => {
          Some(HalaShaderCache::get_instance().borrow_mut().load(
            logical_device.clone(),
            closest_hit_shader_file_path,
            HalaShaderStageFlags::CLOSEST_HIT,
            HalaRayTracingShaderGroupType::TRIANGLES_HIT_GROUP,
            &format!("{}.chit.spv", debug_name),
          )?)
        },
        None => None,
      };
      let any_hit_shader = match hit_shader_desc.any_hit_shader_file_path {
        Some(ref any_hit_shader_file_path) => {
          Some(HalaShaderCache::get_instance().borrow_mut().load(
            logical_device.clone(),
            any_hit_shader_file_path,
            HalaShaderStageFlags::ANY_HIT,
            HalaRayTracingShaderGroupType::TRIANGLES_HIT_GROUP,
            &format!("{}.ahit.spv", debug_name),
          )?)
        },
        None => None,
      };
      let intersection_shader = match hit_shader_desc.intersection_shader_file_path {
        Some(ref intersection_shader_file_path) => {
          Some(HalaShaderCache::get_instance().borrow_mut().load(
            logical_device.clone(),
            intersection_shader_file_path,
            HalaShaderStageFlags::INTERSECTION,
            HalaRayTracingShaderGroupType::PROCEDURAL_HIT_GROUP,
            &format!("{}.isec.spv", debug_name),
          )?)
        },
        None => None,
      };
      hit_shaders.push((closest_hit_shader, any_hit_shader, intersection_shader));
    }
    let mut callable_shaders = Vec::new();
    for callable_shader_file_path in &desc.callable_shader_file_paths {
      let shader = HalaShaderCache::get_instance().borrow_mut().load(
        logical_device.clone(),
        callable_shader_file_path,
        HalaShaderStageFlags::CALLABLE,
        HalaRayTracingShaderGroupType::GENERAL,
        &format!("{}.call.spv", debug_name),
      )?;
      callable_shaders.push(shader);
    }

    let (pipeline, sbt) = {
      let mut shader_stage = HalaShaderStageFlags::default();
      if !raygen_shaders.is_empty() {
        shader_stage |= HalaShaderStageFlags::RAYGEN;
      }
      if !miss_shaders.is_empty() {
        shader_stage |= HalaShaderStageFlags::MISS;
      }
      if !hit_shaders.is_empty() {
        shader_stage |= HalaShaderStageFlags::CLOSEST_HIT | HalaShaderStageFlags::ANY_HIT | HalaShaderStageFlags::INTERSECTION;
      }
      if !callable_shaders.is_empty() {
        shader_stage |= HalaShaderStageFlags::CALLABLE;
      }
      let raygen_shaders = raygen_shaders.iter().map(|shader| { shader.borrow() }).collect::<Vec<_>>();
      let miss_shaders = miss_shaders.iter().map(|shader| { shader.borrow() }).collect::<Vec<_>>();
      let hit_shaders = hit_shaders.iter().map(|(closest_hit_shader, any_hit_shader, intersection_shader)| {
        (
          closest_hit_shader.as_ref().map_or(None, |shader| Some(shader.borrow())),
          any_hit_shader.as_ref().map_or(None, |shader| Some(shader.borrow())),
          intersection_shader.as_ref().map_or(None, |shader| Some(shader.borrow())),
        )
      }).collect::<Vec<_>>();
      let callable_shaders = callable_shaders.iter().map(|shader| { shader.borrow() }).collect::<Vec<_>>();
      let r = raygen_shaders.iter().map(|shader| shader.as_ref()).collect::<Vec<_>>();
      let m = miss_shaders.iter().map(|shader| shader.as_ref()).collect::<Vec<_>>();
      let h = hit_shaders.iter().map(|(closest_hit_shader, any_hit_shader, intersection_shader)| {
        (
          closest_hit_shader.as_ref().map_or(None, |shader| Some(shader.as_ref())),
          any_hit_shader.as_ref().map_or(None, |shader| Some(shader.as_ref())),
          intersection_shader.as_ref().map_or(None, |shader| Some(shader.as_ref())),
        )
      }).collect::<Vec<_>>();
      let c = callable_shaders.iter().map(|shader| shader.as_ref()).collect::<Vec<_>>();
      let push_constant_ranges = if desc.push_constant_size > 0 {
        &[
          hala_gfx::HalaPushConstantRange {
            stage_flags: shader_stage,
            offset: 0,
            size: desc.push_constant_size,
          },
        ]
      } else {
        &[] as &[hala_gfx::HalaPushConstantRange]
      };
      let pipeline = HalaRayTracingPipeline::new(
        logical_device.clone(),
        descriptor_set_layouts,
        push_constant_ranges,
        r.as_slice(),
        m.as_slice(),
        h.as_slice(),
        c.as_slice(),
        desc.ray_recursion_depth,
        pipeline_cache,
        false,
        &format!("{}.raytracing_pipeline", debug_name),
      )?;
      let sbt = HalaShaderBindingTable::new(
        logical_device.clone(),
        r.as_slice(),
        m.as_slice(),
        h.as_slice(),
        c.as_slice(),
        &pipeline,
        staging_buffer,
        transfer_command_buffers,
        &format!("{}.sbt", debug_name),
      )?;

      (pipeline, sbt)
    };

    Ok(Self {
      raygen_shaders,
      miss_shaders,
      hit_shaders,
      callable_shaders,
      pipeline,
      sbt,
    })
  }

  /// Get the raytracing pipeline.
  /// return: The raytracing pipeline.
  pub fn get_pso(&self) -> &HalaRayTracingPipeline {
    &self.pipeline
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
    if !descriptor_sets.is_empty() {
      command_buffers.bind_ray_tracing_descriptor_sets(
        index,
        &self.pipeline,
        0,
        descriptor_sets,
        &[],
      );
    }
  }

  /// Push constants.
  /// param index: The index of the command buffer.
  /// param command_buffers: The command buffers.
  /// param offset: The offset.
  /// param data: The data.
  pub fn push_constants(&self, index: usize, command_buffers: &HalaCommandBufferSet, offset: u32, data: &[u8]) {
    let mut shader_stage = HalaShaderStageFlags::default();
    if !self.raygen_shaders.is_empty() {
      shader_stage |= HalaShaderStageFlags::RAYGEN;
    }
    if !self.miss_shaders.is_empty() {
      shader_stage |= HalaShaderStageFlags::MISS;
    }
    if !self.hit_shaders.is_empty() {
      shader_stage |= HalaShaderStageFlags::CLOSEST_HIT | HalaShaderStageFlags::ANY_HIT | HalaShaderStageFlags::INTERSECTION;
    }
    if !self.callable_shaders.is_empty() {
      shader_stage |= HalaShaderStageFlags::CALLABLE;
    }
    command_buffers.push_constants(index, self.pipeline.layout, shader_stage, offset, data);
  }

  /// Push constants f32.
  /// param index: The index of the command buffer.
  /// param command_buffers: The command buffers.
  /// param offset: The offset.
  /// param data: The data.
  pub fn push_constants_f32(&self, index: usize, command_buffers: &HalaCommandBufferSet, offset: u32, data: &[f32]) {
    let mut shader_stage = HalaShaderStageFlags::default();
    if !self.raygen_shaders.is_empty() {
      shader_stage |= HalaShaderStageFlags::RAYGEN;
    }
    if !self.miss_shaders.is_empty() {
      shader_stage |= HalaShaderStageFlags::MISS;
    }
    if !self.hit_shaders.is_empty() {
      shader_stage |= HalaShaderStageFlags::CLOSEST_HIT | HalaShaderStageFlags::ANY_HIT | HalaShaderStageFlags::INTERSECTION;
    }
    if !self.callable_shaders.is_empty() {
      shader_stage |= HalaShaderStageFlags::CALLABLE;
    }
    command_buffers.push_constants_f32(index, self.pipeline.layout, shader_stage, offset, data);
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

  /// Trace rays with indirect.
  /// param index: The index.
  /// param command_buffers: The command buffers.
  /// param indirect_device_address: The indirect device address.
  pub fn trace_rays_indirect(&self, index: usize, command_buffers: &HalaCommandBufferSet, indirect_device_address: u64) {
    command_buffers.trace_rays_indirect(index, &self.sbt, indirect_device_address);
  }

}