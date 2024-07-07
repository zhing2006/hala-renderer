use std::rc::Rc;
use std::cell::RefCell;
use std::path::Path;
use std::fmt;

use serde::{Serialize, Deserialize, Serializer, Deserializer};
use serde::de::{self, Unexpected, Visitor};

use hala_gfx::{
  HalaBlendState,
  HalaBuffer,
  HalaCommandBufferSet,
  HalaDepthState,
  HalaDescriptorSet,
  HalaDescriptorSetLayout,
  HalaDynamicState,
  HalaGraphicsPipeline,
  HalaImage,
  HalaLogicalDevice,
  HalaPipelineCache,
  HalaPipelineCreateFlags,
  HalaPrimitiveTopology,
  HalaPushConstantRange,
  HalaRasterizerState,
  HalaRayTracingShaderGroupType,
  HalaShader,
  HalaShaderStageFlags,
  HalaStencilState,
  HalaSwapchain,
  HalaVertexInputAttributeDescription,
  HalaVertexInputBindingDescription,
};

use crate::error::HalaRendererError;

/// The graphics program description.
#[derive(Serialize, Deserialize)]
pub struct HalaGraphicsProgramDesc {
  pub vertex_shader_file_path: Option<String>,
  pub task_shader_file_path: Option<String>,
  pub mesh_shader_file_path: Option<String>,
  pub fragment_shader_file_path: String,
  #[serde(serialize_with = "primitive_topology_serialize", deserialize_with = "primitive_topology_deserialize")]
  pub primitive_topology: HalaPrimitiveTopology,
  pub color_blend: HalaBlendState,
  pub alpha_blend: HalaBlendState,
  pub rasterizer_info: HalaRasterizerState,
  pub depth_info: HalaDepthState,
  pub stencil_info: Option<HalaStencilState>,
}

fn primitive_topology_serialize<S>(value: &HalaPrimitiveTopology, serializer: S) -> Result<S::Ok, S::Error>
where
  S: Serializer,
{
  let s = match *value {
    HalaPrimitiveTopology::POINT_LIST => "point_list",
    HalaPrimitiveTopology::LINE_LIST => "list_list",
    HalaPrimitiveTopology::LINE_STRIP => "line_strip",
    HalaPrimitiveTopology::TRIANGLE_LIST => "triangle_list",
    HalaPrimitiveTopology::TRIANGLE_STRIP => "triangle_strip",
    HalaPrimitiveTopology::TRIANGLE_FAN => "triangle_fan",
    HalaPrimitiveTopology::LINE_LIST_WITH_ADJACENCY => "line_list_with_adjacency",
    HalaPrimitiveTopology::LINE_STRIP_WITH_ADJACENCY => "line_strip_with_adjacency",
    HalaPrimitiveTopology::TRIANGLE_LIST_WITH_ADJACENCY => "triangle_list_with_adjacency",
    HalaPrimitiveTopology::TRIANGLE_STRIP_WITH_ADJACENCY => "triangle_strip_with_adjacency",
    HalaPrimitiveTopology::PATCH_LIST => "patch_list",
    _ => "default",
  };

  serializer.serialize_str(s)
}

/// Deserialize the primitive topology.
fn primitive_topology_deserialize<'de, D>(deserializer: D) -> Result<HalaPrimitiveTopology, D::Error>
where
  D: Deserializer<'de>,
{
  struct HalaPrimitiveTopologyVisitor;

  impl<'de> Visitor<'de> for HalaPrimitiveTopologyVisitor {
    type Value = HalaPrimitiveTopology;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
      formatter.write_str("a string of primitive topology")
    }

    fn visit_str<E>(self, value: &str) -> Result<HalaPrimitiveTopology, E>
    where
      E: de::Error,
    {
      match value {
        "POINT_LIST" => Ok(HalaPrimitiveTopology::POINT_LIST),
        "point_list" => Ok(HalaPrimitiveTopology::POINT_LIST),
        "LINE_LIST" => Ok(HalaPrimitiveTopology::LINE_LIST),
        "line_list" => Ok(HalaPrimitiveTopology::LINE_LIST),
        "LINE_STRIP" => Ok(HalaPrimitiveTopology::LINE_STRIP),
        "line_strip" => Ok(HalaPrimitiveTopology::LINE_STRIP),
        "TRIANGLE_LIST" => Ok(HalaPrimitiveTopology::TRIANGLE_LIST),
        "triangle_list" => Ok(HalaPrimitiveTopology::TRIANGLE_LIST),
        "TRIANGLE_STRIP" => Ok(HalaPrimitiveTopology::TRIANGLE_STRIP),
        "triangle_strip" => Ok(HalaPrimitiveTopology::TRIANGLE_STRIP),
        "TRIANGLE_FAN" => Ok(HalaPrimitiveTopology::TRIANGLE_FAN),
        "triangle_fan" => Ok(HalaPrimitiveTopology::TRIANGLE_FAN),
        "LINE_LIST_WITH_ADJACENCY" => Ok(HalaPrimitiveTopology::LINE_LIST_WITH_ADJACENCY),
        "line_list_with_adjacency" => Ok(HalaPrimitiveTopology::LINE_LIST_WITH_ADJACENCY),
        "LINE_STRIP_WITH_ADJACENCY" => Ok(HalaPrimitiveTopology::LINE_STRIP_WITH_ADJACENCY),
        "line_strip_with_adjacency" => Ok(HalaPrimitiveTopology::LINE_STRIP_WITH_ADJACENCY),
        "TRIANGLE_LIST_WITH_ADJACENCY" => Ok(HalaPrimitiveTopology::TRIANGLE_LIST_WITH_ADJACENCY),
        "triangle_list_with_adjacency" => Ok(HalaPrimitiveTopology::TRIANGLE_LIST_WITH_ADJACENCY),
        "TRIANGLE_STRIP_WITH_ADJACENCY" => Ok(HalaPrimitiveTopology::TRIANGLE_STRIP_WITH_ADJACENCY),
        "triangle_strip_with_adjacency" => Ok(HalaPrimitiveTopology::TRIANGLE_STRIP_WITH_ADJACENCY),
        "PATCH_LIST" => Ok(HalaPrimitiveTopology::PATCH_LIST),
        "patch_list" => Ok(HalaPrimitiveTopology::PATCH_LIST),
        "default" => Ok(HalaPrimitiveTopology::default()),
                _ => return Err(de::Error::invalid_value(Unexpected::Str(value), &"a primitive topology")),
      }
    }
  }

  deserializer.deserialize_str(HalaPrimitiveTopologyVisitor)
}

/// The graphics program.
pub struct HalaGraphicsProgram {
  #[allow(dead_code)]
  vertex_shader: Option<HalaShader>,
  #[allow(dead_code)]
  task_shader: Option<HalaShader>,
  #[allow(dead_code)]
  mesh_shader: Option<HalaShader>,
  #[allow(dead_code)]
  fragment_shader: HalaShader,
  pipeline: HalaGraphicsPipeline,
}

/// The implementation of the graphics program.
#[allow(clippy::too_many_arguments)]
impl HalaGraphicsProgram {

  /// Create a new graphics program.
  /// param logical_device: The logical device.
  /// param shader_dir: The shader directory.
  /// param swapchain: The swapchain.
  /// param descriptor_set_layouts: The descriptor set layouts.
  /// param flags: The pipeline create flags.
  /// param vertex_attribute_descriptions: The vertex attribute descriptions.
  /// param vertex_binding_descriptions: The vertex binding descriptions.
  /// param push_constant_ranges: The push constant ranges.
  /// param dynamic_states: The dynamic states.
  /// param desc: The graphics program description.
  /// param pipeline_cache: The pipeline cache.
  /// param debug_name: The debug name.
  /// return: The result of the graphics program.
  pub fn new<P, DSL, VIAD, VIBD, PCR>(
    logical_device: Rc<RefCell<HalaLogicalDevice>>,
    shader_dir: P,
    swapchain: &HalaSwapchain,
    descriptor_set_layouts: &[DSL],
    flags: HalaPipelineCreateFlags,
    vertex_attribute_descriptions: &[VIAD],
    vertex_binding_descriptions: &[VIBD],
    push_constant_ranges: &[PCR],
    dynamic_states: &[HalaDynamicState],
    desc: &HalaGraphicsProgramDesc,
    pipeline_cache: Option<&HalaPipelineCache>,
    debug_name: &str,
  ) -> Result<Self, HalaRendererError>
  where
    P: AsRef<Path>,
    DSL: AsRef<HalaDescriptorSetLayout>,
    VIAD: AsRef<HalaVertexInputAttributeDescription>,
    VIBD: AsRef<HalaVertexInputBindingDescription>,
    PCR: AsRef<HalaPushConstantRange>,
  {
    let vertex_shader = if let Some(ref vertex_shader_file_path) = desc.vertex_shader_file_path {
      Some(HalaShader::with_file(
        logical_device.clone(),
        &format!("{}/{}", shader_dir.as_ref().to_string_lossy(), vertex_shader_file_path),
        HalaShaderStageFlags::VERTEX,
        HalaRayTracingShaderGroupType::GENERAL,
        &format!("{}.vert.spv", debug_name),
      )?)
    } else {
      None
    };

    let task_shader = if let Some(ref task_shader_file_path) = desc.task_shader_file_path {
      Some(HalaShader::with_file(
        logical_device.clone(),
        &format!("{}/{}", shader_dir.as_ref().to_string_lossy(), task_shader_file_path),
        HalaShaderStageFlags::TASK,
        HalaRayTracingShaderGroupType::GENERAL,
        &format!("{}.task.spv", debug_name),
      )?)
    } else {
      None
    };

    let mesh_shader = if let Some(ref mesh_shader_file_path) = desc.mesh_shader_file_path {
      Some(HalaShader::with_file(
        logical_device.clone(),
        &format!("{}/{}", shader_dir.as_ref().to_string_lossy(), mesh_shader_file_path),
        HalaShaderStageFlags::MESH,
        HalaRayTracingShaderGroupType::GENERAL,
        &format!("{}.mesh.spv", debug_name),
      )?)
    } else {
      None
    };

    let fragment_shader = HalaShader::with_file(
      logical_device.clone(),
      &format!("{}/{}", shader_dir.as_ref().to_string_lossy(), desc.fragment_shader_file_path),
      HalaShaderStageFlags::FRAGMENT,
      HalaRayTracingShaderGroupType::GENERAL,
      &format!("{}.frag.spv", debug_name),
    )?;

    let mut shaders = Vec::new();
    if let Some(ref vertex_shader) = vertex_shader {
      shaders.push(vertex_shader);
    }
    if let Some(ref task_shader) = task_shader {
      shaders.push(task_shader);
    }
    if let Some(ref mesh_shader) = mesh_shader {
      shaders.push(mesh_shader);
    }
    shaders.push(&fragment_shader);
    let pipeline = HalaGraphicsPipeline::new(
      logical_device.clone(),
      swapchain,
      descriptor_set_layouts,
      flags,
      vertex_attribute_descriptions,
      vertex_binding_descriptions,
      push_constant_ranges,
      desc.primitive_topology,
      &desc.color_blend,
      &desc.alpha_blend,
      &desc.rasterizer_info,
      &desc.depth_info,
      desc.stencil_info.as_ref(),
      shaders.as_slice(),
      dynamic_states,
      pipeline_cache,
      &format!("{}.graphics_pipeline", debug_name),
    )?;

    Ok(Self {
      vertex_shader,
      task_shader,
      mesh_shader,
      fragment_shader,
      pipeline,
    })
  }

  /// Create a new graphics program with custom render target.
  /// param logical_device: The logical device.
  /// param shader_dir: The shader directory.
  /// param color_images: The color images.
  /// param depth_image: The depth image.
  /// param descriptor_set_layouts: The descriptor set layouts.
  /// param flags: The pipeline create flags.
  /// param vertex_attribute_descriptions: The vertex attribute descriptions.
  /// param vertex_binding_descriptions: The vertex binding descriptions.
  /// param push_constant_ranges: The push constant ranges.
  /// param dynamic_states: The dynamic states.
  /// param desc: The graphics program description.
  /// param pipeline_cache: The pipeline cache.
  /// param debug_name: The debug name.
  /// return: The result of the graphics program.
  pub fn with_rt<P, T, DSL, VIAD, VIBD, PCR>(
    logical_device: Rc<RefCell<HalaLogicalDevice>>,
    shader_dir: P,
    color_images: &[T],
    depth_image: Option<&T>,
    descriptor_set_layouts: &[DSL],
    flags: HalaPipelineCreateFlags,
    vertex_attribute_descriptions: &[VIAD],
    vertex_binding_descriptions: &[VIBD],
    push_constant_ranges: &[PCR],
    dynamic_states: &[HalaDynamicState],
    desc: &HalaGraphicsProgramDesc,
    pipeline_cache: Option<&HalaPipelineCache>,
    debug_name: &str,
  ) -> Result<Self, HalaRendererError>
  where
    P: AsRef<Path>,
    T: AsRef<HalaImage>,
    DSL: AsRef<HalaDescriptorSetLayout>,
    VIAD: AsRef<HalaVertexInputAttributeDescription>,
    VIBD: AsRef<HalaVertexInputBindingDescription>,
    PCR: AsRef<HalaPushConstantRange>,
  {
    let vertex_shader = if let Some(ref vertex_shader_file_path) = desc.vertex_shader_file_path {
      Some(HalaShader::with_file(
        logical_device.clone(),
        &format!("{}/{}", shader_dir.as_ref().to_string_lossy(), vertex_shader_file_path),
        HalaShaderStageFlags::VERTEX,
        HalaRayTracingShaderGroupType::GENERAL,
        &format!("{}.vert.spv", debug_name),
      )?)
    } else {
      None
    };

    let task_shader = if let Some(ref task_shader_file_path) = desc.task_shader_file_path {
      Some(HalaShader::with_file(
        logical_device.clone(),
        &format!("{}/{}", shader_dir.as_ref().to_string_lossy(), task_shader_file_path),
        HalaShaderStageFlags::TASK,
        HalaRayTracingShaderGroupType::GENERAL,
        &format!("{}.task.spv", debug_name),
      )?)
    } else {
      None
    };

    let mesh_shader = if let Some(ref mesh_shader_file_path) = desc.mesh_shader_file_path {
      Some(HalaShader::with_file(
        logical_device.clone(),
        &format!("{}/{}", shader_dir.as_ref().to_string_lossy(), mesh_shader_file_path),
        HalaShaderStageFlags::MESH,
        HalaRayTracingShaderGroupType::GENERAL,
        &format!("{}.mesh.spv", debug_name),
      )?)
    } else {
      None
    };

    let fragment_shader = HalaShader::with_file(
      logical_device.clone(),
      &format!("{}/{}", shader_dir.as_ref().to_string_lossy(), desc.fragment_shader_file_path),
      HalaShaderStageFlags::FRAGMENT,
      HalaRayTracingShaderGroupType::GENERAL,
      &format!("{}.frag.spv", debug_name),
    )?;

    let mut shaders = Vec::new();
    if let Some(ref vertex_shader) = vertex_shader {
      shaders.push(vertex_shader);
    }
    if let Some(ref task_shader) = task_shader {
      shaders.push(task_shader);
    }
    if let Some(ref mesh_shader) = mesh_shader {
      shaders.push(mesh_shader);
    }
    shaders.push(&fragment_shader);
    let pipeline = HalaGraphicsPipeline::with_rt(
      logical_device.clone(),
      color_images,
      depth_image,
      descriptor_set_layouts,
      flags,
      vertex_attribute_descriptions,
      vertex_binding_descriptions,
      push_constant_ranges,
      desc.primitive_topology,
      &desc.color_blend,
      &desc.alpha_blend,
      &desc.rasterizer_info,
      &desc.depth_info,
      desc.stencil_info.as_ref(),
      shaders.as_slice(),
      dynamic_states,
      pipeline_cache,
      &format!("{}.graphics_pipeline", debug_name),
    )?;

    Ok(Self {
      vertex_shader,
      task_shader,
      mesh_shader,
      fragment_shader,
      pipeline,
    })
  }

  /// Get the graphics pipeline.
  /// return: The graphics pipeline.
  pub fn get_pso(&self) -> &HalaGraphicsPipeline {
    &self.pipeline
  }

  /// Bind the graphics program.
  /// param index: The index of the command buffer.
  /// param command_buffers: The command buffers.
  /// param descriptor_sets: The descriptor sets.
  pub fn bind<DS>(&self, index: usize, command_buffers: &HalaCommandBufferSet, descriptor_sets: &[DS])
  where
    DS: AsRef<HalaDescriptorSet>
  {
    command_buffers.bind_graphics_pipeline(index, &self.pipeline);
    if !descriptor_sets.is_empty() {
      command_buffers.bind_graphics_descriptor_sets(
        index,
        &self.pipeline,
        0,
        descriptor_sets,
        &[],
      );
    }
  }

  pub fn push_constants(&self, index: usize, command_buffers: &HalaCommandBufferSet, offset: u32, data: &[u8]) {
    let mut shader_stage = HalaShaderStageFlags::FRAGMENT;
    if self.vertex_shader.is_some() {
      shader_stage |= HalaShaderStageFlags::VERTEX;
    }
    if self.task_shader.is_some() {
      shader_stage |= HalaShaderStageFlags::TASK;
    }
    if self.mesh_shader.is_some() {
      shader_stage |= HalaShaderStageFlags::MESH;
    }
    command_buffers.push_constants(index, &self.pipeline, shader_stage, offset, data);
  }

  pub fn push_constants_f32(&self, index: usize, command_buffers: &HalaCommandBufferSet, offset: u32, data: &[f32]) {
    let mut shader_stage = HalaShaderStageFlags::FRAGMENT;
    if self.vertex_shader.is_some() {
      shader_stage |= HalaShaderStageFlags::VERTEX;
    }
    if self.task_shader.is_some() {
      shader_stage |= HalaShaderStageFlags::TASK;
    }
    if self.mesh_shader.is_some() {
      shader_stage |= HalaShaderStageFlags::MESH;
    }
    command_buffers.push_constants_f32(index, &self.pipeline, shader_stage, offset, data);
  }

  /// Draw.
  /// param index: The index of the command buffer.
  /// param command_buffers: The command buffers.
  /// param vertex_count: The vertex count.
  /// param instance_count: The instance count.
  /// param first_vertex: The first vertex.
  /// param first_instance: The first instance.
  pub fn draw(
    &self,
    index: usize,
    command_buffers: &HalaCommandBufferSet,
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32)
  {
    command_buffers.draw(index, vertex_count, instance_count, first_vertex, first_instance);
  }

  /// Draw indexed.
  /// param index: The index of the command buffer.
  /// param command_buffers: The command buffers.
  /// param index_count: The index count.
  /// param instance_count: The instance count.
  /// param first_index: The first index.
  /// param vertex_offset: The vertex offset.
  /// param first_instance: The first instance.
  pub fn draw_indexed(
    &self,
    index: usize,
    command_buffers: &HalaCommandBufferSet,
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    vertex_offset: i32,
    first_instance: u32)
  {
    command_buffers.draw_indexed(index, index_count, instance_count, first_index, vertex_offset, first_instance);
  }

  /// Draw indirect.
  /// param index: The index of the command buffer.
  /// param command_buffers: The command buffers.
  /// param buffer: The buffer.
  /// param offset: The offset.
  /// param draw_count: The draw count.
  /// param stride: The stride.
  pub fn draw_indirect(
    &self,
    index: usize,
    command_buffers: &HalaCommandBufferSet,
    buffer: &HalaBuffer,
    offset: u64,
    draw_count: u32,
    stride: u32)
  {
    command_buffers.draw_indirect(index, buffer, offset, draw_count, stride);
  }

  /// Draw indexed indirect.
  /// param index: The index of the command buffer.
  /// param command_buffers: The command buffers.
  /// param buffer: The buffer.
  /// param offset: The offset.
  /// param draw_count: The draw count.
  /// param stride: The stride.
  pub fn draw_indexed_indirect(
    &self,
    index: usize,
    command_buffers: &HalaCommandBufferSet,
    buffer: &HalaBuffer,
    offset: u64,
    draw_count: u32,
    stride: u32)
  {
    command_buffers.draw_indexed_indirect(index, buffer, offset, draw_count, stride);
  }

  /// Draw indirect count.
  /// param index: The index of the command buffer.
  /// param command_buffers: The command buffers.
  /// param buffer: The buffer.
  /// param offset: The offset.
  /// param count_buffer: The count buffer.
  /// param count_buffer_offset: The buffer count offset.
  /// param max_draw_count: The max draw count.
  /// param stride: The stride.
  pub fn draw_indirect_count(
    &self,
    index: usize,
    command_buffers: &HalaCommandBufferSet,
    buffer: &HalaBuffer,
    offset: u64,
    count_buffer: &HalaBuffer,
    count_buffer_offset: u64,
    max_draw_count: u32,
    stride: u32)
  {
    command_buffers.draw_indirect_count(index, buffer, offset, count_buffer, count_buffer_offset, max_draw_count, stride);
  }

  /// Draw indexed indirect count.
  /// param index: The index of the command buffer.
  /// param command_buffers: The command buffers.
  /// param buffer: The buffer.
  /// param offset: The offset.
  /// param count_buffer: The count buffer.
  /// param count_buffer_offset: The count buffer offset.
  /// param max_draw_count: The max draw count.
  /// param stride: The stride.
  pub fn draw_indexed_indirect_count(
    &self,
    index: usize,
    command_buffers: &HalaCommandBufferSet,
    buffer: &HalaBuffer,
    offset: u64,
    count_buffer: &HalaBuffer,
    count_buffer_offset: u64,
    max_draw_count: u32,
    stride: u32)
  {
    command_buffers.draw_indexed_indirect_count(index, buffer, offset, count_buffer, count_buffer_offset, max_draw_count, stride);
  }

  /// Draw mesh tasks.
  /// param index: The index of the command buffer.
  /// param command_buffers: The command buffers.
  /// param group_count_x: The group count x.
  /// param group_count_y: The group count y.
  /// param group_count_z: The group count z.
  pub fn draw_mesh_tasks(
    &self,
    index: usize,
    command_buffers: &HalaCommandBufferSet,
    group_count_x: u32,
    group_count_y: u32,
    group_count_z: u32,
  ) {
    command_buffers.draw_mesh_tasks(index, group_count_x, group_count_y, group_count_z);
  }

  /// Draw mesh tasks indirect.
  /// param index: The index of the command buffer.
  /// param command_buffers: The command buffers.
  /// param buffer: The buffer.
  /// param offset: The offset.
  /// param draw_count: The draw count.
  /// param stride: The stride.
  pub fn draw_mesh_tasks_indirect(
    &self,
    index: usize,
    command_buffers: &HalaCommandBufferSet,
    buffer: &HalaBuffer,
    offset: u64,
    draw_count: u32,
    stride: u32,
  ) {
    command_buffers.draw_mesh_tasks_indirect(index, buffer, offset, draw_count, stride);
  }

  /// Draw mesh tasks indirect count.
  /// param index: The index of the command buffer.
  /// param command_buffers: The command buffers.
  /// param buffer: The buffer.
  /// param offset: The offset.
  /// param count_buffer: The count buffer.
  /// param count_buffer_offset: The count buffer offset.
  /// param max_draw_count: The max draw count.
  /// param stride: The stride.
  pub fn draw_mesh_tasks_indirect_count(
    &self,
    index: usize,
    command_buffers: &HalaCommandBufferSet,
    buffer: &HalaBuffer,
    offset: u64,
    count_buffer: &HalaBuffer,
    count_buffer_offset: u64,
    max_draw_count: u32,
    stride: u32,
  ) {
    command_buffers.draw_mesh_tasks_indirect_count(index, buffer, offset, count_buffer, count_buffer_offset, max_draw_count, stride);
  }

}