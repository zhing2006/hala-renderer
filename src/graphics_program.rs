use std::rc::Rc;
use std::cell::RefCell;

use serde::{Serialize, Deserialize};

use hala_gfx::{
  HalaBlendState,
  HalaBuffer,
  HalaCommandBufferSet,
  HalaDepthState,
  HalaDescriptorSet,
  HalaDescriptorSetLayout,
  HalaDescriptorType,
  HalaDynamicState,
  HalaFormat,
  HalaGraphicsPipeline,
  HalaImage,
  HalaLogicalDevice,
  HalaPipelineCache,
  HalaPipelineCreateFlags,
  HalaPrimitiveTopology,
  HalaPushConstantRange,
  HalaRasterizerState,
  HalaMultisampleState,
  HalaRayTracingShaderGroupType,
  HalaShader,
  HalaShaderStageFlags,
  HalaStencilState,
  HalaSwapchain,
  HalaVertexInputAttributeDescription,
  HalaVertexInputBindingDescription,
};

use crate::error::HalaRendererError;
use crate::shader_cache::HalaShaderCache;

type RcRefHalaShader = Rc<RefCell<HalaShader>>;
type OptionRcRefHalaShader = Option<RcRefHalaShader>;

/// The graphics program description.
#[derive(Serialize, Deserialize)]
pub struct HalaGraphicsProgramDesc {
  pub vertex_shader_file_path: Option<String>,
  pub task_shader_file_path: Option<String>,
  pub mesh_shader_file_path: Option<String>,
  pub fragment_shader_file_path: String,
  #[serde(default)]
  pub push_constant_size: u32,
  #[serde(default)]
  pub bindings: Vec<HalaDescriptorType>,
  #[serde(default)]
  pub primitive_topology: HalaPrimitiveTopology,
  #[serde(default)]
  pub color_blend: HalaBlendState,
  #[serde(default)]
  pub alpha_blend: HalaBlendState,
  #[serde(default)]
  pub rasterizer_info: HalaRasterizerState,
  #[serde(default)]
  pub multisample_info: HalaMultisampleState,
  #[serde(default)]
  pub depth_info: HalaDepthState,
  #[serde(default)]
  pub stencil_info: Option<HalaStencilState>,
}

/// The graphics program.
pub struct HalaGraphicsProgram {
  vertex_shader: OptionRcRefHalaShader,
  task_shader: OptionRcRefHalaShader,
  mesh_shader: OptionRcRefHalaShader,
  #[allow(dead_code)]
  fragment_shader: RcRefHalaShader,
  pipeline: HalaGraphicsPipeline,
}

/// The implementation of the graphics program.
#[allow(clippy::too_many_arguments)]
impl HalaGraphicsProgram {

  /// Load the shaders.
  /// param logical_device: The logical device.
  /// param desc: The graphics program description.
  /// param debug_name: The debug name.
  /// return: The result of the shaders.
  fn load_shaders(
    logical_device: Rc<RefCell<HalaLogicalDevice>>,
    desc: &HalaGraphicsProgramDesc,
    debug_name: &str,
  ) -> Result<
    (HalaShaderStageFlags, OptionRcRefHalaShader, OptionRcRefHalaShader, OptionRcRefHalaShader, RcRefHalaShader),
    HalaRendererError,
  > {
    let mut shader_stage = HalaShaderStageFlags::FRAGMENT;
    let vertex_shader = if let Some(ref vertex_shader_file_path) = desc.vertex_shader_file_path {
      shader_stage |= HalaShaderStageFlags::VERTEX;
      Some(HalaShaderCache::get_instance().borrow_mut().load(
        logical_device.clone(),
        vertex_shader_file_path,
        HalaShaderStageFlags::VERTEX,
        HalaRayTracingShaderGroupType::GENERAL,
        &format!("{}.vert.spv", debug_name),
      )?)
    } else {
      None
    };

    let task_shader = if let Some(ref task_shader_file_path) = desc.task_shader_file_path {
      shader_stage |= HalaShaderStageFlags::TASK;
      Some(HalaShaderCache::get_instance().borrow_mut().load(
        logical_device.clone(),
        task_shader_file_path,
        HalaShaderStageFlags::TASK,
        HalaRayTracingShaderGroupType::GENERAL,
        &format!("{}.task.spv", debug_name),
      )?)
    } else {
      None
    };

    let mesh_shader = if let Some(ref mesh_shader_file_path) = desc.mesh_shader_file_path {
      shader_stage |= HalaShaderStageFlags::MESH;
      Some(HalaShaderCache::get_instance().borrow_mut().load(
        logical_device.clone(),
        mesh_shader_file_path,
        HalaShaderStageFlags::MESH,
        HalaRayTracingShaderGroupType::GENERAL,
        &format!("{}.mesh.spv", debug_name),
      )?)
    } else {
      None
    };

    let fragment_shader = HalaShaderCache::get_instance().borrow_mut().load(
      logical_device.clone(),
      &desc.fragment_shader_file_path,
      HalaShaderStageFlags::FRAGMENT,
      HalaRayTracingShaderGroupType::GENERAL,
      &format!("{}.frag.spv", debug_name),
    )?;

    Ok((shader_stage, vertex_shader, task_shader, mesh_shader, fragment_shader))
  }

  /// Create a new graphics program.
  /// param logical_device: The logical device.
  /// param swapchain: The swapchain.
  /// param descriptor_set_layouts: The descriptor set layouts.
  /// param flags: The pipeline create flags.
  /// param vertex_attribute_descriptions: The vertex attribute descriptions.
  /// param vertex_binding_descriptions: The vertex binding descriptions.
  /// param dynamic_states: The dynamic states.
  /// param desc: The graphics program description.
  /// param pipeline_cache: The pipeline cache.
  /// param debug_name: The debug name.
  /// return: The result of the graphics program.
  pub fn new<DSL, VIAD, VIBD>(
    logical_device: Rc<RefCell<HalaLogicalDevice>>,
    swapchain: &HalaSwapchain,
    descriptor_set_layouts: &[DSL],
    flags: HalaPipelineCreateFlags,
    vertex_attribute_descriptions: &[VIAD],
    vertex_binding_descriptions: &[VIBD],
    dynamic_states: &[HalaDynamicState],
    desc: &HalaGraphicsProgramDesc,
    pipeline_cache: Option<&HalaPipelineCache>,
    debug_name: &str,
  ) -> Result<Self, HalaRendererError>
    where
      DSL: AsRef<HalaDescriptorSetLayout>,
      VIAD: AsRef<HalaVertexInputAttributeDescription>,
      VIBD: AsRef<HalaVertexInputBindingDescription>,
  {
    Self::with_formats_and_size(
      logical_device,
      &[swapchain.desc.format],
      Some(swapchain.depth_stencil_format),
      swapchain.desc.dims.width,
      swapchain.desc.dims.height,
      descriptor_set_layouts,
      flags,
      vertex_attribute_descriptions,
      vertex_binding_descriptions,
      dynamic_states,
      desc,
      pipeline_cache,
      debug_name,
    )
  }

  /// Create a new graphics program with custom render target.
  /// param logical_device: The logical device.
  /// param color_images: The color images.
  /// param depth_image: The depth image.
  /// param descriptor_set_layouts: The descriptor set layouts.
  /// param flags: The pipeline create flags.
  /// param vertex_attribute_descriptions: The vertex attribute descriptions.
  /// param vertex_binding_descriptions: The vertex binding descriptions.
  /// param dynamic_states: The dynamic states.
  /// param desc: The graphics program description.
  /// param pipeline_cache: The pipeline cache.
  /// param debug_name: The debug name.
  /// return: The result of the graphics program.
  pub fn with_rt<T, DSL, VIAD, VIBD>(
    logical_device: Rc<RefCell<HalaLogicalDevice>>,
    color_images: &[T],
    depth_image: Option<&T>,
    descriptor_set_layouts: &[DSL],
    flags: HalaPipelineCreateFlags,
    vertex_attribute_descriptions: &[VIAD],
    vertex_binding_descriptions: &[VIBD],
    dynamic_states: &[HalaDynamicState],
    desc: &HalaGraphicsProgramDesc,
    pipeline_cache: Option<&HalaPipelineCache>,
    debug_name: &str,
  ) -> Result<Self, HalaRendererError>
    where
      T: AsRef<HalaImage>,
      DSL: AsRef<HalaDescriptorSetLayout>,
      VIAD: AsRef<HalaVertexInputAttributeDescription>,
      VIBD: AsRef<HalaVertexInputBindingDescription>,
  {
    Self::with_formats_and_size(
      logical_device,
      color_images.iter().map(|i| i.as_ref().format).collect::<Vec<_>>().as_slice(),
      depth_image.map(|i| i.as_ref().format),
      color_images[0].as_ref().extent.width,
      color_images[0].as_ref().extent.height,
      descriptor_set_layouts,
      flags,
      vertex_attribute_descriptions,
      vertex_binding_descriptions,
      dynamic_states,
      desc,
      pipeline_cache,
      debug_name,
    )
  }

  /// Create a new graphics program with custom formats and size.
  /// param logical_device: The logical device.
  /// param color_formats: The color formats.
  /// param depth_format: The depth format.
  /// param width: The width.
  /// param height: The height.
  /// param descriptor_set_layouts: The descriptor set layouts.
  /// param flags: The pipeline create flags.
  /// param vertex_attribute_descriptions: The vertex attribute descriptions.
  /// param vertex_binding_descriptions: The vertex binding descriptions.
  /// param dynamic_states: The dynamic states.
  /// param desc: The graphics program description.
  /// param pipeline_cache: The pipeline cache.
  /// param debug_name: The debug name.
  /// return: The result of the graphics program.
  pub fn with_formats_and_size<DSL, VIAD, VIBD>(
    logical_device: Rc<RefCell<HalaLogicalDevice>>,
    color_formats: &[HalaFormat],
    depth_format: Option<HalaFormat>,
    width: u32,
    height: u32,
    descriptor_set_layouts: &[DSL],
    flags: HalaPipelineCreateFlags,
    vertex_attribute_descriptions: &[VIAD],
    vertex_binding_descriptions: &[VIBD],
    dynamic_states: &[HalaDynamicState],
    desc: &HalaGraphicsProgramDesc,
    pipeline_cache: Option<&HalaPipelineCache>,
    debug_name: &str,
  ) -> Result<Self, HalaRendererError>
    where
      DSL: AsRef<HalaDescriptorSetLayout>,
      VIAD: AsRef<HalaVertexInputAttributeDescription>,
      VIBD: AsRef<HalaVertexInputBindingDescription>,
  {
    let (
      shader_stage,
      vertex_shader,
      task_shader,
      mesh_shader,
      fragment_shader,
    ) = Self::load_shaders(
      logical_device.clone(),
      desc,
      debug_name,
    )?;

    let pipeline = {
      let mut shaders = Vec::new();
      if let Some(ref vertex_shader) = vertex_shader {
        shaders.push(vertex_shader.borrow());
      }
      if let Some(ref task_shader) = task_shader {
        shaders.push(task_shader.borrow());
      }
      if let Some(ref mesh_shader) = mesh_shader {
        shaders.push(mesh_shader.borrow());
      }
      shaders.push(fragment_shader.borrow());
      let push_constant_ranges = if desc.push_constant_size > 0 {
        &[
          HalaPushConstantRange {
            stage_flags: shader_stage,
            offset: 0,
            size: desc.push_constant_size,
          },
        ]
      } else {
        &[] as &[HalaPushConstantRange]
      };
      let color_blends = vec![&desc.color_blend; color_formats.len()];
      let alpha_blends = vec![&desc.alpha_blend; color_formats.len()];
      HalaGraphicsPipeline::with_format_and_size(
        logical_device.clone(),
        color_formats,
        depth_format,
        width,
        height,
        descriptor_set_layouts,
        flags,
        vertex_attribute_descriptions,
        vertex_binding_descriptions,
        push_constant_ranges,
        desc.primitive_topology,
        color_blends.as_slice(),
        alpha_blends.as_slice(),
        &desc.rasterizer_info,
        &desc.multisample_info,
        &desc.depth_info,
        desc.stencil_info.as_ref(),
        &shaders.iter().map(|s| s.as_ref()).collect::<Vec<_>>(),
        dynamic_states,
        pipeline_cache,
        &format!("{}.graphics_pipeline", debug_name),
      )?
    };

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

  /// Push constants.
  /// param index: The index of the command buffer.
  /// param command_buffers: The command buffers.
  /// param offset: The offset.
  /// param data: The data.
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
    command_buffers.push_constants(index, self.pipeline.layout, shader_stage, offset, data);
  }

  /// Push constants f32.
  /// param index: The index of the command buffer.
  /// param command_buffers: The command buffers.
  /// param offset: The offset.
  /// param data: The data.
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
    command_buffers.push_constants_f32(index, self.pipeline.layout, shader_stage, offset, data);
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