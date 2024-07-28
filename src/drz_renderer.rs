use std::rc::Rc;

use hala_gfx::{
  HalaGPURequirements,
  HalaShader,
};

use crate::error::HalaRendererError;
use crate::scene::{
  cpu,
  gpu,
  loader,
};

use crate::renderer::{
  HalaRendererInfo,
  HalaRendererResources,
  HalaRendererData,
  HalaRendererStatistics,
  HalaRendererTrait,
};

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

  pub(crate) info: HalaRendererInfo,

  pub(crate) use_mesh_shader: bool,

  pub(crate) resources: std::mem::ManuallyDrop<HalaRendererResources>,

  pub(crate) use_deferred: bool,
  pub(crate) depth_image: Option<hala_gfx::HalaImage>,
  pub(crate) albedo_image: Option<hala_gfx::HalaImage>,
  pub(crate) normal_image: Option<hala_gfx::HalaImage>,

  pub(crate) static_descriptor_set: std::mem::ManuallyDrop<hala_gfx::HalaDescriptorSet>,
  pub(crate) global_uniform_buffer: std::mem::ManuallyDrop<hala_gfx::HalaBuffer>,
  pub(crate) dynamic_descriptor_set: Option<hala_gfx::HalaDescriptorSet>,
  pub(crate) object_uniform_buffers: Vec<Vec<hala_gfx::HalaBuffer>>,

  // Vertex Shader, Fragment Shader.
  pub(crate) traditional_shaders: Vec<(hala_gfx::HalaShader, hala_gfx::HalaShader)>,
  // Task Shader, Mesh Shader and Fragment Shader.
  pub(crate) shaders: Vec<(Option<hala_gfx::HalaShader>, hala_gfx::HalaShader, hala_gfx::HalaShader)>,
  // Compute Shader.
  pub(crate) compute_shaders: Vec<hala_gfx::HalaShader>,

  pub(crate) scene_in_gpu: Option<gpu::HalaScene>,

  pub(crate) forward_graphics_pipelines: Vec<hala_gfx::HalaGraphicsPipeline>,
  pub(crate) deferred_graphics_pipelines: Vec<hala_gfx::HalaGraphicsPipeline>,
  pub(crate) compute_pipelines: Vec<hala_gfx::HalaComputePipeline>,
  pub(crate) textures_descriptor_set: Option<hala_gfx::HalaDescriptorSet>,

  pub(crate) data: HalaRendererData,
  pub(crate) statistics: HalaRendererStatistics,

}

/// The Drop implementation of the renderer.
impl Drop for HalaRenderer {

  fn drop(&mut self) {
    self.textures_descriptor_set = None;
    self.forward_graphics_pipelines.clear();
    self.deferred_graphics_pipelines.clear();
    self.compute_pipelines.clear();

    self.scene_in_gpu = None;

    self.traditional_shaders.clear();
    self.shaders.clear();
    self.compute_shaders.clear();

    self.use_deferred = false;
    self.depth_image = None;
    self.albedo_image = None;
    self.normal_image = None;

    self.object_uniform_buffers.clear();
    self.dynamic_descriptor_set = None;
    unsafe {
      std::mem::ManuallyDrop::drop(&mut self.global_uniform_buffer);
      std::mem::ManuallyDrop::drop(&mut self.static_descriptor_set);
      std::mem::ManuallyDrop::drop(&mut self.resources);
    }

    log::debug!("A HalaRenderer \"{}\" is dropped.", self.info().name);
  }

}

/// The implementation of the renderer trait.
impl HalaRendererTrait for HalaRenderer {

  fn info(&self) -> &HalaRendererInfo {
    &self.info
  }

  fn info_mut(&mut self) -> &mut HalaRendererInfo {
    &mut self.info
  }

  fn resources(&self) -> &HalaRendererResources {
    &self.resources
  }

  fn resources_mut(&mut self) -> &mut HalaRendererResources {
    &mut self.resources
  }

  fn data(&self) -> &HalaRendererData {
    &self.data
  }

  fn data_mut(&mut self) -> &mut HalaRendererData {
    &mut self.data
  }

  fn statistics(&self) -> &HalaRendererStatistics {
    &self.statistics
  }

  fn statistics_mut(&mut self) -> &mut HalaRendererStatistics {
    &mut self.statistics
  }

  fn get_descriptor_sizes() -> Vec<(hala_gfx::HalaDescriptorType, usize)> {
    vec![
      (
        hala_gfx::HalaDescriptorType::STORAGE_IMAGE,
        8,
      ),
      (
        hala_gfx::HalaDescriptorType::STORAGE_BUFFER,
        32,
      ),
      (
        hala_gfx::HalaDescriptorType::UNIFORM_BUFFER,
        256,
      ),
      (
        hala_gfx::HalaDescriptorType::SAMPLED_IMAGE,
        256,
      ),
      (
        hala_gfx::HalaDescriptorType::SAMPLER,
        256,
      ),
      (
        hala_gfx::HalaDescriptorType::COMBINED_IMAGE_SAMPLER,
        256,
      ),
    ]
  }

  /// Commit all GPU resources.
  /// return: The result.
  fn commit(&mut self) -> Result<(), HalaRendererError> {
    let context = self.resources.context.borrow();
    let scene = self.scene_in_gpu.as_ref().ok_or(HalaRendererError::new("The scene in GPU is none!", None))?;

    // Assert camera count.
    if scene.camera_view_matrices.is_empty() || scene.camera_proj_matrices.is_empty() {
      return Err(HalaRendererError::new("There is no camera in the scene!", None));
    }

    // Collect vertex and index buffers.
    let mut vertex_buffers = Vec::new();
    let mut index_buffers = Vec::new();
    let mut meshlet_buffers = Vec::new();
    let mut meshlet_vertex_buffers = Vec::new();
    let mut meshlet_primitive_buffers = Vec::new();
    for mesh in scene.meshes.iter() {
      for primitive in mesh.primitives.iter() {
        vertex_buffers.push(primitive.vertex_buffer.as_ref());
        index_buffers.push(primitive.index_buffer.as_ref());
        if self.use_mesh_shader {
          meshlet_buffers.push(primitive.meshlet_buffer.as_ref().ok_or(HalaRendererError::new("The meshlet buffer is none!", None))?);
          meshlet_vertex_buffers.push(primitive.meshlet_vertex_buffer.as_ref().ok_or(HalaRendererError::new("The meshlet vertex buffer is none!", None))?);
          meshlet_primitive_buffers.push(primitive.meshlet_primitive_buffer.as_ref().ok_or(HalaRendererError::new("The meshlet primitive buffer is none!", None))?);
        }
      }
    }

    // Create dynamic descriptor set.
    let dynamic_descriptor_set = hala_gfx::HalaDescriptorSet::new(
      Rc::clone(&context.logical_device),
      Rc::clone(&self.resources.descriptor_pool),
      hala_gfx::HalaDescriptorSetLayout::new(
        Rc::clone(&context.logical_device),
        &[
          hala_gfx::HalaDescriptorSetLayoutBinding { // Materials uniform buffers.
            binding_index: 0,
            descriptor_type: hala_gfx::HalaDescriptorType::UNIFORM_BUFFER,
            descriptor_count: scene.materials.len() as u32,
            stage_flags: hala_gfx::HalaShaderStageFlags::FRAGMENT | hala_gfx::HalaShaderStageFlags::COMPUTE
              | (if self.use_mesh_shader { hala_gfx::HalaShaderStageFlags::TASK | hala_gfx::HalaShaderStageFlags::MESH } else { hala_gfx::HalaShaderStageFlags::VERTEX }),
            binding_flags: hala_gfx::HalaDescriptorBindingFlags::PARTIALLY_BOUND
          },
          hala_gfx::HalaDescriptorSetLayoutBinding { // Object uniform buffers.
            binding_index: 1,
            descriptor_type: hala_gfx::HalaDescriptorType::UNIFORM_BUFFER,
            descriptor_count: scene.meshes.len() as u32,
            stage_flags: hala_gfx::HalaShaderStageFlags::FRAGMENT | hala_gfx::HalaShaderStageFlags::COMPUTE
              | (if self.use_mesh_shader { hala_gfx::HalaShaderStageFlags::TASK | hala_gfx::HalaShaderStageFlags::MESH } else { hala_gfx::HalaShaderStageFlags::VERTEX }),
            binding_flags: hala_gfx::HalaDescriptorBindingFlags::PARTIALLY_BOUND
          },
          hala_gfx::HalaDescriptorSetLayoutBinding { // Vertex storage buffers.
            binding_index: 2,
            descriptor_type: hala_gfx::HalaDescriptorType::STORAGE_BUFFER,
            descriptor_count: vertex_buffers.len() as u32,
            stage_flags: hala_gfx::HalaShaderStageFlags::FRAGMENT | hala_gfx::HalaShaderStageFlags::COMPUTE
              | (if self.use_mesh_shader { hala_gfx::HalaShaderStageFlags::TASK | hala_gfx::HalaShaderStageFlags::MESH } else { hala_gfx::HalaShaderStageFlags::VERTEX }),
            binding_flags: hala_gfx::HalaDescriptorBindingFlags::PARTIALLY_BOUND
          },
          hala_gfx::HalaDescriptorSetLayoutBinding { // Index storage buffers.
            binding_index: 3,
            descriptor_type: hala_gfx::HalaDescriptorType::STORAGE_BUFFER,
            descriptor_count: index_buffers.len() as u32,
            stage_flags: hala_gfx::HalaShaderStageFlags::FRAGMENT | hala_gfx::HalaShaderStageFlags::COMPUTE
              | (if self.use_mesh_shader { hala_gfx::HalaShaderStageFlags::TASK | hala_gfx::HalaShaderStageFlags::MESH } else { hala_gfx::HalaShaderStageFlags::VERTEX }),
            binding_flags: hala_gfx::HalaDescriptorBindingFlags::PARTIALLY_BOUND
          },
          hala_gfx::HalaDescriptorSetLayoutBinding { // Meshlet information storage buffers.
            binding_index: 4,
            descriptor_type: hala_gfx::HalaDescriptorType::STORAGE_BUFFER,
            descriptor_count: meshlet_buffers.len() as u32,
            stage_flags: hala_gfx::HalaShaderStageFlags::FRAGMENT | hala_gfx::HalaShaderStageFlags::COMPUTE
              | (if self.use_mesh_shader { hala_gfx::HalaShaderStageFlags::TASK | hala_gfx::HalaShaderStageFlags::MESH } else { hala_gfx::HalaShaderStageFlags::VERTEX }),
            binding_flags: hala_gfx::HalaDescriptorBindingFlags::PARTIALLY_BOUND
          },
          hala_gfx::HalaDescriptorSetLayoutBinding { // Meshlet vertex storage buffers.
            binding_index: 5,
            descriptor_type: hala_gfx::HalaDescriptorType::STORAGE_BUFFER,
            descriptor_count: meshlet_vertex_buffers.len() as u32,
            stage_flags: hala_gfx::HalaShaderStageFlags::FRAGMENT | hala_gfx::HalaShaderStageFlags::COMPUTE
              | (if self.use_mesh_shader { hala_gfx::HalaShaderStageFlags::TASK | hala_gfx::HalaShaderStageFlags::MESH } else { hala_gfx::HalaShaderStageFlags::VERTEX }),
            binding_flags: hala_gfx::HalaDescriptorBindingFlags::PARTIALLY_BOUND
          },
          hala_gfx::HalaDescriptorSetLayoutBinding { // Meshlet primitive storage buffers.
            binding_index: 6,
            descriptor_type: hala_gfx::HalaDescriptorType::STORAGE_BUFFER,
            descriptor_count: meshlet_primitive_buffers.len() as u32,
            stage_flags: hala_gfx::HalaShaderStageFlags::FRAGMENT | hala_gfx::HalaShaderStageFlags::COMPUTE
              | (if self.use_mesh_shader { hala_gfx::HalaShaderStageFlags::TASK | hala_gfx::HalaShaderStageFlags::MESH } else { hala_gfx::HalaShaderStageFlags::VERTEX }),
            binding_flags: hala_gfx::HalaDescriptorBindingFlags::PARTIALLY_BOUND
          },
        ],
        "main_dynamic.descriptor_set_layout",
      )?,
      context.swapchain.num_of_images,
      0,
      "main_dynamic.descriptor_set",
    )?;

    for (mesh_index, _mesh) in scene.meshes.iter().enumerate() {
      // Create object uniform buffer.
      let mut buffers = Vec::with_capacity(context.swapchain.num_of_images);
      for index in 0..context.swapchain.num_of_images {
        let buffer = hala_gfx::HalaBuffer::new(
          Rc::clone(&context.logical_device),
          std::mem::size_of::<HalaObjectUniform>() as u64,
          hala_gfx::HalaBufferUsageFlags::UNIFORM_BUFFER,
          hala_gfx::HalaMemoryLocation::CpuToGpu,
          &format!("object_{}_{}.uniform_buffer", mesh_index, index),
        )?;

        buffers.push(buffer);
      }

      self.object_uniform_buffers.push(buffers);
    }

    for index in 0..context.swapchain.num_of_images {
      dynamic_descriptor_set.update_uniform_buffers(
        index,
        0,
        scene.materials.as_slice(),
      );
      dynamic_descriptor_set.update_uniform_buffers(
        index,
        1,
        self.object_uniform_buffers.iter().map(|buffers| &buffers[index]).collect::<Vec<_>>().as_slice(),
      );
      dynamic_descriptor_set.update_storage_buffers(
        index,
        2,
        vertex_buffers.as_slice(),
      );
      dynamic_descriptor_set.update_storage_buffers(
        index,
        3,
        index_buffers.as_slice(),
      );
      if !meshlet_buffers.is_empty() {
        dynamic_descriptor_set.update_storage_buffers(
          index,
          4,
          meshlet_buffers.as_slice(),
        );
      }
      if !meshlet_vertex_buffers.is_empty() {
        dynamic_descriptor_set.update_storage_buffers(
          index,
          5,
          meshlet_vertex_buffers.as_slice(),
        );
      }
      if !meshlet_primitive_buffers.is_empty() {
        dynamic_descriptor_set.update_storage_buffers(
          index,
          6,
          meshlet_primitive_buffers.as_slice(),
        );
      }
    }

    // Update static descriptor set.
    self.static_descriptor_set.update_uniform_buffers(0, 0, &[self.global_uniform_buffer.as_ref()]);
    self.static_descriptor_set.update_uniform_buffers(0, 1, &[scene.cameras.as_ref()]);
    self.static_descriptor_set.update_uniform_buffers(0, 2, &[scene.lights.as_ref()]);

    // Create texture descriptor set.
    let textures_descriptor_set = hala_gfx::HalaDescriptorSet::new_static(
      Rc::clone(&context.logical_device),
      Rc::clone(&self.resources.descriptor_pool),
      hala_gfx::HalaDescriptorSetLayout::new(
        Rc::clone(&context.logical_device),
        &[
          hala_gfx::HalaDescriptorSetLayoutBinding { // All textures in the scene.
            binding_index: 0,
            descriptor_type: hala_gfx::HalaDescriptorType::SAMPLED_IMAGE,
            descriptor_count: scene.textures.len() as u32,
            stage_flags: hala_gfx::HalaShaderStageFlags::FRAGMENT | hala_gfx::HalaShaderStageFlags::COMPUTE
              | (if self.use_mesh_shader { hala_gfx::HalaShaderStageFlags::TASK | hala_gfx::HalaShaderStageFlags::MESH } else { hala_gfx::HalaShaderStageFlags::VERTEX }),
            binding_flags: hala_gfx::HalaDescriptorBindingFlags::PARTIALLY_BOUND
          },
          hala_gfx::HalaDescriptorSetLayoutBinding { // All samplers in the scene.
            binding_index: 1,
            descriptor_type: hala_gfx::HalaDescriptorType::SAMPLER,
            descriptor_count: scene.textures.len() as u32,
            stage_flags: hala_gfx::HalaShaderStageFlags::FRAGMENT | hala_gfx::HalaShaderStageFlags::COMPUTE
              | (if self.use_mesh_shader { hala_gfx::HalaShaderStageFlags::TASK | hala_gfx::HalaShaderStageFlags::MESH } else { hala_gfx::HalaShaderStageFlags::VERTEX }),
            binding_flags: hala_gfx::HalaDescriptorBindingFlags::PARTIALLY_BOUND
          },
        ],
        "textures.descriptor_set_layout",
      )?,
      0,
      "textures.descriptor_set",
    )?;

    let textures: &Vec<_> = scene.textures.as_ref();
    let samplers: &Vec<_> = scene.samplers.as_ref();
    let images: &Vec<_> = scene.images.as_ref();
    let mut final_images = Vec::new();
    let mut final_samplers = Vec::new();
    for (sampler_index, image_index) in textures.iter().enumerate() {
      let image = images.get(*image_index as usize).ok_or(HalaRendererError::new("The image is none!", None))?;
      let sampler = samplers.get(sampler_index).ok_or(HalaRendererError::new("The sampler is none!", None))?;
      final_images.push(image);
      final_samplers.push(sampler);
    }
    if !final_images.is_empty() && !final_samplers.is_empty() {
      textures_descriptor_set.update_sampled_images(0, 0, final_images.as_slice());
      textures_descriptor_set.update_samplers(0, 1, final_samplers.as_slice());
    }

    // If we have cache file at ./out/pipeline_cache.bin, we can load it.
    let pipeline_cache = if std::path::Path::new("./out/pipeline_cache.bin").exists() {
      log::debug!("Load pipeline cache from file: ./out/pipeline_cache.bin");
      hala_gfx::HalaPipelineCache::with_cache_file(
        Rc::clone(&context.logical_device),
        "./out/pipeline_cache.bin",
      )?
    } else {
      log::debug!("Create a new pipeline cache.");
      hala_gfx::HalaPipelineCache::new(
        Rc::clone(&context.logical_device),
      )?
    };

    let mut pso_shader_list: Vec<Vec<&HalaShader>> = Vec::new();
    if self.use_mesh_shader {
      // Collect modern graphics shaders.
      for (task_shader, mesh_shader, fragment_shader) in self.shaders.iter() {
        let mut shaders = Vec::with_capacity(3);
        if let Some(task_shader) = task_shader {
          shaders.push(task_shader.as_ref());
        }
        shaders.push(mesh_shader.as_ref());
        shaders.push(fragment_shader.as_ref());
        pso_shader_list.push(shaders);
      }
    } else {
      // Collect traditional graphics shaders.
      for shaders in self.traditional_shaders.iter() {
        pso_shader_list.push(vec![shaders.0.as_ref(), shaders.1.as_ref()])
      }
    }

    // Create graphics pipelines.
    for (i, shaders) in pso_shader_list.iter().enumerate() {
      let descriptor_set_layouts = [&self.static_descriptor_set.layout, &dynamic_descriptor_set.layout, &textures_descriptor_set.layout];
      let flags = hala_gfx::HalaPipelineCreateFlags::default();
      let vertex_attribute_descriptions = [
        hala_gfx::HalaVertexInputAttributeDescription {
          binding: 0,
          location: 0,
          offset: 0,
          format: hala_gfx::HalaFormat::R32G32B32_SFLOAT, // Position.
        },
        hala_gfx::HalaVertexInputAttributeDescription {
          binding: 0,
          location: 1,
          offset: 12,
          format: hala_gfx::HalaFormat::R32G32B32_SFLOAT, // Normal.
        },
        hala_gfx::HalaVertexInputAttributeDescription {
          binding: 0,
          location: 2,
          offset: 24,
          format: hala_gfx::HalaFormat::R32G32B32_SFLOAT, // Tangent.
        },
        hala_gfx::HalaVertexInputAttributeDescription {
          binding: 0,
          location: 3,
          offset: 36,
          format: hala_gfx::HalaFormat::R32G32_SFLOAT,  // UV.
        },
      ];
      let vertex_binding_descriptions = [
        hala_gfx::HalaVertexInputBindingDescription {
          binding: 0,
          stride: 44,
          input_rate: hala_gfx::HalaVertexInputRate::VERTEX,
        }
      ];
      let push_constant_ranges = [
        hala_gfx::HalaPushConstantRange {
          stage_flags: hala_gfx::HalaShaderStageFlags::FRAGMENT
            | (if self.use_mesh_shader { hala_gfx::HalaShaderStageFlags::TASK | hala_gfx::HalaShaderStageFlags::MESH } else { hala_gfx::HalaShaderStageFlags::VERTEX }),
          offset: 0,
          size: if !self.use_mesh_shader {
            12  // Mesh index, Material index and Primitive index.
          } else {
            20  // Mesh index, Material index, Primitive index, Meshlet count and Dispatch size X.
          }
        },
      ];

      self.forward_graphics_pipelines.push(
        hala_gfx::HalaGraphicsPipeline::new(
          Rc::clone(&context.logical_device),
          &context.swapchain,
          &descriptor_set_layouts,
          flags,
          &vertex_attribute_descriptions,
          &vertex_binding_descriptions,
          &push_constant_ranges,
          hala_gfx::HalaPrimitiveTopology::TRIANGLE_LIST,
          &hala_gfx::HalaBlendState::new(hala_gfx::HalaBlendFactor::SRC_ALPHA, hala_gfx::HalaBlendFactor::ONE_MINUS_SRC_ALPHA, hala_gfx::HalaBlendOp::ADD),
          &hala_gfx::HalaBlendState::new(hala_gfx::HalaBlendFactor::ONE, hala_gfx::HalaBlendFactor::ZERO, hala_gfx::HalaBlendOp::ADD),
          &hala_gfx::HalaRasterizerState::new(hala_gfx::HalaFrontFace::COUNTER_CLOCKWISE, hala_gfx::HalaCullModeFlags::BACK, hala_gfx::HalaPolygonMode::FILL, 1.0),
          &hala_gfx::HalaDepthState::new(true, true, hala_gfx::HalaCompareOp::GREATER), // We use reverse Z, so greater is less.
          None,
          shaders.as_slice(),
          &[hala_gfx::HalaDynamicState::VIEWPORT],
          Some(&pipeline_cache),
          &if self.use_mesh_shader {
            format!("modern_forward_{}.graphics_pipeline", i)
          } else {
            format!("traditional_forward_{}.graphics_pipeline", i)
          },
        )?
      );
      if self.use_deferred {
        let depth_image = self.depth_image.as_ref().ok_or(
          HalaRendererError::new("The deferred flag is setted, but the G-Buffer depth image is none!", None)
        )?;
        let albedo_image = self.albedo_image.as_ref().ok_or(
          HalaRendererError::new("The deferred flag is setted, but the G-Buffer albedo image is none!", None)
        )?;
        let normal_image = self.normal_image.as_ref().ok_or(
          HalaRendererError::new("The deferred flag is setted, but the G-Buffer normal image is none!", None)
        )?;
        self.deferred_graphics_pipelines.push(
          hala_gfx::HalaGraphicsPipeline::with_format_and_size(
            Rc::clone(&context.logical_device),
            &[albedo_image.format, normal_image.format],
            Some(depth_image.format),
            self.info.width,
            self.info.height,
            &descriptor_set_layouts,
            flags,
            &vertex_attribute_descriptions,
            &vertex_binding_descriptions,
            &push_constant_ranges,
            hala_gfx::HalaPrimitiveTopology::TRIANGLE_LIST,
            &[
              &hala_gfx::HalaBlendState::new(hala_gfx::HalaBlendFactor::ONE, hala_gfx::HalaBlendFactor::ZERO, hala_gfx::HalaBlendOp::ADD),
              &hala_gfx::HalaBlendState::new(hala_gfx::HalaBlendFactor::ONE, hala_gfx::HalaBlendFactor::ZERO, hala_gfx::HalaBlendOp::ADD),
            ],
            &[
              &hala_gfx::HalaBlendState::new(hala_gfx::HalaBlendFactor::ONE, hala_gfx::HalaBlendFactor::ZERO, hala_gfx::HalaBlendOp::ADD),
              &hala_gfx::HalaBlendState::new(hala_gfx::HalaBlendFactor::ONE, hala_gfx::HalaBlendFactor::ZERO, hala_gfx::HalaBlendOp::ADD),
            ],
            &hala_gfx::HalaRasterizerState::new(hala_gfx::HalaFrontFace::COUNTER_CLOCKWISE, hala_gfx::HalaCullModeFlags::BACK, hala_gfx::HalaPolygonMode::FILL, 1.0),
            &hala_gfx::HalaDepthState::new(true, true, hala_gfx::HalaCompareOp::GREATER), // We use reverse Z, so greater is less.
            None,
            shaders.as_slice(),
            &[hala_gfx::HalaDynamicState::VIEWPORT],
            Some(&pipeline_cache),
            &if self.use_mesh_shader {
              format!("modern_deferred_{}.graphics_pipeline", i)
            } else {
              format!("traditional_deferred_{}.graphics_pipeline", i)
            },
          )?
        );
      }
    }

    // Save pipeline cache.
    pipeline_cache.save("./out/pipeline_cache.bin")?;

    self.dynamic_descriptor_set = Some(dynamic_descriptor_set);
    self.textures_descriptor_set = Some(textures_descriptor_set);

    Ok(())
  }

  /// Update the renderer.
  /// param delta_time: The delta time.
  /// param width: The width of the window.
  /// param height: The height of the window.
  /// param ui_fn: The draw UI function.
  /// return: The result.
  fn update<F>(&mut self, _delta_time: f64, width: u32, height: u32, ui_fn: F) -> Result<(), HalaRendererError>
    where F: FnOnce(usize, &hala_gfx::HalaCommandBufferSet) -> Result<(), hala_gfx::HalaGfxError>
  {
    self.pre_update(width, height)?;

    let scene = self.scene_in_gpu.as_ref().ok_or(HalaRendererError::new("The scene in GPU is none!", None))?;
    let context = self.resources.context.borrow();

    // Update global uniform buffer(Only use No.1 camera).
    self.global_uniform_buffer.update_memory(0, &[HalaGlobalUniform {
      v_mtx: scene.camera_view_matrices[0],
      p_mtx: scene.camera_proj_matrices[0],
      vp_mtx: scene.camera_proj_matrices[0] * scene.camera_view_matrices[0],
    }])?;

    // Update object uniform buffers.
    for (mesh_index, mesh) in scene.meshes.iter().enumerate() {
      // Prepare object data.
      let mv_mtx = scene.camera_view_matrices[0] * mesh.transform;
      let object_uniform = HalaObjectUniform {
        m_mtx: mesh.transform,
        i_m_mtx: mesh.transform.inverse(),
        mv_mtx,
        t_mv_mtx: mv_mtx.transpose(),
        it_mv_mtx: mv_mtx.inverse().transpose(),
        mvp_mtx: scene.camera_proj_matrices[0] * mv_mtx,
      };

      for index in 0..context.swapchain.num_of_images {
        let buffer = self.object_uniform_buffers[mesh_index][index].as_ref();
        buffer.update_memory(0, &[object_uniform])?;
      }
    }

    if self.use_deferred {
      self.record_deferred_command_buffer(
        self.data.image_index,
        &self.resources.graphics_command_buffers,
        ui_fn,
      )?;
    } else {
      context.record_graphics_command_buffer(
        self.data.image_index,
        &self.resources.graphics_command_buffers,
        Some([25.0 / 255.0, 118.0 / 255.0, 210.0 / 255.0, 1.0]),
        Some(0.0),
        Some(0),  // We use reverse Z, so clear depth to 0.0.
        |index, command_buffers| {
          self.draw_scene(index, command_buffers, true)?;

          ui_fn(index, command_buffers)?;

          Ok(())
        },
        None,
        |_, _| {
          Ok(false)
        },
      )?;
    }

    Ok(())
  }

}

/// The implementation of the renderer.
impl HalaRenderer {

  /// Create a new renderer.
  /// param name: The name of the renderer.
  /// param gpu_req: The GPU requirements of the renderer.
  /// param window: The window of the renderer.
  /// return: The renderer.
  pub fn new(
    name: &str,
    gpu_req: &HalaGPURequirements,
    window: &winit::window::Window,
  ) -> Result<Self, HalaRendererError> {
    let width = gpu_req.width;
    let height = gpu_req.height;

    let resources = HalaRendererResources::new(
      name,
      gpu_req,
      window,
      &Self::get_descriptor_sizes(),
    )?;

    let static_descriptor_set = hala_gfx::HalaDescriptorSet::new_static(
      Rc::clone(&resources.context.borrow().logical_device),
      Rc::clone(&resources.descriptor_pool),
      hala_gfx::HalaDescriptorSetLayout::new(
        Rc::clone(&resources.context.borrow().logical_device),
        &[
          hala_gfx::HalaDescriptorSetLayoutBinding { // Global uniform buffer.
            binding_index: 0,
            descriptor_type: hala_gfx::HalaDescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
            stage_flags: hala_gfx::HalaShaderStageFlags::FRAGMENT | hala_gfx::HalaShaderStageFlags::COMPUTE
              | (if resources.context.borrow().gpu_req.require_mesh_shader { hala_gfx::HalaShaderStageFlags::TASK | hala_gfx::HalaShaderStageFlags::MESH } else { hala_gfx::HalaShaderStageFlags::VERTEX }),
            binding_flags: hala_gfx::HalaDescriptorBindingFlags::PARTIALLY_BOUND
          },
          hala_gfx::HalaDescriptorSetLayoutBinding { // Cameras uniform buffer.
            binding_index: 1,
            descriptor_type: hala_gfx::HalaDescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
            stage_flags: hala_gfx::HalaShaderStageFlags::FRAGMENT | hala_gfx::HalaShaderStageFlags::COMPUTE
              | (if resources.context.borrow().gpu_req.require_mesh_shader { hala_gfx::HalaShaderStageFlags::TASK | hala_gfx::HalaShaderStageFlags::MESH } else { hala_gfx::HalaShaderStageFlags::VERTEX }),
            binding_flags: hala_gfx::HalaDescriptorBindingFlags::PARTIALLY_BOUND
          },
          hala_gfx::HalaDescriptorSetLayoutBinding { // Lights uniform buffer.
            binding_index: 2,
            descriptor_type: hala_gfx::HalaDescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
            stage_flags: hala_gfx::HalaShaderStageFlags::FRAGMENT | hala_gfx::HalaShaderStageFlags::COMPUTE
              | (if resources.context.borrow().gpu_req.require_mesh_shader { hala_gfx::HalaShaderStageFlags::TASK | hala_gfx::HalaShaderStageFlags::MESH } else { hala_gfx::HalaShaderStageFlags::VERTEX }),
            binding_flags: hala_gfx::HalaDescriptorBindingFlags::PARTIALLY_BOUND
          },
        ],
        "main_static.descriptor_set_layout",
      )?,
      0,
      "main_static.descriptor_set",
    )?;

    // Create global uniform buffer.
    let global_uniform_buffer = hala_gfx::HalaBuffer::new(
      Rc::clone(&resources.context.borrow().logical_device),
      std::mem::size_of::<HalaGlobalUniform>() as u64,
      hala_gfx::HalaBufferUsageFlags::UNIFORM_BUFFER,
      hala_gfx::HalaMemoryLocation::CpuToGpu,
      "global.uniform_buffer",
    )?;

    // Return the renderer.
    log::debug!("A HalaRenderer \"{}\"[{} x {}] is created.", name, width, height);
    Ok(Self {
      info: HalaRendererInfo::new(name, width, height),
      use_mesh_shader: gpu_req.require_mesh_shader,

      resources: std::mem::ManuallyDrop::new(resources),

      use_deferred: false,
      depth_image: None,
      albedo_image: None,
      normal_image: None,

      static_descriptor_set: std::mem::ManuallyDrop::new(static_descriptor_set),
      dynamic_descriptor_set: None,
      global_uniform_buffer: std::mem::ManuallyDrop::new(global_uniform_buffer),
      object_uniform_buffers: Vec::new(),

      traditional_shaders: Vec::new(),
      shaders: Vec::new(),
      compute_shaders: Vec::new(),

      scene_in_gpu: None,

      forward_graphics_pipelines: Vec::new(),
      deferred_graphics_pipelines: Vec::new(),
      compute_pipelines: Vec::new(),

      textures_descriptor_set: None,

      data: HalaRendererData::new(),
      statistics: HalaRendererStatistics::new(),
    })
  }

  /// Draw the scene.
  /// param index: The index of the current image.
  /// param command_buffers: The command buffers.
  /// return: The result.
  fn draw_scene(&self, index: usize, command_buffers: &hala_gfx::HalaCommandBufferSet, is_forward: bool) -> Result<(), HalaRendererError> {
    command_buffers.set_viewport(
      index,
      0,
      &[
        (
          0.,
          self.info.height as f32,
          self.info.width as f32,
          -(self.info.height as f32), // For vulkan y is down.
          0.,
          1.
        ),
      ],
    );

    // Render the scene.
    let mut primitive_index = 0u32;
    let scene = self.scene_in_gpu.as_ref().ok_or(hala_gfx::HalaGfxError::new("The scene in GPU is none!", None))?;
    for (mesh_index, mesh) in scene.meshes.iter().enumerate() {
      for primitive in mesh.primitives.iter() {
        let material_type = scene.material_types[primitive.material_index as usize] as usize;
        if material_type >= scene.materials.len() {
          return Err(HalaRendererError::new("The material type index is out of range!", None));
        }
        let material_deferred = scene.material_deferred_flags[primitive.material_index as usize];

        if !material_deferred == is_forward {
          // Build push constants.
          let dispatch_size_x = (primitive.meshlet_count + 32 - 1) / 32;  // 32 threads per task group.
          let mut push_constants = Vec::new();
          push_constants.extend_from_slice(&(mesh_index as u32).to_le_bytes());
          push_constants.extend_from_slice(&primitive.material_index.to_le_bytes());
          push_constants.extend_from_slice(&primitive_index.to_le_bytes());
          if self.use_mesh_shader {
            push_constants.extend_from_slice(&primitive.meshlet_count.to_le_bytes());
            push_constants.extend_from_slice(&dispatch_size_x.to_le_bytes());
          }

          // Use specific material type pipeline state object.
          command_buffers.bind_graphics_pipeline(index, &self.forward_graphics_pipelines[material_type]);

          // Bind descriptor sets.
          command_buffers.bind_graphics_descriptor_sets(
            index,
            &self.forward_graphics_pipelines[material_type],
            0,
            &[
              self.static_descriptor_set.as_ref(),
              self.dynamic_descriptor_set.as_ref().ok_or(hala_gfx::HalaGfxError::new("The dynamic descriptor set is none!", None))?,
              self.textures_descriptor_set.as_ref().ok_or(hala_gfx::HalaGfxError::new("The textures descriptor set is none!", None))?],
            &[],
          );

          // Push constants.
          command_buffers.push_constants(
            index,
            self.forward_graphics_pipelines[material_type].layout,
            if !self.use_mesh_shader { hala_gfx::HalaShaderStageFlags::VERTEX } else { hala_gfx::HalaShaderStageFlags::TASK | hala_gfx::HalaShaderStageFlags::MESH }
              | hala_gfx::HalaShaderStageFlags::FRAGMENT,
            0,
            push_constants.as_slice(),
          );

          // Draw.
          if !self.use_mesh_shader {
            // Bind vertex buffers.
            command_buffers.bind_vertex_buffers(
              index,
              0,
              &[primitive.vertex_buffer.as_ref()],
              &[0]);

            // Bind index buffer.
            command_buffers.bind_index_buffers(
              index,
              &[primitive.index_buffer.as_ref()],
              &[0],
              hala_gfx::HalaIndexType::UINT32);

            command_buffers.draw_indexed(
              index,
              primitive.index_count,
              1,
              0,
              0,
              0
            );
          } else {
            command_buffers.draw_mesh_tasks(
              index,
              dispatch_size_x,
              1,
              1,
            );
          }
        }

        primitive_index += 1;
      }
    }

    Ok(())
  }

  /// Record the deferred rendering command buffer.
  /// param index: The index of the current image.
  /// param command_buffers: The command buffers.
  /// param ui_fn: The draw UI function.
  /// return: The result.
  fn record_deferred_command_buffer<F>(&self, index: usize, command_buffers: &hala_gfx::HalaCommandBufferSet, ui_fn: F) -> Result<(), HalaRendererError>
    where F: FnOnce(usize, &hala_gfx::HalaCommandBufferSet) -> Result<(), hala_gfx::HalaGfxError>
  {
    let context = self.resources.context.borrow();

    // Prepare the command buffer and timestamp.
    command_buffers.reset(index, false)?;
    command_buffers.begin(index, hala_gfx::HalaCommandBufferUsageFlags::empty())?;
    command_buffers.reset_query_pool(index, &context.timestamp_query_pool, (index * 2) as u32, 2);
    command_buffers.write_timestamp(index, hala_gfx::HalaPipelineStageFlags2::NONE, &context.timestamp_query_pool, (index * 2) as u32);

    if cfg!(debug_assertions) {
      command_buffers.begin_debug_label(index, "Draw", [1.0, 1.0, 1.0, 1.0]);
      command_buffers.begin_debug_label(index, "Draw G-Buffer", [1.0, 0.0, 0.0, 1.0]);
    }

    let depth_image = self.depth_image.as_ref().ok_or(HalaRendererError::new("The depth image is none!", None))?;
    let albedo_image = self.albedo_image.as_ref().ok_or(HalaRendererError::new("The albedo image is none!", None))?;
    let normal_image = self.normal_image.as_ref().ok_or(HalaRendererError::new("The normal image is none!", None))?;

    // Setup deferred G-buffer write barriers.
    command_buffers.set_image_barriers(
      index,
      &[
        hala_gfx::HalaImageBarrierInfo {
          old_layout: hala_gfx::HalaImageLayout::UNDEFINED,
          new_layout: hala_gfx::HalaImageLayout::COLOR_ATTACHMENT_OPTIMAL,
          src_access_mask: hala_gfx::HalaAccessFlags2::NONE,
          dst_access_mask: hala_gfx::HalaAccessFlags2::COLOR_ATTACHMENT_WRITE,
          src_stage_mask: hala_gfx::HalaPipelineStageFlags2::TOP_OF_PIPE,
          dst_stage_mask: hala_gfx::HalaPipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
          aspect_mask: hala_gfx::HalaImageAspectFlags::COLOR,
          image: albedo_image.raw,
          ..Default::default()
        },
        hala_gfx::HalaImageBarrierInfo {
          old_layout: hala_gfx::HalaImageLayout::UNDEFINED,
          new_layout: hala_gfx::HalaImageLayout::COLOR_ATTACHMENT_OPTIMAL,
          src_access_mask: hala_gfx::HalaAccessFlags2::NONE,
          dst_access_mask: hala_gfx::HalaAccessFlags2::COLOR_ATTACHMENT_WRITE,
          src_stage_mask: hala_gfx::HalaPipelineStageFlags2::TOP_OF_PIPE,
          dst_stage_mask: hala_gfx::HalaPipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
          aspect_mask: hala_gfx::HalaImageAspectFlags::COLOR,
          image: normal_image.raw,
          ..Default::default()
        },
        hala_gfx::HalaImageBarrierInfo {
          old_layout: hala_gfx::HalaImageLayout::UNDEFINED,
          new_layout: hala_gfx::HalaImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
          src_access_mask: hala_gfx::HalaAccessFlags2::NONE,
          dst_access_mask: hala_gfx::HalaAccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
          src_stage_mask: hala_gfx::HalaPipelineStageFlags2::TOP_OF_PIPE,
          dst_stage_mask: hala_gfx::HalaPipelineStageFlags2::EARLY_FRAGMENT_TESTS | hala_gfx::HalaPipelineStageFlags2::LATE_FRAGMENT_TESTS,
          aspect_mask: hala_gfx::HalaImageAspectFlags::DEPTH,
          image: depth_image.raw,
          ..Default::default()
        },
      ],
    );

    command_buffers.begin_rendering_with_rt(
      index,
      &[albedo_image, normal_image],
      Some(depth_image),
      (0, 0, self.info.width, self.info.height),
      &[Some([0.0, 0.0, 0.0, 1.0]), Some([0.0, 0.0, 0.0, 1.0])],
      Some(0.0),
      None,
    );

    self.draw_scene(index, command_buffers, false)?;

    command_buffers.end_rendering(index);

    // Setup deferred G-buffer read barriers.
    command_buffers.set_image_barriers(
      index,
      &[
        hala_gfx::HalaImageBarrierInfo {
          old_layout: hala_gfx::HalaImageLayout::COLOR_ATTACHMENT_OPTIMAL,
          new_layout: hala_gfx::HalaImageLayout::SHADER_READ_ONLY_OPTIMAL,
          src_access_mask: hala_gfx::HalaAccessFlags2::COLOR_ATTACHMENT_WRITE,
          dst_access_mask: hala_gfx::HalaAccessFlags2::INPUT_ATTACHMENT_READ,
          src_stage_mask: hala_gfx::HalaPipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
          dst_stage_mask: hala_gfx::HalaPipelineStageFlags2::FRAGMENT_SHADER,
          aspect_mask: hala_gfx::HalaImageAspectFlags::COLOR,
          image: albedo_image.raw,
          ..Default::default()
        },
        hala_gfx::HalaImageBarrierInfo {
          old_layout: hala_gfx::HalaImageLayout::COLOR_ATTACHMENT_OPTIMAL,
          new_layout: hala_gfx::HalaImageLayout::SHADER_READ_ONLY_OPTIMAL,
          src_access_mask: hala_gfx::HalaAccessFlags2::COLOR_ATTACHMENT_WRITE,
          dst_access_mask: hala_gfx::HalaAccessFlags2::INPUT_ATTACHMENT_READ,
          src_stage_mask: hala_gfx::HalaPipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
          dst_stage_mask: hala_gfx::HalaPipelineStageFlags2::FRAGMENT_SHADER,
          aspect_mask: hala_gfx::HalaImageAspectFlags::COLOR,
          image: normal_image.raw,
          ..Default::default()
        },
        hala_gfx::HalaImageBarrierInfo {
          old_layout: hala_gfx::HalaImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
          new_layout: hala_gfx::HalaImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
          src_access_mask: hala_gfx::HalaAccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
          dst_access_mask: hala_gfx::HalaAccessFlags2::INPUT_ATTACHMENT_READ,
          src_stage_mask: hala_gfx::HalaPipelineStageFlags2::EARLY_FRAGMENT_TESTS | hala_gfx::HalaPipelineStageFlags2::LATE_FRAGMENT_TESTS,
          dst_stage_mask: hala_gfx::HalaPipelineStageFlags2::FRAGMENT_SHADER,
          aspect_mask: hala_gfx::HalaImageAspectFlags::DEPTH,
          image: depth_image.raw,
          ..Default::default()
        },
      ],
    );

    if cfg!(debug_assertions) {
      command_buffers.end_debug_label(index);
      command_buffers.begin_debug_label(index, "Lighting", [0.0, 1.0, 0.0, 1.0]);
    }

    // Setup swapchain barrier.
    command_buffers.set_swapchain_image_barrier(
      index,
      &context.swapchain,
      &hala_gfx::HalaImageBarrierInfo {
        old_layout: hala_gfx::HalaImageLayout::UNDEFINED,
        new_layout: hala_gfx::HalaImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        src_access_mask: hala_gfx::HalaAccessFlags2::NONE,
        dst_access_mask: hala_gfx::HalaAccessFlags2::COLOR_ATTACHMENT_WRITE,
        src_stage_mask: hala_gfx::HalaPipelineStageFlags2::TOP_OF_PIPE,
        dst_stage_mask: hala_gfx::HalaPipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        aspect_mask: hala_gfx::HalaImageAspectFlags::COLOR,
        ..Default::default()
      },
      &hala_gfx::HalaImageBarrierInfo {
        old_layout: hala_gfx::HalaImageLayout::UNDEFINED,
        new_layout: hala_gfx::HalaImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        src_access_mask: hala_gfx::HalaAccessFlags2::NONE,
        dst_access_mask: hala_gfx::HalaAccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
        src_stage_mask: hala_gfx::HalaPipelineStageFlags2::EARLY_FRAGMENT_TESTS | hala_gfx::HalaPipelineStageFlags2::LATE_FRAGMENT_TESTS,
        dst_stage_mask: hala_gfx::HalaPipelineStageFlags2::EARLY_FRAGMENT_TESTS | hala_gfx::HalaPipelineStageFlags2::LATE_FRAGMENT_TESTS,
        aspect_mask: hala_gfx::HalaImageAspectFlags::DEPTH | if context.swapchain.has_stencil { hala_gfx::HalaImageAspectFlags::STENCIL } else { hala_gfx::HalaImageAspectFlags::empty() },
        ..Default::default()
      }
    );

    // Rendering.
    command_buffers.begin_rendering(
      index,
      &context.swapchain,
      (0, 0, self.info.width, self.info.height),
      Some([64.0 / 255.0, 46.0 / 255.0, 122.0 / 255.0, 1.0]),
      Some(0.0),
      Some(0),
    );

    // Setup viewport.
    command_buffers.set_viewport(
      index,
      0,
      &[
        (
          0.,
          self.info.height as f32,
          self.info.width as f32,
          -(self.info.height as f32), // For vulkan y is down.
          0.,
          1.
        ),
      ],
    );

    // Draw UI.
    if cfg!(debug_assertions) {
      command_buffers.end_debug_label(index);
      command_buffers.begin_debug_label(index, "Draw UI", [0.0, 0.0, 1.0, 1.0]);
    }
    ui_fn(index, command_buffers)?;
    if cfg!(debug_assertions) {
      command_buffers.end_debug_label(index);
    }

    command_buffers.end_rendering(index);

    // Setup swapchain barrier.
    command_buffers.set_image_barriers(
      index,
      &[hala_gfx::HalaImageBarrierInfo {
        old_layout: hala_gfx::HalaImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        new_layout: hala_gfx::HalaImageLayout::PRESENT_SRC,
        src_access_mask: hala_gfx::HalaAccessFlags2::COLOR_ATTACHMENT_WRITE,
        dst_access_mask: hala_gfx::HalaAccessFlags2::NONE,
        src_stage_mask: hala_gfx::HalaPipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        dst_stage_mask: hala_gfx::HalaPipelineStageFlags2::BOTTOM_OF_PIPE,
        aspect_mask: hala_gfx::HalaImageAspectFlags::COLOR,
        image: context.swapchain.images[index],
        ..Default::default()
      }],
    );
    if cfg!(debug_assertions) {
      command_buffers.end_debug_label(index);
    }

    // Write end timestamp and end command buffer.
    command_buffers.write_timestamp(
      index,
      hala_gfx::HalaPipelineStageFlags2::ALL_COMMANDS,
      &context.timestamp_query_pool,
      (index * 2 + 1) as u32);
    command_buffers.end(index)?;

    Ok(())
  }

  /// Create G-buffer images.
  /// param use_transient: Use transient images or not.
  /// param albedo_format: The format of the albedo image.
  /// param normal_format: The format of the normal image.
  /// return: The result.
  pub fn create_gbuffer_images(
    &mut self,
    use_transient: bool,
    albedo_format: hala_gfx::HalaFormat,
    normal_format: hala_gfx::HalaFormat,
  ) -> Result<(), HalaRendererError> {
    let rt_usage_flags = if use_transient {
      hala_gfx::HalaImageUsageFlags::INPUT_ATTACHMENT | hala_gfx::HalaImageUsageFlags::TRANSIENT_ATTACHMENT
    } else {
      hala_gfx::HalaImageUsageFlags::INPUT_ATTACHMENT
    };

    // Create depth image.
    let depth_image = hala_gfx::HalaImage::new_2d(
      Rc::clone(&self.resources.context.borrow().logical_device),
      hala_gfx::HalaImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | rt_usage_flags,
      hala_gfx::HalaFormat::D32_SFLOAT,
      self.info.width,
      self.info.height,
      1,
      1,
      hala_gfx::HalaMemoryLocation::GpuOnly,
      "depth.image",
    )?;

    // Create albedo image.
    let albedo_image = hala_gfx::HalaImage::new_2d(
      Rc::clone(&self.resources.context.borrow().logical_device),
      hala_gfx::HalaImageUsageFlags::COLOR_ATTACHMENT | rt_usage_flags,
      albedo_format,
      self.info.width,
      self.info.height,
      1,
      1,
      hala_gfx::HalaMemoryLocation::GpuOnly,
      "albedo.image",
    )?;

    // Create normal image.
    let normal_image = hala_gfx::HalaImage::new_2d(
      Rc::clone(&self.resources.context.borrow().logical_device),
      hala_gfx::HalaImageUsageFlags::COLOR_ATTACHMENT | rt_usage_flags,
      normal_format,
      self.info.width,
      self.info.height,
      1,
      1,
      hala_gfx::HalaMemoryLocation::GpuOnly,
      "normal.image",
    )?;

    self.use_deferred = true;
    self.depth_image = Some(depth_image);
    self.albedo_image = Some(albedo_image);
    self.normal_image = Some(normal_image);

    Ok(())
  }

  /// Push traditional shaders to the renderer.
  /// param vertex_file_path: The vertex shader file path.
  /// param fragment_file_path: The fragment shader file path.
  /// param debug_name: The debug name of the shader.
  /// return: The result.
  pub fn push_traditional_shaders_with_file(
    &mut self,
    vertex_file_path: &str,
    fragment_file_path: &str,
    debug_name: &str) -> Result<(), HalaRendererError>
  {
    assert!(!self.use_mesh_shader, "The renderer is not support mesh shader!");

    let context = self.resources.context.borrow();

    let vertex_shader = hala_gfx::HalaShader::with_file(
      Rc::clone(&context.logical_device),
      vertex_file_path,
      hala_gfx::HalaShaderStageFlags::VERTEX,
      hala_gfx::HalaRayTracingShaderGroupType::GENERAL,
      &format!("{}.vert", debug_name),
    )?;

    let fragment_shader = hala_gfx::HalaShader::with_file(
      Rc::clone(&context.logical_device),
      fragment_file_path,
      hala_gfx::HalaShaderStageFlags::FRAGMENT,
      hala_gfx::HalaRayTracingShaderGroupType::GENERAL,
      &format!("{}.frag", debug_name),
    )?;

    self.traditional_shaders.push((vertex_shader, fragment_shader));

    Ok(())
  }

  /// Push shaders to the renderer.
  /// param task_file_path: The task shader file path.
  /// param mesh_file_path: The mesh shader file path.
  /// param fragment_file_path: The fragment shader file path.
  /// param debug_name: The debug name of the shader.
  /// return: The result.
  pub fn push_shaders_with_file(
    &mut self,
    task_file_path: Option<&str>,
    mesh_file_path: &str,
    fragment_file_path: &str,
    debug_name: &str) -> Result<(), HalaRendererError>
  {
    assert!(self.use_mesh_shader, "The renderer is not support traditional shader!");

    let context = self.resources.context.borrow();

    let task_shader = match task_file_path {
      Some(file_path) => Some(hala_gfx::HalaShader::with_file(
        Rc::clone(&context.logical_device),
        file_path,
        hala_gfx::HalaShaderStageFlags::TASK,
        hala_gfx::HalaRayTracingShaderGroupType::GENERAL,
        &format!("{}.task", debug_name),
      )?),
      None => None,
    };

    let mesh_shader = hala_gfx::HalaShader::with_file(
      Rc::clone(&context.logical_device),
      mesh_file_path,
      hala_gfx::HalaShaderStageFlags::MESH,
      hala_gfx::HalaRayTracingShaderGroupType::GENERAL,
      &format!("{}.mesh", debug_name),
    )?;

    let fragment_shader = hala_gfx::HalaShader::with_file(
      Rc::clone(&context.logical_device),
      fragment_file_path,
      hala_gfx::HalaShaderStageFlags::FRAGMENT,
      hala_gfx::HalaRayTracingShaderGroupType::GENERAL,
      &format!("{}.frag", debug_name),
    )?;

    self.shaders.push((task_shader, mesh_shader, fragment_shader));

    Ok(())
  }

  /// Push compute shaders to the renderer.
  /// param file_path: The compute shader file path.
  /// param debug_name: The debug name of the shader.
  /// return: The result.
  pub fn push_compute_shaders_with_file(
    &mut self,
    file_path: &str,
    debug_name: &str) -> Result<(), HalaRendererError>
  {
    let context = self.resources.context.borrow();

    let compute_shader = hala_gfx::HalaShader::with_file(
      Rc::clone(&context.logical_device),
      file_path,
      hala_gfx::HalaShaderStageFlags::COMPUTE,
      hala_gfx::HalaRayTracingShaderGroupType::GENERAL,
      &format!("{}.comp", debug_name),
    )?;

    self.compute_shaders.push(compute_shader);

    Ok(())
  }

  /// Set the scene to be rendered.
  /// param scene_in_cpu: The scene in the CPU.
  /// return: The result.
  pub fn set_scene(&mut self, scene_in_cpu: &mut cpu::HalaScene) -> Result<(), HalaRendererError> {
    let context = self.resources.context.borrow();
    // Release the old scene in the GPU.
    self.scene_in_gpu = None;

    // Upload the new scene to the GPU.
    let scene_in_gpu = loader::HalaSceneGPUUploader::upload(
      &context,
      &self.resources.graphics_command_buffers,
      &self.resources.transfer_command_buffers,
      scene_in_cpu,
      self.use_mesh_shader,
    false)?;

    self.scene_in_gpu = Some(scene_in_gpu);

    Ok(())
  }

}