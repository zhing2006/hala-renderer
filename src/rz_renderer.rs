use std::rc::Rc;

use hala_gfx::HalaGPURequirements;

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

    // Update global uniform buffer(Only use No.1 camera).
    self.global_uniform_buffer.update_memory(0, &[HalaGlobalUniform {
      v_mtx: scene.camera_view_matrices[0],
      p_mtx: scene.camera_proj_matrices[0],
      vp_mtx: scene.camera_proj_matrices[0] * scene.camera_view_matrices[0],
    }])?;

    // Create dynamic descriptor set.
    let dynamic_descriptor_set = hala_gfx::HalaDescriptorSet::new(
      Rc::clone(&context.logical_device),
      Rc::clone(&self.resources.descriptor_pool),
      hala_gfx::HalaDescriptorSetLayout::new(
        Rc::clone(&context.logical_device),
        &[
          ( // Materials uniform buffer.
            0,
            hala_gfx::HalaDescriptorType::UNIFORM_BUFFER,
            scene.materials.len() as u32,
            hala_gfx::HalaShaderStageFlags::VERTEX | hala_gfx::HalaShaderStageFlags::FRAGMENT | hala_gfx::HalaShaderStageFlags::COMPUTE,
            hala_gfx::HalaDescriptorBindingFlags::PARTIALLY_BOUND
          ),
          ( // Object uniform buffer.
            1,
            hala_gfx::HalaDescriptorType::UNIFORM_BUFFER,
            scene.meshes.len() as u32,
            hala_gfx::HalaShaderStageFlags::VERTEX | hala_gfx::HalaShaderStageFlags::FRAGMENT | hala_gfx::HalaShaderStageFlags::COMPUTE,
            hala_gfx::HalaDescriptorBindingFlags::PARTIALLY_BOUND
          ),
        ],
        "main_dynamic.descriptor_set_layout",
      )?,
      context.swapchain.num_of_images,
      0,
      "main_dynamic.descriptor_set",
    )?;

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

        buffer.update_memory(0, &[object_uniform])?;

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
          ( // All textures in the scene.
            0,
            hala_gfx::HalaDescriptorType::SAMPLED_IMAGE,
            scene
              .textures.len() as u32,
            hala_gfx::HalaShaderStageFlags::VERTEX | hala_gfx::HalaShaderStageFlags::FRAGMENT | hala_gfx::HalaShaderStageFlags::COMPUTE,
            hala_gfx::HalaDescriptorBindingFlags::PARTIALLY_BOUND
          ),
          (
            1,
            hala_gfx::HalaDescriptorType::SAMPLER,
            scene
              .textures.len() as u32,
            hala_gfx::HalaShaderStageFlags::VERTEX | hala_gfx::HalaShaderStageFlags::FRAGMENT | hala_gfx::HalaShaderStageFlags::COMPUTE,
            hala_gfx::HalaDescriptorBindingFlags::PARTIALLY_BOUND
          ),
        ],
        "textures.descriptor_set_layout",
      )?,
      1,
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

    // Create traditional graphics pipelines.
    for (i, (vertex_shader, fragment_shader)) in self.traditional_shaders.iter().enumerate() {
      self.pso.push(
        hala_gfx::HalaGraphicsPipeline::new(
          Rc::clone(&context.logical_device),
          &context.swapchain,
          &[&self.static_descriptor_set.layout, &dynamic_descriptor_set.layout, &textures_descriptor_set.layout],
          &[
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
          ],
          &[
            hala_gfx::HalaVertexInputBindingDescription {
              binding: 0,
              stride: 44,
              input_rate: hala_gfx::HalaVertexInputRate::VERTEX,
            }
          ],
          &[
            hala_gfx::HalaPushConstantRange {
              stage_flags: hala_gfx::HalaShaderStageFlags::VERTEX | hala_gfx::HalaShaderStageFlags::FRAGMENT,
              offset: 0,
              size: 8,  // Mesh index, Material index.
            },
          ],
          hala_gfx::HalaPrimitiveTopology::TRIANGLE_LIST,
          (hala_gfx::HalaBlendFactor::SRC_ALPHA, hala_gfx::HalaBlendFactor::ONE_MINUS_SRC_ALPHA, hala_gfx::HalaBlendOp::ADD),
          (hala_gfx::HalaBlendFactor::ONE, hala_gfx::HalaBlendFactor::ZERO, hala_gfx::HalaBlendOp::ADD),
          (1.0, hala_gfx::HalaFrontFace::COUNTER_CLOCKWISE, hala_gfx::HalaCullModeFlags::BACK, hala_gfx::HalaPolygonMode::FILL),
          (true, true, hala_gfx::HalaCompareOp::GREATER), // We use reverse Z, so greater is less.
          &[vertex_shader, fragment_shader],
          &[hala_gfx::HalaDynamicState::VIEWPORT],
          Some(&pipeline_cache),
          &format!("traditional_{}.graphics_pipeline", i),
        )?
      );
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
    let context = self.resources.context.borrow();

    // Update the renderer.
    self.data.image_index = context.prepare_frame()?;
    context.record_graphics_command_buffer(
      self.data.image_index,
      &self.resources.graphics_command_buffers,
      Some(([25.0 / 255.0, 118.0 / 255.0, 210.0 / 255.0, 1.0], 0.0, 0)),  // We use reverse Z, so clear depth to 0.0.
      |index, command_buffers| {
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
        let scene = self.scene_in_gpu.as_ref().ok_or(hala_gfx::HalaGfxError::new("The scene in GPU is none!", None))?;
        for (mesh_index, mesh) in scene.meshes.iter().enumerate() {
          for primitive in mesh.primitives.iter() {
            let material_type = scene.material_types[primitive.material_index as usize] as usize;
            if material_type >= scene.materials.len() {
              return Err(hala_gfx::HalaGfxError::new("The material type index is out of range!", None));
            }

            // Use specific material type pipeline state object.
            command_buffers.bind_graphics_pipeline(index, &self.pso[material_type]);

            // Bind descriptor sets.
            command_buffers.bind_graphics_descriptor_sets(
              index,
              &self.pso[material_type],
              0,
              &[
                self.static_descriptor_set.as_ref(),
                self.dynamic_descriptor_set.as_ref().ok_or(hala_gfx::HalaGfxError::new("The dynamic descriptor set is none!", None))?,
                self.textures_descriptor_set.as_ref().ok_or(hala_gfx::HalaGfxError::new("The textures descriptor set is none!", None))?],
              &[],
            );

            // Push constants.
            let mut push_constants = Vec::new();
            push_constants.extend_from_slice(&(mesh_index as u32).to_le_bytes());
            push_constants.extend_from_slice(&primitive.material_index.to_le_bytes());
            command_buffers.push_constants(
              index,
              &self.pso[material_type],
              hala_gfx::HalaShaderStageFlags::VERTEX | hala_gfx::HalaShaderStageFlags::FRAGMENT,
              0,
              push_constants.as_slice(),
            );

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

            // Draw.
            command_buffers.draw_indexed(
              index,
              primitive.index_count,
              1,
              0,
              0,
              0);
          }
        }

        ui_fn(index, command_buffers)?;

        Ok(())
      },
      None,
      |_, _| {
        Ok(false)
      },
    )?;

    Ok(())
  }

}

/// The rasterization renderer.
/// NOTICE: Only support object movement every frame, camera and light can NOT be moved.
pub struct HalaRenderer {

  pub(crate) info: HalaRendererInfo,

  pub(crate) use_mesh_shader: bool,

  pub(crate) resources: std::mem::ManuallyDrop<HalaRendererResources>,

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

  pub(crate) pso: Vec<hala_gfx::HalaGraphicsPipeline>,
  pub(crate) textures_descriptor_set: Option<hala_gfx::HalaDescriptorSet>,

  pub(crate) data: HalaRendererData,
  pub(crate) statistics: HalaRendererStatistics,

}

/// The Drop implementation of the renderer.
impl Drop for HalaRenderer {

  fn drop(&mut self) {
    self.textures_descriptor_set = None;
    self.pso.clear();

    self.scene_in_gpu = None;

    self.traditional_shaders.clear();
    self.shaders.clear();
    self.compute_shaders.clear();

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
          ( // Global uniform buffer.
            0,
            hala_gfx::HalaDescriptorType::UNIFORM_BUFFER,
            1,
            hala_gfx::HalaShaderStageFlags::VERTEX | hala_gfx::HalaShaderStageFlags::FRAGMENT | hala_gfx::HalaShaderStageFlags::COMPUTE,
            hala_gfx::HalaDescriptorBindingFlags::PARTIALLY_BOUND
          ),
          ( // Cameras uniform buffer.
            1,
            hala_gfx::HalaDescriptorType::UNIFORM_BUFFER,
            1,
            hala_gfx::HalaShaderStageFlags::VERTEX | hala_gfx::HalaShaderStageFlags::FRAGMENT | hala_gfx::HalaShaderStageFlags::COMPUTE,
            hala_gfx::HalaDescriptorBindingFlags::PARTIALLY_BOUND
          ),
          ( // Lights uniform buffer.
            2,
            hala_gfx::HalaDescriptorType::UNIFORM_BUFFER,
            1,
            hala_gfx::HalaShaderStageFlags::VERTEX | hala_gfx::HalaShaderStageFlags::FRAGMENT | hala_gfx::HalaShaderStageFlags::COMPUTE,
            hala_gfx::HalaDescriptorBindingFlags::PARTIALLY_BOUND
          ),
        ],
        "main_static.descriptor_set_layout",
      )?,
      1,
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

      static_descriptor_set: std::mem::ManuallyDrop::new(static_descriptor_set),
      dynamic_descriptor_set: None,
      global_uniform_buffer: std::mem::ManuallyDrop::new(global_uniform_buffer),
      object_uniform_buffers: Vec::new(),

      traditional_shaders: Vec::new(),
      shaders: Vec::new(),
      compute_shaders: Vec::new(),

      scene_in_gpu: None,

      pso: Vec::new(),
      textures_descriptor_set: None,

      data: HalaRendererData::new(),
      statistics: HalaRendererStatistics::new(),
    })
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
  pub fn set_scene(&mut self, scene_in_cpu: &cpu::HalaScene) -> Result<(), HalaRendererError> {
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