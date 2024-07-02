use std::rc::Rc;

use glam::Vec4Swizzles;

use hala_gfx::{
  HalaContext,
  HalaBuffer,
  HalaBufferUsageFlags,
  HalaMemoryLocation,
  HalaFilter,
  HalaSampler,
  HalaSamplerMipmapMode,
  HalaSamplerAddressMode,
  HalaImage,
  HalaImageUsageFlags,
  HalaAccelerationStructureLevel,
  HalaAccelerationStructureGeometry,
  HalaAccelerationStructureGeometryTrianglesData,
  HalaAccelerationStructureGeometryAabbsData,
  HalaAccelerationStructure,
  HalaAccelerationStructureInstance,
  HalaAccelerationStructureGeometryInstancesData,
  HalaAccelerationStructureBuildRangeInfo,
  HalaCommandBufferSet,
  HalaAABB,
};

use crate::{
  error::HalaRendererError,
  scene::{
    HalaVertex,
    HalaMeshlet,
  },
};
use super::super::cpu;
use super::super::gpu;

const MAX_CAMERA_COUNT: usize = 8;
const MAX_LIGHT_COUNT: usize = 32;

/// Upload the scene to the GPU from the CPU.
pub struct HalaSceneGPUUploader;

/// The implementation of the scene uploader.
impl HalaSceneGPUUploader {
  /// Upload the scene to the GPU from the CPU for rasterization.
  /// param context: The gfx context.
  /// param graphics_command_buffers: The graphics command buffers.
  /// param transfer_command_buffers: The transfer command buffers.
  /// param scene_in_cpu: The scene in the CPU.
  /// param use_for_mesh_shader: Whether the scene is used for mesh shader.
  /// param use_for_ray_tracing: Whether the scene is used for ray tracing.
  /// return: The scene in the GPU.
  pub fn upload(
    context: &HalaContext,
    graphics_command_buffers: &HalaCommandBufferSet,
    transfer_command_buffers: &HalaCommandBufferSet,
    scene_in_cpu: &mut cpu::HalaScene,
    use_for_mesh_shader: bool,
    use_for_ray_tracing: bool,
  ) -> Result<gpu::HalaScene, HalaRendererError> {
    // Calculate the buffer size.
    let camera_buffer_size = (std::mem::size_of::<gpu::HalaCamera>() * MAX_CAMERA_COUNT) as u64;
    let light_buffer_size = (std::mem::size_of::<gpu::HalaLight>() * MAX_LIGHT_COUNT) as u64;
    let light_aabb_buffer_size = (std::mem::size_of::<HalaAABB>() * MAX_LIGHT_COUNT) as u64;
    let material_buffer_size = (std::mem::size_of::<gpu::HalaMaterial>()) as u64;

    let max_buffer_size = std::cmp::max(
      std::cmp::max(camera_buffer_size, light_buffer_size),
      material_buffer_size);

    // Create the staging buffer.
    let staging_buffer = HalaBuffer::new(
      Rc::clone(&context.logical_device),
      max_buffer_size,
      HalaBufferUsageFlags::TRANSFER_SRC,
      HalaMemoryLocation::CpuToGpu,
      "staging.buffer")?;

    // Create the camera buffer.
    let camera_buffer = HalaBuffer::new(
      Rc::clone(&context.logical_device),
      camera_buffer_size,
      HalaBufferUsageFlags::UNIFORM_BUFFER | HalaBufferUsageFlags::TRANSFER_DST,
      HalaMemoryLocation::GpuOnly,
      "cameras.buffer")?;

    // Copy the camera data to GPU by the staging buffer.
    if scene_in_cpu.cameras.len() > MAX_CAMERA_COUNT {
      log::warn!(
        "The camera count {} exceeds the maximum camera count {}.\nOnly the first {} cameras will be uploaded to the GPU.",
        scene_in_cpu.cameras.len(), MAX_CAMERA_COUNT, MAX_CAMERA_COUNT
      );
    }
    let mut camera_view_matrices = Vec::with_capacity(scene_in_cpu.cameras.len());
    let mut camera_proj_matrices = Vec::with_capacity(scene_in_cpu.cameras.len());
    let mut cameras = Vec::with_capacity(scene_in_cpu.cameras.len());
    for (index, camera) in scene_in_cpu.cameras.iter().enumerate() {
      if index >= MAX_CAMERA_COUNT {
        break;
      }
      let camera_node = scene_in_cpu.nodes.iter().find(|&node| node.camera_index == index as u32)
        .ok_or(HalaRendererError::new(&format!("The camera node of the camera {} is not found.", index), None))?;
      camera_view_matrices.push(camera_node.world_transform.inverse());
      camera_proj_matrices.push(camera.get_proj_matrix());
      cameras.push(gpu::HalaCamera::new(camera_node, camera));
    }
    camera_buffer.update_gpu_memory_with_buffer_raw(
      cameras.as_ptr() as *const u8,
      std::mem::size_of::<gpu::HalaCamera>() * cameras.len(),
      &staging_buffer,
      transfer_command_buffers)?;

    // Create the light buffer.
    let light_buffer = HalaBuffer::new(
      Rc::clone(&context.logical_device),
      light_buffer_size,
      HalaBufferUsageFlags::UNIFORM_BUFFER | HalaBufferUsageFlags::TRANSFER_DST,
      HalaMemoryLocation::GpuOnly,
      "lights.buffer")?;
    let light_aabb_buffer = HalaBuffer::new(
      Rc::clone(&context.logical_device),
      light_aabb_buffer_size,
      HalaBufferUsageFlags::STORAGE_BUFFER |
      HalaBufferUsageFlags::TRANSFER_DST |
      HalaBufferUsageFlags::SHADER_DEVICE_ADDRESS |
      (if use_for_ray_tracing { HalaBufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY } else { HalaBufferUsageFlags::default()}),
      HalaMemoryLocation::GpuOnly,
      "light_aabbs.buffer")?;

    // Copy the light data to GPU by the staging buffer.
    if scene_in_cpu.lights.len() > MAX_LIGHT_COUNT {
      log::warn!(
        "The light count {} exceeds the maximum light count {}.\nOnly the first {} lights will be uploaded to the GPU.",
        scene_in_cpu.lights.len(), MAX_LIGHT_COUNT, MAX_LIGHT_COUNT
      );
    }
    let mut lights = Vec::with_capacity(scene_in_cpu.lights.len());
    let mut light_aabbs = Vec::new();
    for node in scene_in_cpu.nodes.iter() {
      if node.light_index == u32::MAX {
        continue;
      }

      let light_index = node.light_index as usize;
      let light_in_cpu = &scene_in_cpu.lights[light_index];
      let (light, light_aabb) = match light_in_cpu.light_type {
        cpu::light::HalaLightType::POINT => {
          (
            gpu::HalaLight {
              intensity: (light_in_cpu.color * light_in_cpu.intensity).into(),
              position: node.world_transform.w_axis.xyz().into(),
              u: glam::Vec3A::ZERO,
              v: glam::Vec3::ZERO,
              radius: 0.0,
              area: 0.0,
              _type: 0,
            },
            HalaAABB {
              min: [
                node.world_transform.w_axis.x,
                node.world_transform.w_axis.y,
                node.world_transform.w_axis.z,
              ],
              max: [
                node.world_transform.w_axis.x,
                node.world_transform.w_axis.y,
                node.world_transform.w_axis.z,
              ],
            }
          )
        },
        cpu::light::HalaLightType::DIRECTIONAL => {
          (
            gpu::HalaLight {
              intensity: (light_in_cpu.color * light_in_cpu.intensity).into(),
              position: glam::Vec3A::ZERO,
              u: (-node.world_transform.z_axis.xyz()).into(),
              v: glam::Vec3::new((0.5 * light_in_cpu.params.0).cos(), 0.0, 0.0),
              radius: 0.0,
              area: 0.0,
              _type: 1,
            },
            HalaAABB {
              min: [0.0, 0.0, 0.0],
              max: [0.0, 0.0, 0.0],
            }
          )
        },
        cpu::light::HalaLightType::SPOT => {
          (
            gpu::HalaLight {
              intensity: (light_in_cpu.color * light_in_cpu.intensity).into(),
              position: node.world_transform.w_axis.xyz().into(),
              u: (-node.world_transform.z_axis.xyz()).into(),
              v: glam::Vec3::new(light_in_cpu.params.0.cos(), light_in_cpu.params.1.cos(), 0.0),
              radius: 0.0,
              area: 0.0,
              _type: 2,
            },
            HalaAABB {
              min: [
                node.world_transform.w_axis.x,
                node.world_transform.w_axis.y,
                node.world_transform.w_axis.z,
              ],
              max: [
                node.world_transform.w_axis.x,
                node.world_transform.w_axis.y,
                node.world_transform.w_axis.z,
              ],
            }
          )
        },
        cpu::light::HalaLightType::QUAD => {
          let mut position = node.world_transform.w_axis.xyz();
          position -= node.world_transform.x_axis.xyz() * light_in_cpu.params.0 * 0.5;
          position -= node.world_transform.y_axis.xyz() * light_in_cpu.params.1 * 0.5;
          let another = position + node.world_transform.x_axis.xyz() * light_in_cpu.params.0 + node.world_transform.y_axis.xyz() * light_in_cpu.params.1 + node.world_transform.z_axis.xyz() * 0.01;
          (
            gpu::HalaLight {
              intensity: (light_in_cpu.color * light_in_cpu.intensity).into(),
              position: position.into(),
              u: (node.world_transform.x_axis.xyz() * light_in_cpu.params.0).into(),
              v: node.world_transform.y_axis.xyz() * light_in_cpu.params.1,
              radius: 0.0,
              area: light_in_cpu.params.0 * light_in_cpu.params.1,
              _type: 3,
            },
            HalaAABB {
              min: [
                position.x,
                position.y,
                position.z,
              ],
              max: [
                another.x,
                another.y,
                another.z,
              ],
            }
          )
        },
        cpu::light::HalaLightType::SPHERE => {
          let min = node.world_transform.w_axis.xyz() - glam::Vec3::splat(light_in_cpu.params.0);
          let max = node.world_transform.w_axis.xyz() + glam::Vec3::splat(light_in_cpu.params.0);
          (
            gpu::HalaLight {
              intensity: (light_in_cpu.color * light_in_cpu.intensity).into(),
              position: node.world_transform.w_axis.xyz().into(),
              u: glam::Vec3A::ZERO,
              v: glam::Vec3::ZERO,
              radius: light_in_cpu.params.0,
              area: 4.0 * std::f32::consts::PI * light_in_cpu.params.0 * light_in_cpu.params.0,
              _type: 4,
            },
            HalaAABB {
              min: [min.x, min.y, min.z],
              max: [max.x, max.y, max.z],
            }
          )
        },
        _ => panic!("Invalid light type."),
      };

      let min_x = f32::min(light_aabb.min[0], light_aabb.max[0]);
      let min_y = f32::min(light_aabb.min[1], light_aabb.max[1]);
      let min_z = f32::min(light_aabb.min[2], light_aabb.max[2]);
      let max_x = f32::max(light_aabb.min[0], light_aabb.max[0]);
      let max_y = f32::max(light_aabb.min[1], light_aabb.max[1]);
      let max_z = f32::max(light_aabb.min[2], light_aabb.max[2]);
      lights.push(light);
      light_aabbs.push(
        HalaAABB {
          min: [min_x, min_y, min_z],
          max: [max_x, max_y, max_z],
        }
      );

      if lights.len() >= MAX_LIGHT_COUNT {
        break;
      }
    }
    light_buffer.update_gpu_memory_with_buffer_raw(
      lights.as_ptr() as *const u8,
      std::mem::size_of::<gpu::HalaLight>() * lights.len(),
      &staging_buffer,
      transfer_command_buffers)?;
    light_aabb_buffer.update_gpu_memory_with_buffer_raw(
      light_aabbs.as_ptr() as *const u8,
      std::mem::size_of::<HalaAABB>() * light_aabbs.len(),
      &staging_buffer,
      transfer_command_buffers)?;

    // Create the material buffers.
    let mut material_buffers = Vec::with_capacity(scene_in_cpu.materials.len());
    let mut material_types = Vec::with_capacity(scene_in_cpu.materials.len());

    // Copy the material data to GPU by the staging buffer.
    for (material_index, material) in scene_in_cpu.materials.iter().enumerate() {
      let gpu_material = gpu::HalaMaterial::from(material);

      let material_buffer = HalaBuffer::new(
        Rc::clone(&context.logical_device),
        material_buffer_size,
        HalaBufferUsageFlags::UNIFORM_BUFFER | HalaBufferUsageFlags::TRANSFER_DST,
        HalaMemoryLocation::GpuOnly,
        &format!("material_{}.buffer", material_index)
      )?;

      material_buffer.update_gpu_memory_with_buffer_raw(
        &gpu_material as *const gpu::HalaMaterial as *const u8,
        material_buffer_size as usize,
        &staging_buffer,
        transfer_command_buffers)?;

      material_buffers.push(material_buffer);
      material_types.push(gpu_material._type);
    }

    // Create the samplers and images.
    let mut samplers = Vec::with_capacity(scene_in_cpu.texture2image_mapping.len());
    let mut textures = Vec::with_capacity(scene_in_cpu.texture2image_mapping.len());
    for (index, image_index) in scene_in_cpu.texture2image_mapping.iter() {
      let data_index = scene_in_cpu.image2data_mapping.get(image_index).ok_or(HalaRendererError::new(&format!("The image {} is not found.", image_index), None))?;
      textures.push(*data_index);

      samplers.push(
        HalaSampler::new(
          Rc::clone(&context.logical_device),
          (HalaFilter::LINEAR, HalaFilter::LINEAR),
          HalaSamplerMipmapMode::LINEAR,
          (HalaSamplerAddressMode::REPEAT, HalaSamplerAddressMode::REPEAT, HalaSamplerAddressMode::REPEAT),
          0.0,
          false,
          0.0,
          (0.0, 0.0),
          &format!("texture_{}.sampler", index)
        )?
      );
    }

    let mut images = Vec::with_capacity(scene_in_cpu.image_data.len());
    let max_texture_size = scene_in_cpu.image_data.iter().map(|texture| texture.num_of_bytes).max().unwrap_or(0);
    if max_texture_size > 0 {
      let image_staging = HalaBuffer::new(
        Rc::clone(&context.logical_device),
        max_texture_size as u64,
        HalaBufferUsageFlags::TRANSFER_SRC,
        HalaMemoryLocation::CpuToGpu,
        "image_staging.buffer")?;
      for (index, texture) in scene_in_cpu.image_data.iter().enumerate() {
        let image = HalaImage::new_2d(
          Rc::clone(&context.logical_device),
          HalaImageUsageFlags::SAMPLED | HalaImageUsageFlags::TRANSFER_DST,
          texture.format,
          texture.width,
          texture.height,
          1,
          1,
          HalaMemoryLocation::GpuOnly,
          &format!("texture_{}.image", index)
        )?;
        match texture.data_type {
          cpu::image_data::HalaImageDataType::ByteData(ref data) => {
            image.update_gpu_memory_with_buffer(
              data.as_slice(),
              (if use_for_mesh_shader { hala_gfx::HalaPipelineStageFlags2::TASK_SHADER_EXT | hala_gfx::HalaPipelineStageFlags2::MESH_SHADER_EXT } else { hala_gfx::HalaPipelineStageFlags2::default() })
                | (if use_for_ray_tracing { hala_gfx::HalaPipelineStageFlags2::RAY_TRACING_SHADER } else { hala_gfx::HalaPipelineStageFlags2::default() })
                | hala_gfx::HalaPipelineStageFlags2::VERTEX_SHADER | hala_gfx::HalaPipelineStageFlags2::FRAGMENT_SHADER
                | hala_gfx::HalaPipelineStageFlags2::COMPUTE_SHADER
                | hala_gfx::HalaPipelineStageFlags2::TRANSFER,
              &image_staging,
              graphics_command_buffers)?;
          },
          cpu::image_data::HalaImageDataType::FloatData(ref data) => {
            image.update_gpu_memory_with_buffer(
              data.as_slice(),
              (if use_for_mesh_shader { hala_gfx::HalaPipelineStageFlags2::TASK_SHADER_EXT | hala_gfx::HalaPipelineStageFlags2::MESH_SHADER_EXT } else { hala_gfx::HalaPipelineStageFlags2::default() })
                | (if use_for_ray_tracing { hala_gfx::HalaPipelineStageFlags2::RAY_TRACING_SHADER } else { hala_gfx::HalaPipelineStageFlags2::default() })
                | hala_gfx::HalaPipelineStageFlags2::VERTEX_SHADER | hala_gfx::HalaPipelineStageFlags2::FRAGMENT_SHADER
                | hala_gfx::HalaPipelineStageFlags2::COMPUTE_SHADER
                | hala_gfx::HalaPipelineStageFlags2::TRANSFER,
              &image_staging,
              graphics_command_buffers)?;
          }
        };
        images.push(image);
      }
    }

    // Create the meshes.
    let mut meshes = Vec::with_capacity(scene_in_cpu.meshes.len());
    let vertex_size = std::mem::size_of::<HalaVertex>();
    let max_vertex_buffer_size = scene_in_cpu.meshes.iter().map(
      |mesh| mesh.primitives.iter().map(|prim| prim.vertices.len() * vertex_size).max().unwrap_or(0)
    ).max().unwrap_or(0);
    let max_index_buffer_size = scene_in_cpu.meshes.iter().map(
      |mesh| mesh.primitives.iter().map(|prim| prim.indices.len() * std::mem::size_of::<u32>()).max().unwrap_or(0)
    ).max().unwrap_or(0);
    let mesh_staging_buffer_size = std::cmp::max(max_vertex_buffer_size, max_index_buffer_size) as u64;
    let mesh_staging_buffer = HalaBuffer::new(
      Rc::clone(&context.logical_device),
      mesh_staging_buffer_size,
      HalaBufferUsageFlags::TRANSFER_SRC,
      HalaMemoryLocation::CpuToGpu,
      "mesh_staging.buffer")?;
    for (mesh_index, mesh) in scene_in_cpu.meshes.iter().enumerate() {
      let mut primitives = Vec::with_capacity(mesh.primitives.len());
      for (prim_index, prim) in mesh.primitives.iter().enumerate() {
        let vertex_buffer_size = (prim.vertices.len() * std::mem::size_of::<HalaVertex>()) as u64;
        let vertex_buffer = HalaBuffer::new(
          Rc::clone(&context.logical_device),
          vertex_buffer_size,
          HalaBufferUsageFlags::VERTEX_BUFFER
            | HalaBufferUsageFlags::TRANSFER_DST
            | (if use_for_ray_tracing { HalaBufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY } else { HalaBufferUsageFlags::default() })
            | HalaBufferUsageFlags::SHADER_DEVICE_ADDRESS
            | HalaBufferUsageFlags::STORAGE_BUFFER,
          HalaMemoryLocation::GpuOnly,
          &format!("mesh_{}_prim_{}_vertex.buffer", mesh_index, prim_index))?;
        vertex_buffer.update_gpu_memory_with_buffer_raw(
          prim.vertices.as_ptr() as *const u8,
          vertex_buffer_size as usize,
          &mesh_staging_buffer,
          transfer_command_buffers)?;

        let index_buffer_size = (prim.indices.len() * std::mem::size_of::<u32>()) as u64;
        let index_buffer = HalaBuffer::new(
          Rc::clone(&context.logical_device),
          index_buffer_size,
          HalaBufferUsageFlags::INDEX_BUFFER
            | HalaBufferUsageFlags::TRANSFER_DST
            | (if use_for_ray_tracing { HalaBufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY } else { HalaBufferUsageFlags::default() })
            | HalaBufferUsageFlags::SHADER_DEVICE_ADDRESS
            | HalaBufferUsageFlags::STORAGE_BUFFER,
          HalaMemoryLocation::GpuOnly,
          &format!("mesh_{}_prim_{}_index.buffer", mesh_index, prim_index))?;
        index_buffer.update_gpu_memory_with_buffer_raw(
          prim.indices.as_ptr() as *const u8,
          index_buffer_size as usize,
          &mesh_staging_buffer,
          transfer_command_buffers)?;

        let material_index = prim.material_index;

        primitives.push(gpu::HalaPrimitive {
          vertex_buffer,
          index_buffer,
          vertex_count: prim.vertices.len() as u32,
          index_count: prim.indices.len() as u32,
          material_index,
          meshlet_count: 0u32,
          meshlet_buffer: None,
          meshlet_vertex_buffer: None,
          meshlet_primitive_buffer: None,
          btlas: None,
        });
      }
      meshes.push(gpu::HalaMesh {
        transform: glam::Mat4::IDENTITY,
        primitives,
      });
    }

    // Update the transform of the meshs.
    for node in scene_in_cpu.nodes.iter() {
      if node.mesh_index == u32::MAX {
        continue;
      }

      let mesh_index = node.mesh_index as usize;
      let mesh = &mut meshes[mesh_index];
      mesh.transform = node.world_transform;
    }

    let mut scene_in_gpu = gpu::HalaScene {
      camera_view_matrices,
      camera_proj_matrices,
      cameras: camera_buffer,
      lights: light_buffer,
      light_aabbs: light_aabb_buffer,
      materials: material_buffers,
      material_types,
      textures,
      samplers,
      images,
      meshes,
      instances: None,
      tplas: None,
      primitives: Vec::new(),
      light_btlas: None,
      light_data: lights,
    };

    if use_for_mesh_shader {
      Self::additively_upload_for_mesh_shader(
        context,
        graphics_command_buffers,
        transfer_command_buffers,
        scene_in_cpu,
        &mut scene_in_gpu,
      )?;
    }

    if use_for_ray_tracing {
      Self::additively_upload_for_ray_tracing(
        context,
        graphics_command_buffers,
        transfer_command_buffers,
        scene_in_cpu,
        &mut scene_in_gpu,
      )?;
    }

    Ok(scene_in_gpu)
  }

  /// Additively upload the scene to the GPU from the CPU for mesh shader.
  /// param context: The gfx context.
  /// param graphics_command_buffers: The graphics command buffers.
  /// param transfer_command_buffers: The transfer command buffers.
  /// param scene_in_cpu: The scene in the CPU.
  /// param scene_in_gpu: The scene in the GPU.
  /// return: The result.
  fn additively_upload_for_mesh_shader(
    context: &HalaContext,
    _graphics_command_buffers: &HalaCommandBufferSet,
    transfer_command_buffers: &HalaCommandBufferSet,
    scene_in_cpu: &mut cpu::HalaScene,
    scene_in_gpu: &mut gpu::HalaScene,
  ) -> Result<(), HalaRendererError> {
    let mut staging_buffer_size = 0u64;

    for mesh_in_cpu in scene_in_cpu.meshes.iter_mut() {
      for prim_in_cpu in mesh_in_cpu.primitives.iter_mut() {
        let vertex_data_adapter = unsafe {
          meshopt::VertexDataAdapter::new(
            std::slice::from_raw_parts(prim_in_cpu.vertices.as_ptr() as *const u8, prim_in_cpu.vertices.len() * std::mem::size_of::<HalaVertex>()),
            std::mem::size_of::<HalaVertex>(),
            0,
          ).map_err(|err| HalaRendererError::new("Failed to create vertex data adapter.", Some(Box::new(err))))?
        };
        let meshlets_in_cpu = meshopt::clusterize::build_meshlets(
          prim_in_cpu.indices.as_slice(),
          &vertex_data_adapter,
          64,
          124,
          0.5,
        );
        for (meshlet_index, meshlet_in_cpu) in meshlets_in_cpu.meshlets.iter().enumerate() {
          let wrapped_meshlet_in_cpu = meshlets_in_cpu.get(meshlet_index);
          let bounds = meshopt::clusterize::compute_meshlet_bounds(
            wrapped_meshlet_in_cpu,
            &vertex_data_adapter,
          );

          assert!(meshlet_in_cpu.triangle_offset % 4 == 0, "The triangle offset of the meshlet is not a multiple of 4.");
          assert!(wrapped_meshlet_in_cpu.triangles.len() % 3 == 0, "The triangle count of the meshlet is not a multiple of 3.");
          let meshlet = HalaMeshlet {
            center: bounds.center,
            radius: bounds.radius,
            cone_apex: bounds.cone_apex,
            cone_axis: bounds.cone_axis,
            cone_cutoff: bounds.cone_cutoff,
            offset_of_vertices: prim_in_cpu.meshlet_vertices.len() as u32,
            num_of_vertices: meshlet_in_cpu.vertex_count,
            offset_of_primitives: prim_in_cpu.meshlet_primitives.len() as u32,
            num_of_primitives: (wrapped_meshlet_in_cpu.triangles.len() / 3) as u32,
            padding: 0f32,
          };
          // log::info!("Meshlet: V[{}, {}], P[{}, {}]", meshlet.offset_of_vertices, meshlet.num_of_vertices, meshlet.offset_of_primitives, meshlet.num_of_primitives);

          prim_in_cpu.meshlets.push(meshlet);
          for i in wrapped_meshlet_in_cpu.vertices.iter() {
            prim_in_cpu.meshlet_vertices.push(*i);
          }
          for c in wrapped_meshlet_in_cpu.triangles.chunks(3) {
            prim_in_cpu.meshlet_primitives.push((c[0] as u32) | (c[1] as u32) << 8 | (c[2] as u32) << 16);
          }
        }

        let meshlet_buffer_size = (std::mem::size_of::<HalaMeshlet>() * prim_in_cpu.meshlets.len()) as u64;
        let meshlet_vertex_buffer_size = (std::mem::size_of::<u32>() * prim_in_cpu.meshlet_vertices.len()) as u64;
        let meshlet_primitive_buffer_size = (std::mem::size_of::<u32>() * prim_in_cpu.meshlet_primitives.len()) as u64;
        staging_buffer_size = std::cmp::max(
          staging_buffer_size,
          std::cmp::max(
            meshlet_buffer_size,
            std::cmp::max(
              meshlet_vertex_buffer_size,
              meshlet_primitive_buffer_size
            )
          )
        );
      }
    }

    // Create staging buffer.
    let staging_buffer = HalaBuffer::new(
      Rc::clone(&context.logical_device),
      staging_buffer_size,
      HalaBufferUsageFlags::TRANSFER_SRC,
      HalaMemoryLocation::CpuToGpu,
      "staging.buffer")?;

    // Create meshlet buffers.
    for (mesh_index, mesh) in scene_in_gpu.meshes.iter_mut().enumerate() {
      let mesh_in_cpu = &scene_in_cpu.meshes[mesh_index];
      for (prim_index, prim) in mesh.primitives.iter_mut().enumerate() {
        let prim_in_cpu = &mesh_in_cpu.primitives[prim_index];

        prim.meshlet_count = prim_in_cpu.meshlets.len() as u32;

        // Create meshlet informatin buffer.
        let meshlet_size = std::mem::size_of::<HalaMeshlet>();
        let meshlet_buffer_size = (meshlet_size * prim_in_cpu.meshlets.len()) as u64;
        let meshlet_buffer = HalaBuffer::new(
          Rc::clone(&context.logical_device),
          meshlet_buffer_size,
          HalaBufferUsageFlags::SHADER_DEVICE_ADDRESS
            | HalaBufferUsageFlags::STORAGE_BUFFER
            | HalaBufferUsageFlags::TRANSFER_DST,
          HalaMemoryLocation::GpuOnly,
          &format!("meshlet_info_{}_{}.buffer", mesh_index, prim_index)
        )?;
        meshlet_buffer.update_gpu_memory_with_buffer_raw(
          prim_in_cpu.meshlets.as_ptr() as *const u8,
          meshlet_buffer_size as usize,
          &staging_buffer,
          transfer_command_buffers)?;

        prim.meshlet_buffer = Some(meshlet_buffer);

        // Create meshlet vertex buffer.
        let meshlet_vertex_buffer_size = (std::mem::size_of::<u32>() * prim_in_cpu.meshlet_vertices.len()) as u64;
        let meshlet_vertex_buffer = HalaBuffer::new(
          Rc::clone(&context.logical_device),
          meshlet_vertex_buffer_size,
          HalaBufferUsageFlags::SHADER_DEVICE_ADDRESS
            | HalaBufferUsageFlags::STORAGE_BUFFER
            | HalaBufferUsageFlags::TRANSFER_DST,
          HalaMemoryLocation::GpuOnly,
          &format!("meshlet_vertex_{}_{}.buffer", mesh_index, prim_index)
        )?;
        meshlet_vertex_buffer.update_gpu_memory_with_buffer(
          prim_in_cpu.meshlet_vertices.as_slice(),
          &staging_buffer,
          transfer_command_buffers)?;

        prim.meshlet_vertex_buffer = Some(meshlet_vertex_buffer);

        // Create meshlet primitive buffer.
        let meshlet_primitive_buffer_size = (std::mem::size_of::<u32>() * prim_in_cpu.meshlet_primitives.len()) as u64;
        let meshlet_primitive_buffer = HalaBuffer::new(
          Rc::clone(&context.logical_device),
          meshlet_primitive_buffer_size,
          HalaBufferUsageFlags::SHADER_DEVICE_ADDRESS
            | HalaBufferUsageFlags::STORAGE_BUFFER
            | HalaBufferUsageFlags::TRANSFER_DST,
          HalaMemoryLocation::GpuOnly,
          &format!("meshlet_primitive_{}_{}.buffer", mesh_index, prim_index)
        )?;
        meshlet_primitive_buffer.update_gpu_memory_with_buffer(
          prim_in_cpu.meshlet_primitives.as_slice(),
          &staging_buffer,
          transfer_command_buffers)?;

        prim.meshlet_primitive_buffer = Some(meshlet_primitive_buffer);
      }
    }

    Ok(())
  }

  /// Additively upload the scene to the GPU from the CPU for ray tracing.
  /// param context: The gfx context.
  /// param graphics_command_buffers: The graphics command buffers.
  /// param transfer_command_buffers: The transfer command buffers.
  /// param scene_in_cpu: The scene in the CPU.
  /// param scene_in_gpu: The scene in the GPU.
  /// return: The result.
  fn additively_upload_for_ray_tracing(
    context: &HalaContext,
    graphics_command_buffers: &HalaCommandBufferSet,
    transfer_command_buffers: &HalaCommandBufferSet,
    scene_in_cpu: &cpu::HalaScene,
    scene_in_gpu: &mut gpu::HalaScene) -> Result<(), HalaRendererError>
  {
    // Build bottom level acceleration structure for each mesh.
    for (mesh_index, mesh) in scene_in_gpu.meshes.iter_mut().enumerate() {
      for (prim_index, prim) in mesh.primitives.iter_mut().enumerate() {
        let btlas = HalaAccelerationStructure::new(
          Rc::clone(&context.logical_device),
          graphics_command_buffers,
          HalaAccelerationStructureLevel::BOTTOM_LEVEL,
          &[HalaAccelerationStructureGeometry {
            ty: hala_gfx::HalaGeometryType::TRIANGLES,
            flags: hala_gfx::HalaGeometryFlags::OPAQUE,
            triangles_data: Some(HalaAccelerationStructureGeometryTrianglesData {
              vertex_format: hala_gfx::HalaFormat::R32G32B32_SFLOAT,
              vertex_data_address: prim.vertex_buffer.get_device_address(),
              vertex_stride: std::mem::size_of::<HalaVertex>() as u64,
              vertex_count: prim.vertex_count,
              index_type: hala_gfx::HalaIndexType::UINT32,
              index_data_address: prim.index_buffer.get_device_address(),
              transform_data_address: 0,
            }),
            aabbs_data: None,
            instances_data: None,
          }],
          &[&[HalaAccelerationStructureBuildRangeInfo {
            primitive_count: prim.index_count / 3,
            primitive_offset: 0,
            first_vertex: 0,
            transform_offset: 0,
          }]],
          &[prim.index_count / 3],
          &format!("mesh_{}_prim_{}.btlas", mesh_index, prim_index),
        )?;

        prim.btlas = Some(btlas);
      }
    }

    // Build bottom level acceleration structure for each light.
    let light_btlas = HalaAccelerationStructure::new(
      Rc::clone(&context.logical_device),
      graphics_command_buffers,
      HalaAccelerationStructureLevel::BOTTOM_LEVEL,
      &[HalaAccelerationStructureGeometry {
        ty: hala_gfx::HalaGeometryType::AABBS,
        flags: hala_gfx::HalaGeometryFlags::OPAQUE,
        triangles_data: None,
        aabbs_data: Some(HalaAccelerationStructureGeometryAabbsData {
          data_address: scene_in_gpu.light_aabbs.get_device_address(),
          stride: std::mem::size_of::<HalaAABB>() as u64,
        }),
        instances_data: None,
      }],
      &[&[HalaAccelerationStructureBuildRangeInfo {
        primitive_count: scene_in_gpu.light_data.len() as u32,
        primitive_offset: 0,
        first_vertex: 0,
        transform_offset: 0,
      }]],
      &[scene_in_gpu.light_data.len() as u32],
      "light.btlas",
    )?;

    // Build top level instance buffer.
    let mut primitives = Vec::new();
    let mut instances = Vec::with_capacity(scene_in_cpu.nodes.len());
    for node in scene_in_cpu.nodes.iter() {
      if node.mesh_index == u32::MAX {
        continue;
      }

      let mesh_index = node.mesh_index as usize;
      let mesh = &scene_in_gpu.meshes[mesh_index];
      for prim in mesh.primitives.iter() {
        let as_instance = HalaAccelerationStructureInstance {
          transform: [
            node.world_transform.x_axis.x, node.world_transform.y_axis.x, node.world_transform.z_axis.x, node.world_transform.w_axis.x,
            node.world_transform.x_axis.y, node.world_transform.y_axis.y, node.world_transform.z_axis.y, node.world_transform.w_axis.y,
            node.world_transform.x_axis.z, node.world_transform.y_axis.z, node.world_transform.z_axis.z, node.world_transform.w_axis.z,
          ],
          custom_index: primitives.len() as u32,
          mask: 0xff,
          shader_binding_table_record_offset: 0,
          shader_binding_table_flags: hala_gfx::HalaGeometryInstanceFlags::TRIANGLE_FACING_CULL_DISABLE,
          acceleration_structure_device_address: prim.btlas.as_ref().unwrap_or_else(|| panic!("mesh_{} do NOT has btlas.", mesh_index)).address
        };

        primitives.push(gpu::mesh::HalaMeshData {
          transform: node.world_transform,
          material_index: prim.material_index,
          vertices: prim.vertex_buffer.get_device_address(),
          indices: prim.index_buffer.get_device_address(),
        });

        instances.push(as_instance.as_data());
      }
    }

    let light_as_instance = HalaAccelerationStructureInstance {
      transform: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
      custom_index: 0u32,
      mask: 0xff,
      shader_binding_table_record_offset: 1,
      shader_binding_table_flags: hala_gfx::HalaGeometryInstanceFlags::TRIANGLE_FACING_CULL_DISABLE,
      acceleration_structure_device_address: light_btlas.address
    };
    instances.push(light_as_instance.as_data());

    // Perpare primitive buffers.
    let primitives_buffer_size = std::mem::size_of_val(&primitives[0]) as u64;
    let mut primitive_buffers = Vec::new();

    // Create instances buffer.
    let instances_buffer_size = (std::mem::size_of_val(&instances[0]) * instances.len()) as u64;
    let instances_buffer = HalaBuffer::new(
      Rc::clone(&context.logical_device),
      instances_buffer_size,
      HalaBufferUsageFlags::SHADER_DEVICE_ADDRESS
        | HalaBufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY
        | HalaBufferUsageFlags::TRANSFER_DST,
      HalaMemoryLocation::GpuOnly,
      "scene.instance_buffer",
    )?;

    // Create staging buffer.
    let staging_buffer = HalaBuffer::new(
      Rc::clone(&context.logical_device),
      std::cmp::max(primitives_buffer_size, instances_buffer_size),
      HalaBufferUsageFlags::TRANSFER_SRC,
      HalaMemoryLocation::CpuToGpu,
      "staging.buffer")?;

    // Upload the primitive buffers.
    for primitive in primitives.iter() {
      let primitives_buffer = HalaBuffer::new(
        Rc::clone(&context.logical_device),
        primitives_buffer_size,
        HalaBufferUsageFlags::UNIFORM_BUFFER | HalaBufferUsageFlags::TRANSFER_DST,
        HalaMemoryLocation::GpuOnly,
        "scene.primitives_buffer",
      )?;

      primitives_buffer.update_gpu_memory_with_buffer(
        std::slice::from_ref(primitive),
        &staging_buffer,
        transfer_command_buffers
      )?;

      primitive_buffers.push(primitives_buffer);
    }

    // Upload the instance buffer.
    instances_buffer.update_gpu_memory_with_buffer(
      instances.as_slice(),
      &staging_buffer,
      transfer_command_buffers)?;

    // Build top level acceleration structure.
    let tplas = HalaAccelerationStructure::new(
      Rc::clone(&context.logical_device),
      graphics_command_buffers,
      HalaAccelerationStructureLevel::TOP_LEVEL,
      &[HalaAccelerationStructureGeometry {
        ty: hala_gfx::HalaGeometryType::INSTANCES,
        flags: hala_gfx::HalaGeometryFlags::OPAQUE,
        triangles_data: None,
        aabbs_data: None,
        instances_data: Some(HalaAccelerationStructureGeometryInstancesData {
          array_of_pointers: false,
          data_address: instances_buffer.get_device_address(),
        }),
      }],
      &[&[HalaAccelerationStructureBuildRangeInfo {
        primitive_count: instances.len() as u32,
        primitive_offset: 0,
        first_vertex: 0,
        transform_offset: 0,
      }]],
      &[instances.len() as u32],
      "scene.tplas",
    )?;

    scene_in_gpu.instances = Some(instances_buffer);
    scene_in_gpu.tplas = Some(tplas);
    scene_in_gpu.primitives = primitive_buffers;
    scene_in_gpu.light_btlas = Some(light_btlas);

    Ok(())
  }

}