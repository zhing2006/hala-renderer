use std::path::Path;
use std::collections::{
  BTreeMap,
  VecDeque,
};

use glam::{
  Vec2,
  Vec3,
  Vec4,
  Vec4Swizzles,
};
use serde::{
  Deserialize, Serialize
};
use serde_json;

use hala_gfx::HalaFormat;
use crate::error::HalaRendererError;
use crate::scene::HalaVertex;
use super::super::{
  cpu::scene::HalaScene,
  cpu::node::HalaNode,
  cpu::material::{HalaMaterial, HalaMaterialType, HalaMedium, HalaMediumType},
  cpu::image_data::{HalaImageDataType, HalaImageData},
  cpu::mesh::{HalaPrimitive, HalaMesh},
  cpu::light::{HalaLightType, HalaLight},
  cpu::camera::{HalaCamera, HalaPerspectiveCamera, HalaOrthographicCamera},
};

/// The glTF loader.
pub struct HalaGltfLoader;

fn default_as_one() -> f32 {
  1.0
}

fn default_as_10() -> f32 {
  10.0
}

/// The glTF camera custom info.
#[derive(Serialize, Deserialize)]
struct _CameraCustomInfo {
  #[serde(default = "default_as_10")]
  pub focal_dist: f32,
  #[serde(default)]
  pub aperture: f32,
}

/// The glTF light custom info.
#[derive(Serialize, Deserialize)]
struct _LightCustomInfo {
  #[serde(rename = "type", default)]
  pub _type: u8,                // 1: Quad, 2: Sphere
  #[serde(default)]
  pub param0: f32,
  #[serde(default)]
  pub param1: f32,
}

/// The glTF material custom info.
#[derive(Serialize, Deserialize)]
struct _MaterialCustomInfo {
  #[serde(rename = "type")]
  pub _type: u8,                // 0: Diffuse, 1: Disney
  #[serde(default = "default_as_one")]
  pub opacity: f32,
  #[serde(default)]
  pub anisotropic: f32,
  #[serde(default)]
  pub subsurface: f32,
  #[serde(default)]
  pub specular_tint: f32,
  #[serde(default)]
  pub sheen: f32,
  #[serde(default)]
  pub sheen_tint: f32,
  #[serde(default)]
  pub clearcoat: f32,
  #[serde(default)]
  pub clearcoat_roughness: f32,
  #[serde(default)]
  pub clearcoat_tint: [f32; 3],
  #[serde(default)]
  pub medium_type: u8,
  #[serde(default)]
  pub medium_color: [f32; 3],
  #[serde(default)]
  pub medium_density: f32,
  #[serde(default)]
  pub medium_anisotropy: f32,
}

impl Default for _MaterialCustomInfo {
  fn default() -> Self {
    _MaterialCustomInfo {
      _type: 0,
      opacity: 1.0,
      anisotropic: 0.0,
      subsurface: 0.0,
      specular_tint: 0.0,
      sheen: 0.0,
      sheen_tint: 0.0,
      clearcoat: 0.0,
      clearcoat_roughness: 0.0,
      clearcoat_tint: [1.0, 1.0, 1.0],
      medium_type: 0,
      medium_color: [0.0, 0.0, 0.0],
      medium_density: 0.0,
      medium_anisotropy: 0.0,
    }
  }
}

/// The implementation of the glTF loader.
impl HalaGltfLoader {
  /// Load the glTF file from the given path.
  /// param path The path of the glTF file.
  /// return The loaded scene.
  pub fn load<P: AsRef<Path>>(path: P) -> Result<HalaScene, HalaRendererError> {
    let path = path.as_ref();
    let (gltf, mesh_data, image_data) = gltf::import(path)
      .map_err(|err| HalaRendererError::new(&format!("Load glTF file \"{:?}\" failed.", path), Some(Box::new(err))))?;

    // Load all nodes.
    let mut loaded_nodes = Vec::new();
    let scenes = gltf.scenes();
    if scenes.len() == 0 {
      return Err(HalaRendererError::new(&format!("No scene in glTF file \"{:?}\".", path), None));
    } else if scenes.len() > 1 {
      log::warn!("More than one scene in glTF file \"{:?}\". Only the first scene will be loaded.", path);
    }
    for scene in scenes {
      log::debug!("Loading scene \"{}\".", scene.name().unwrap_or("<Unnamed>"));

      let mut node_queue = VecDeque::new();
      node_queue.extend(scene.nodes().map(|node| (u32::MAX, node)));

      while !node_queue.is_empty() {
        let node_pair = node_queue.pop_front();
        if let Some((parent_idx, node)) = node_pair {
          let local_mtx = node.transform().matrix();
          let current_index = loaded_nodes.len() as u32;
          let mut loaded_node = HalaNode {
            name: node.name().unwrap_or("<Unnamed>").to_owned(),
            parent: if parent_idx == u32::MAX { None } else { Some(parent_idx) },
            local_transform: glam::Mat4::from_cols_array_2d(&local_mtx),
            ..Default::default()
          };

          // If the node has a mesh, set the mesh index.
          if let Some(mesh) = node.mesh() {
            loaded_node.mesh_index = mesh.index() as u32;
          }

          // If the node has a camera, set the camera index.
          if let Some(camera) = node.camera() {
            loaded_node.camera_index = camera.index() as u32;
          }

          // If the node has a light, set the light index.
          if let Some(light) = node.light() {
            loaded_node.light_index = light.index() as u32;
          }

          loaded_nodes.push(loaded_node);
          node_queue.extend(node.children().map(|child| (current_index, child)));
        } else {
          return Err(HalaRendererError::new(&format!("Node queue is empty while loading scene \"{}\".", scene.name().unwrap_or("<Unnamed>")), None));
        }
      }
    }

    // Load all meshes.
    let mut loaded_meshes = Vec::new();
    for mesh in gltf.meshes() {
      loaded_meshes.push(Self::load_mesh(&mesh, &mesh_data)?);
    }

    // Load all materials.
    let mut loaded_materials = Vec::new();
    for material in gltf.materials() {
      loaded_materials.push(Self::load_material(&material)?);
    }

    // Load all images and textures.
    let mut loaded_texture2image_mapping = BTreeMap::new();
    for (index, texture) in gltf.textures().enumerate() {
      let image = texture.source();
      loaded_texture2image_mapping.insert(index as u32, image.index() as u32);
    }
    let mut loaded_image2data_mapping = BTreeMap::new();
    for (index, image) in gltf.images().enumerate() {
      log::debug!("Loading image \"{}\".", image.name().unwrap_or("<Unnamed>"));
      loaded_image2data_mapping.insert(index as u32, image.index() as u32);
    }
    let mut loaded_textures = Vec::new();
    for data in image_data {
      loaded_textures.push(Self::load_image_data(&data)?);
    }

    // Load all lights.
    let mut loaded_lights = Vec::new();
    if let Some(lights) = gltf.lights() {
      for light in lights {
        loaded_lights.push(Self::load_light(&light)?);
      }
    }

    // Load all cameras.
    let mut loaded_cameras = Vec::new();
    for camera in gltf.cameras() {
      loaded_cameras.push(Self::load_camera(&camera)?);
    }

    Ok(HalaScene {
      nodes: loaded_nodes,
      meshes: loaded_meshes,
      materials: loaded_materials,
      texture2image_mapping: loaded_texture2image_mapping,
      image2data_mapping: loaded_image2data_mapping,
      image_data: loaded_textures,
      lights: loaded_lights,
      cameras: loaded_cameras,
    })
  }

  /// Load the mesh.
  /// param mesh The gltf mesh.
  /// param buffers The gltf buffers.
  fn load_mesh(mesh: &gltf::Mesh, buffers: &[gltf::buffer::Data]) -> Result<HalaMesh, HalaRendererError> {
    let mesh_name = mesh.name().unwrap_or("<Unnamed>");
    log::debug!("Loading mesh \"{}\".", mesh_name);
    let primitives = mesh.primitives();

    let mut loaded_primitives = Vec::new();
    for primitive in primitives {
      log::debug!("Loading primitive {} from mesh \"{}\".", primitive.index(), mesh_name);
      let reader = primitive.reader(|i| Some(&buffers[i.index()]));

      let indices = reader.read_indices()
        .ok_or(HalaRendererError::new(&format!("Read indices from mesh \"{}\" failed.", mesh_name), None))?
        .into_u32().collect::<Vec<_>>();
      let positions = reader.read_positions()
        .ok_or(HalaRendererError::new(&format!("Read positions from mesh \"{}\" failed.", mesh_name), None))?
        .map(Vec3::from).collect::<Vec<_>>();
      let normals = reader.read_normals()
        .ok_or(HalaRendererError::new(&format!("Read normals from mesh \"{}\" failed.", mesh_name), None))?
        .map(Vec3::from).collect::<Vec<_>>();
      let tex_coords = reader.read_tex_coords(0)
        .ok_or(HalaRendererError::new(&format!("Read tex_coords from mesh \"{}\" failed.", mesh_name), None))?
        .into_f32().map(Vec2::from).collect::<Vec<_>>();

      let tangents = if let Some(tangents) = reader.read_tangents() {
        tangents.map(|tangent| {
          let t: [f32; 3] = [tangent[0] / tangent[3], tangent[1] / tangent[3], tangent[2] / tangent[3]];
          Vec3::from(t)
        }).collect::<Vec<_>>()
      } else {
        // Fill the tangents with zero.
        let mut tangents = vec![Vec3::ZERO; positions.len()];
        // Calculate tangent.
        for tri_indices in indices.chunks(3) {
          let v0 = positions[tri_indices[0] as usize];
          let v1 = positions[tri_indices[1] as usize];
          let v2 = positions[tri_indices[2] as usize];
          let uv0 = tex_coords[tri_indices[0] as usize];
          let uv1 = tex_coords[tri_indices[1] as usize];
          let uv2 = tex_coords[tri_indices[2] as usize];

          let delta_pos1 = v1 - v0;
          let delta_pos2 = v2 - v0;

          let delta_uv1 = uv1 - uv0;
          let delta_uv2 = uv2 - uv0;

          let invdet = 1.0 / (delta_uv1.x * delta_uv2.y - delta_uv1.y * delta_uv2.x);

          let tangent = ((delta_pos1 * delta_uv2.y - delta_pos2 * delta_uv1.y) * invdet).normalize();
          tangents[tri_indices[0] as usize] = tangent;
          tangents[tri_indices[1] as usize] = tangent;
          tangents[tri_indices[2] as usize] = tangent;
        }
        tangents
      };

      let mut vertices = Vec::new();
      for i in 0..positions.len() {
        vertices.push(HalaVertex {
          position: positions[i].into(),
          normal: normals[i].into(),
          tangent: tangents[i].into(),
          tex_coord: tex_coords[i].into(),
        });
      }

      let material_index = primitive.material().index().map_or(u32::MAX, |idx| idx as u32);

      loaded_primitives.push(HalaPrimitive {
        indices,
        vertices,
        material_index,
        meshlets: Vec::new(),
        meshlet_vertices: Vec::new(),
        meshlet_primitives: Vec::new(),
      });
    }

    Ok(HalaMesh{
      primitives: loaded_primitives,
    })
  }

  /// Load the material.
  /// param material The gltf material.
  /// return The loaded material.
  fn load_material(material: &gltf::Material) -> Result<HalaMaterial, HalaRendererError> {
    log::debug!("Loading material \"{}\".", material.name().unwrap_or("<Unnamed>"));
    let pbr = material.pbr_metallic_roughness();

    let custom_info = match material.extras() {
      Some(extras) => {
        serde_json::from_str::<_MaterialCustomInfo>(extras.get())
          .map_err(|err| HalaRendererError::new("Parse material extras failed.", Some(Box::new(err))))?
      },
      None => {
        _MaterialCustomInfo::default()
      },
    };

    let base_color: Vec4 = pbr.base_color_factor().into();
    let metallic = pbr.metallic_factor();
    let roughness = pbr.roughness_factor();
    let mut emission: Vec3 = material.emissive_factor().into();
    if let Some(emissive_strength) = material.emissive_strength() {
      emission *= emissive_strength;
    }

    let specular_transmission = match material.transmission() {
      Some(transmission) => transmission.transmission_factor(),
      None => 0.0,
    };
    let ior = material.ior().unwrap_or(1.5);

    let base_color_map_index = pbr.base_color_texture()
    .map_or(u32::MAX, |texture| texture.texture().index() as u32);
    let normal_map_index = material.normal_texture()
      .map_or(u32::MAX, |texture| texture.texture().index() as u32);
    let metallic_roughness_map_index = pbr.metallic_roughness_texture()
      .map_or(u32::MAX, |texture| texture.texture().index() as u32);
    let emission_map_index = material.emissive_texture()
      .map_or(u32::MAX, |texture| texture.texture().index() as u32);

    Ok(HalaMaterial {
      _type: HalaMaterialType::from_u8(custom_info._type),
      base_color: base_color.xyz(),
      opacity: custom_info.opacity,
      emission,
      anisotropic: custom_info.anisotropic,
      metallic,
      roughness,
      subsurface: custom_info.subsurface,
      specular_tint: custom_info.specular_tint,
      sheen: custom_info.sheen,
      sheen_tint: custom_info.sheen_tint,
      clearcoat: custom_info.clearcoat,
      clearcoat_roughness: custom_info.clearcoat_roughness,
      clearcoat_tint: Vec3::from(custom_info.clearcoat_tint),
      specular_transmission,
      ior,

      medium: HalaMedium {
        _type: HalaMediumType::from_u8(custom_info.medium_type),
        color: Vec3::from(custom_info.medium_color),
        density: custom_info.medium_density,
        anisotropy: custom_info.medium_anisotropy,
      },

      base_color_map_index,
      emission_map_index,
      normal_map_index,
      metallic_roughness_map_index,
    })
  }

  /// Load the image data.
  /// param image The gltf image.
  /// param image_data The gltf image data.
  /// return The loaded texture.
  fn load_image_data(image_data: &gltf::image::Data) -> Result<HalaImageData, HalaRendererError> {
    let format = match image_data.format {
      gltf::image::Format::R8 => HalaFormat::R8_UNORM,
      gltf::image::Format::R8G8 => HalaFormat::R8G8_UNORM,
      gltf::image::Format::R8G8B8 => HalaFormat::R8G8B8A8_SRGB, // Do NOT support R8G8B8 format. See below.
      gltf::image::Format::R8G8B8A8 => HalaFormat::R8G8B8A8_SRGB,
      gltf::image::Format::R16 => HalaFormat::R16_UNORM,
      gltf::image::Format::R16G16 => HalaFormat::R16G16_UNORM,
      gltf::image::Format::R16G16B16 => HalaFormat::R16G16B16_UNORM,
      gltf::image::Format::R16G16B16A16 => HalaFormat::R16G16B16A16_UNORM,
      gltf::image::Format::R32G32B32FLOAT => HalaFormat::R32G32B32_SFLOAT,
      gltf::image::Format::R32G32B32A32FLOAT => HalaFormat::R32G32B32A32_SFLOAT,
    };
    let width = image_data.width;
    let height = image_data.height;

    // Our GPU do NOT support R8G8B8 format, so we need to convert it to R8G8B8A8 format.
    let pixels = if image_data.format == gltf::image::Format::R8G8B8 {
      let mut pixels = Vec::with_capacity(image_data.pixels.len() / 3 * 4);
      for i in 0..image_data.pixels.len() / 3 {
        pixels.push(image_data.pixels[i * 3]);
        pixels.push(image_data.pixels[i * 3 + 1]);
        pixels.push(image_data.pixels[i * 3 + 2]);
        pixels.push(255);
      }
      pixels
    } else {
      image_data.pixels.clone()
    };

    let num_of_bytes = pixels.len();
    Ok(HalaImageData {
      format,
      width,
      height,
      data_type: HalaImageDataType::ByteData(pixels),
      num_of_bytes,
    })
  }

  /// Load the light.
  /// param light The gltf light.
  /// return The loaded light.
  fn load_light(light: &gltf::khr_lights_punctual::Light) -> Result<HalaLight, HalaRendererError> {
    log::debug!("Loading light \"{}\".", light.name().unwrap_or("<Unnamed>"));

    let color: Vec3 = light.color().into();
    let mut intensity = light.intensity();
    let (mut light_type, mut param0, mut param1) = match light.kind() {
      gltf::khr_lights_punctual::Kind::Directional => (HalaLightType::DIRECTIONAL, 0.0, 0.0),
      gltf::khr_lights_punctual::Kind::Point => (HalaLightType::POINT, 0.0, 0.0),
      gltf::khr_lights_punctual::Kind::Spot{
        inner_cone_angle,
        outer_cone_angle,
      } => {
        (HalaLightType::SPOT, inner_cone_angle, outer_cone_angle)
      },
    };
    if let Some(extras) = light.extras() {
      let custom_info: _LightCustomInfo = serde_json::from_str(extras.get())
        .map_err(|err| HalaRendererError::new("Parse light extras failed.", Some(Box::new(err))))?;
      if custom_info._type == 1 {
        light_type = HalaLightType::QUAD;
      } else if custom_info._type == 2 {
        light_type = HalaLightType::SPHERE;
      }
      param0 = custom_info.param0;
      param1 = custom_info.param1;
    }
    match light_type {
      HalaLightType::DIRECTIONAL => {
        param0 = param0.clamp(0.0, 90.0);
        param0 = param0.to_radians();
      },
      HalaLightType::SPOT => {
        param0 = param0.clamp(0.0, 90.0);
        param1 = param1.clamp(0.0, 90.0);
        if param0 > param1 {
          std::mem::swap(&mut param0, &mut param1);
        };
      },
      HalaLightType::QUAD => {
        // Quad light is exported as point light in Blender, So we need to recalculate the intensity.
        // Quad light is single side, so the total area is 0.5 * param0 * param1.
        intensity /= 0.5 * param0 * param1;
      },
      _ => {},
    }
    let params = (param0, param1);

    Ok(HalaLight {
      color,
      intensity,
      light_type,
      params,
    })
  }

  /// Load the camera.
  /// param camera The gltf camera.
  /// return The loaded camera.
  fn load_camera(camera: &gltf::Camera) -> Result<HalaCamera, HalaRendererError> {
    log::debug!("Loading camera \"{}\".", camera.name().unwrap_or("<Unnamed>"));

    match camera.projection() {
      gltf::camera::Projection::Orthographic(orthographic) => {
        let xmag = orthographic.xmag();
        let ymag = orthographic.ymag();
        let znear = orthographic.znear();
        let zfar = orthographic.zfar();

        let orthography = glam::Mat4::orthographic_rh(-xmag, xmag, -ymag, ymag, znear, zfar);

        Ok(HalaCamera::Orthographic(HalaOrthographicCamera {
          xmag,
          ymag,
          orthography,
        }))
      },
      gltf::camera::Projection::Perspective(perspective) => {
        let aspect = perspective.aspect_ratio().unwrap_or(1.0);
        let yfov = perspective.yfov();
        let znear = perspective.znear();
        let zfar = perspective.zfar().unwrap_or(1000.0);

        // Use infinite reverse perspective projection(depth range: 1 to 0).
        let projection = glam::Mat4::perspective_infinite_reverse_rh(yfov, aspect, znear);

        let (focal_distance, aperture) =if let Some(extras) = camera.extras() {
          let custom_info: _CameraCustomInfo = serde_json::from_str(extras.get())
            .map_err(|err| HalaRendererError::new("Parse camera extras failed.", Some(Box::new(err))))?;
          (custom_info.focal_dist, custom_info.aperture)
        } else {
          (10.0, 0.0)
        };

        Ok(HalaCamera::Perspective(HalaPerspectiveCamera {
          aspect,
          yfov,
          znear,
          zfar,
          focal_distance,
          aperture,
          projection,
        }))
      },
    }
  }
}
