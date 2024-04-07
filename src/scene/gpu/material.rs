use glam::Vec3;

use crate::scene::cpu::material::{HalaMaterial as HalaMaterialInCPU, HalaMaterialType};

/// The medium information in the GPU.
#[repr(C, align(16))]
pub struct HalaMedium {
  pub color: Vec3,
  pub density: f32,
  pub anisotropy: f32,
  pub _type: u32,
  pub _padding: [f32; 2],
}

/// The material information in the GPU.
#[repr(C, align(16))]
pub struct HalaMaterial {
  pub medium: HalaMedium,

  pub base_color: Vec3,
  pub opacity: f32,
  pub emission: Vec3,
  pub anisotropic: f32,
  pub metallic: f32,
  pub roughness: f32,
  pub subsurface: f32,
  pub specular_tint: f32,
  pub sheen: f32,
  pub sheen_tint: f32,
  pub clearcoat: f32,
  pub clearcoat_roughness: f32,
  pub specular_transmission: f32,
  pub ior: f32,

  pub ax: f32,
  pub ay: f32,

  pub base_color_map_index: u32,
  pub normal_map_index: u32,
  pub metallic_roughness_map_index: u32,
  pub emission_map_index: u32,

  pub _type: u32,
}

/// The From implementation of the material.
impl std::convert::From<&HalaMaterialInCPU> for HalaMaterial {
  fn from(material: &HalaMaterialInCPU) -> Self {
    let (roughness, ax, ay) = if material._type == HalaMaterialType::DIFFUSE {
      let sigma = material.roughness * 0.5 * std::f32::consts::FRAC_PI_2;
      let sigma2 = sigma * sigma;
      (
        material.roughness,
        1.0 - (sigma2 / (2.0 * (sigma2 + 0.33))),
        0.45 * sigma2 / (sigma2 + 0.09),
      )
    } else {
      let roughness = material.roughness * material.roughness;
      let aspect = f32::sqrt(1.0 - f32::clamp(material.anisotropic, 0.0, 1.0) * 0.9);
      (
        roughness,
        f32::max(0.001, roughness / aspect),
        f32::max(0.001, roughness * aspect),
      )
    };

    Self {
      base_color: material.base_color,
      opacity: material.opacity,
      emission: material.emission,
      anisotropic: material.anisotropic,
      metallic: material.metallic,
      roughness,
      subsurface: material.subsurface,
      specular_tint: material.specular_tint,
      sheen: material.sheen,
      sheen_tint: material.sheen_tint,
      clearcoat: material.clearcoat,
      clearcoat_roughness: material.clearcoat_roughness,
      specular_transmission: material.specular_transmission,
      ior: material.ior,

      ax,
      ay,

      medium: HalaMedium {
        color: material.medium.color,
        density: material.medium.density,
        anisotropy: material.medium.anisotropy,
        _type: material.medium._type.to_u8() as u32,
        _padding: [0.0; 2],
      },

      base_color_map_index: material.base_color_map_index,
      normal_map_index: material.normal_map_index,
      metallic_roughness_map_index: material.metallic_roughness_map_index,
      emission_map_index: material.emission_map_index,
      _type: material._type.to_u8() as u32,
    }
  }
}