use glam::Vec3;

/// The type of the material.
#[derive(PartialEq, Eq)]
pub struct HalaMaterialType(u8);
impl HalaMaterialType {
  pub const DIFFUSE: Self = Self(0);
  pub const DISNEY: Self = Self(1);

  pub fn from_u8(value: u8) -> Self {
    match value {
      0 => Self::DIFFUSE,
      1 => Self::DISNEY,
      _ => panic!("Invalid material type."),
    }
  }

  pub fn to_u8(&self) -> u8 {
    self.0
  }
}

/// A material for objects.
pub struct HalaMaterial {
  pub _type: HalaMaterialType,
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

  pub medium: HalaMedium,

  pub base_color_map_index: u32,
  pub emission_map_index: u32,
  pub normal_map_index: u32,
  pub metallic_roughness_map_index: u32,
}

/// The type of medium.
#[derive(PartialEq, Eq)]
pub struct HalaMediumType(u8);
impl HalaMediumType {
  pub const NONE: Self = Self(0);
  pub const ABSORB: Self = Self(1);
  pub const SCATTER: Self = Self(2);
  pub const EMISSIVE: Self = Self(3);

  pub fn from_u8(value: u8) -> Self {
    match value {
      0 => Self::NONE,
      1 => Self::ABSORB,
      2 => Self::SCATTER,
      3 => Self::EMISSIVE,
      _ => panic!("Invalid medium type."),
    }
  }

  pub fn to_u8(&self) -> u8 {
    self.0
  }
}

/// A medium for objects.
pub struct HalaMedium {
  pub _type: HalaMediumType,
  pub color: Vec3,
  pub density: f32,
  pub anisotropy: f32,
}