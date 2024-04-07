use glam::Vec3;

/// The type of the light.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HalaLightType(u8);
impl HalaLightType {
  pub const POINT: Self = Self(0);
  pub const DIRECTIONAL: Self = Self(1);
  pub const SPOT: Self = Self(2);
  pub const QUAD: Self = Self(3);
  pub const SPHERE: Self = Self(4);

  pub fn from_u8(value: u8) -> Self {
    match value {
      0 => Self::POINT,
      1 => Self::DIRECTIONAL,
      2 => Self::SPOT,
      3 => Self::QUAD,
      4 => Self::SPHERE,
      _ => panic!("Invalid light type."),
    }
  }

  pub fn to_u8(&self) -> u8 {
    self.0
  }
}

/// A light source in the scene.
pub struct HalaLight {
  pub color: Vec3,
  pub intensity: f32,
  pub light_type: HalaLightType,
  /// For directional light, param0 is the angle of the soft shadow edge.
  /// For spot light, param0 is the cosine of the inner cone angle, param1 is the cosine of the outer cone angle.
  /// For quad light, param0 is the width, param1 is the height.
  /// For sphere light, param0 is the radius, param1 is unused.
  pub params: (f32, f32),
}