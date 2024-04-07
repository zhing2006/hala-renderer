use glam::{
  Vec3,
  Vec3A,
};

/// The light information in the GPU.
#[repr(C, align(16))]
pub struct HalaLight {
  pub intensity: Vec3A,
  // For point light, position is the position.
  // For directional light, position is unused.
  // For spot light, quad light and sphere light, position is the position.
  pub position: Vec3A,
  // For point light, u is unused.
  // For directional light and spot light, u is the direction.
  // For quad light, u is the right direction and length.
  // For sphere light, u is unused.
  pub u: Vec3A,
  // For point light v is unused.
  // For directional light, v.x is the cosine of the cone angle.
  // For spot light, v.x is the cosine of the inner cone angle, v.y is the cosine of the outer cone angle.
  // For quad light, v is the up direction and length.
  // For sphere light, v is unused.
  pub v: Vec3,
  // For point light, directional light, spot light and quad light, radius is unused.
  // For sphere light, radius is the radius.
  pub radius: f32,
  // For point light, directional light and spot light, area is unused.
  // For quad light and sphere light, area is the area.
  pub area: f32,
  pub _type: u32,
}
