use glam::{
  Vec2,
  Vec3A,
};

/// The vertex.
#[repr(C, align(16))]
pub struct HalaVertex {
  pub position: Vec3A,
  pub normal: Vec3A,
  pub tangent: Vec3A,
  pub tex_coord: Vec2,
}