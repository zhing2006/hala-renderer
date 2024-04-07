use glam::Mat4;

/// A camera in the scene.
pub struct HalaCamera {
  pub aspect: f32,
  pub yfov: f32,
  pub znear: f32,
  pub zfar: f32,
  pub focal_distance: f32,
  pub aperture: f32,

  pub projection: Mat4,
}