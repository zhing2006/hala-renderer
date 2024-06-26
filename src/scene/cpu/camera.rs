use glam::Mat4;

/// A perspective camera in the scene.
pub struct HalaPerspectiveCamera {
  pub aspect: f32,
  pub yfov: f32,
  pub znear: f32,
  pub zfar: f32,
  pub focal_distance: f32,
  pub aperture: f32,

  pub projection: Mat4,
}

/// A orthographic camera in the scene.
pub struct HalaOrthographicCamera {
  pub xmag: f32,
  pub ymag: f32,

  pub orthography: Mat4,
}

/// A camera in the scene.
pub enum HalaCamera {
  Perspective(HalaPerspectiveCamera),
  Orthographic(HalaOrthographicCamera),
}

impl HalaCamera {

  pub fn get_proj_matrix(&self) -> Mat4 {
    match self {
      HalaCamera::Perspective(camera) => camera.projection,
      HalaCamera::Orthographic(camera) => camera.orthography,
    }
  }

}