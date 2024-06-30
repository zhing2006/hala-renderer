/// The vertex.
#[repr(C, align(4))]
#[derive(Debug, Copy, Clone)]
pub struct HalaVertex {
  pub position: [f32; 3],
  pub normal: [f32; 3],
  pub tangent: [f32; 3],
  pub tex_coord: [f32; 2],
}