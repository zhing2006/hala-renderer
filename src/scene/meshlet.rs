/// The meshlet.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct HalaMeshlet {
  pub center: [f32; 3],
  pub radius: f32,
  pub cone_apex: [f32; 3],
  pub num_of_vertices: u32,
  pub cone_axis: [f32; 3],
  pub num_of_triangles: u32,
}