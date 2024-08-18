/// The meshlet.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
pub struct HalaMeshlet {
  pub center: [f32; 3],
  pub radius: f32,
  pub cone_apex: [f32; 3],
  pub cone_cutoff: f32,
  pub cone_axis: [f32; 3],
  pub num_of_vertices: u32,
  pub num_of_primitives: u32,
  pub offset_of_vertices: u32,
  pub offset_of_primitives: u32,
  pub draw_index: u32,
}