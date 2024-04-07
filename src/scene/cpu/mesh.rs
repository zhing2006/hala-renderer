use crate::scene::HalaVertex;

pub struct HalaPrimitive {
  pub indices: Vec<u32>,
  pub vertices: Vec<HalaVertex>,
  pub material_index: u32,
}


/// A mesh is a collection of vertices and indices that define a 3D object.
pub struct HalaMesh {
  pub primitives: Vec<HalaPrimitive>,
}