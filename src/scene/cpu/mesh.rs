use crate::scene::{
  HalaVertex,
  HalaMeshlet,
};

pub struct HalaPrimitive {
  pub indices: Vec<u32>,
  pub vertices: Vec<HalaVertex>,
  pub material_index: u32,
  pub meshlets: Vec<HalaMeshlet>,
  pub meshlet_vertices: Vec<u32>,
  pub meshlet_primitives: Vec<u8>,
}

/// A mesh is a collection of vertices and indices that define a 3D object.
pub struct HalaMesh {
  pub primitives: Vec<HalaPrimitive>,
}