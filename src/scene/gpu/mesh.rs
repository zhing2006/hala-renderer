use hala_gfx::{
  HalaBuffer,
  HalaAccelerationStructure,
};

/// The primitive in the GPU.
pub struct HalaPrimitive {
  pub vertex_buffer: HalaBuffer,
  pub index_buffer: HalaBuffer,
  pub vertex_count: u32,
  pub index_count: u32,
  pub material_index: u32,

  pub meshlet_count: u32,
  pub meshlet_buffer: Option<HalaBuffer>,
  pub meshlet_vertex_buffer: Option<HalaBuffer>,
  pub meshlet_primitive_buffer: Option<HalaBuffer>,

  pub btlas: Option<HalaAccelerationStructure>,
}

/// The mesh in the GPU.
pub struct HalaMesh {
  pub transform: glam::Mat4,
  pub primitives: Vec<HalaPrimitive>,
}

/// The mesh data in the GPU buffer
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct HalaMeshData {
  pub transform: glam::Mat4,
  pub material_index: u32,
  pub vertices: u64,
  pub indices: u64,
}