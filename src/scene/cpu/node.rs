/// A node is a collection of transforms and child indices.
pub struct HalaNode {
  pub name: String,
  pub parent: Option<u32>,
  pub children: Vec<u32>,
  pub local_transform: glam::Mat4,
  pub world_transform: glam::Mat4,

  pub mesh_index: u32,
  pub camera_index: u32,
  pub light_index: u32,
}

/// The default implementation of the node.
impl Default for HalaNode {
  fn default() -> Self {
    Self {
      name: String::new(),
      parent: None,
      children: Vec::new(),
      local_transform: glam::Mat4::IDENTITY,
      world_transform: glam::Mat4::IDENTITY,
      mesh_index: u32::MAX,
      camera_index: u32::MAX,
      light_index: u32::MAX,
    }
  }
}