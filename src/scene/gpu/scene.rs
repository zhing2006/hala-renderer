use hala_gfx::{
  HalaBuffer,
  HalaSampler,
  HalaImage,
  HalaAccelerationStructure,
};

use crate::scene::gpu::HalaMesh;

/// The scene in the GPU.
pub struct HalaScene {
  pub camera_view_matrices: Vec<glam::Mat4>,
  pub camera_proj_matrices: Vec<glam::Mat4>,

  pub cameras: HalaBuffer,
  pub lights: HalaBuffer,
  pub light_aabbs: HalaBuffer,
  pub materials: Vec<HalaBuffer>,
  pub textures: Vec<u32>, // indices to the images.
  pub samplers: Vec<HalaSampler>,
  pub images: Vec<HalaImage>,
  pub meshes: Vec<HalaMesh>,

  pub instances: Option<HalaBuffer>,
  pub tplas: Option<HalaAccelerationStructure>,
  pub primitives: Vec<HalaBuffer>,
  pub light_btlas: Option<HalaAccelerationStructure>,

  pub light_data: Vec<crate::scene::gpu::HalaLight>,
}