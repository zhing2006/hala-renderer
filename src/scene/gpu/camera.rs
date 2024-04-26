use glam::{
  Vec3A,
  Vec3,
};

use crate::scene::cpu::node::HalaNode as HalaNodeInCPU;
use crate::scene::cpu::camera::HalaCamera as HalaCameraInCPU;

/// The camera information in the GPU.
#[repr(C, align(16))]
pub struct HalaCamera {
  pub position: Vec3A,
  pub right: Vec3A,
  pub up: Vec3A,
  pub forward: Vec3,
  pub yfov: f32,
  pub focal_distance_or_xmag: f32,
  pub aperture_or_ymag: f32,
  pub _type: u32,
}

/// The implementation of the camera in the GPU.
impl HalaCamera {

  /// Create a new camera in the GPU.
  /// param node_in_cpu: The camera node in the CPU.
  /// param camera_in_cpu: The camera in the CPU.
  pub fn new(node_in_cpu: &HalaNodeInCPU, camera_in_cpu: &HalaCameraInCPU) -> Self {
    let position = node_in_cpu.world_transform.w_axis.truncate();
    let right = node_in_cpu.world_transform.x_axis.truncate();
    let up = node_in_cpu.world_transform.y_axis.truncate();
    let forward = -node_in_cpu.world_transform.z_axis.truncate();

    match camera_in_cpu {
      HalaCameraInCPU::Perspective(camera) => {
        Self {
          position: position.into(),
          right: right.into(),
          up: up.into(),
          forward,
          yfov: camera.yfov,
          focal_distance_or_xmag: camera.focal_distance,
          aperture_or_ymag: camera.aperture,
          _type: 0,
        }
      },
      HalaCameraInCPU::Orthographic(camera) => {
        Self {
          position: position.into(),
          right: right.into(),
          up: up.into(),
          forward,
          yfov: 0.0,
          focal_distance_or_xmag: camera.xmag,
          aperture_or_ymag: camera.ymag,
          _type: 1,
        }
      },
    }

  }

}