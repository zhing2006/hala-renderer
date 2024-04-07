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
  pub focal_distance: f32,
  pub aperture: f32,
  _padding: f32,
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

    Self {
      position: position.into(),
      right: right.into(),
      up: up.into(),
      forward,
      yfov: camera_in_cpu.yfov,
      focal_distance: camera_in_cpu.focal_distance,
      aperture: camera_in_cpu.aperture,
      _padding: 0.0,
    }
  }

}