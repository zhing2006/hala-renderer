use std::path::Path;
use std::collections::BTreeMap;

use crate::error::HalaRendererError;
use super::node::HalaNode;
use super::mesh::HalaMesh;
use super::material::{
  HalaMaterial,
  HalaMediumType
};
use super::image_data::HalaImageData;
use super::light::HalaLight;
use super::camera::HalaCamera;
use super::super::loader::HalaGltfLoader;

/// A scene is a collection of objects and lights.
pub struct HalaScene {
  pub nodes: Vec<HalaNode>,
  pub meshes: Vec<HalaMesh>,
  pub materials: Vec<HalaMaterial>,
  pub texture2image_mapping: BTreeMap<u32, u32>,
  pub image2data_mapping: BTreeMap<u32, u32>,
  pub image_data: Vec<HalaImageData>,
  pub lights: Vec<HalaLight>,
  pub cameras: Vec<HalaCamera>,
}

/// The Drop implementation of the scene.
impl Drop for HalaScene {
  fn drop(&mut self) {
    log::debug!("A HalaScene dropped.");
  }
}

/// The implementation of the scene.
impl HalaScene {
  /// Create a new scene from glTF file.
  /// param path: The path to the glTF file.
  /// return: The scene.
  pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, HalaRendererError> {
    // Check the file extension.
    let path = path.as_ref();
    let extension = path.extension()
      .ok_or(HalaRendererError::new(&format!("Get file \"{:?}\" extension failed.", path), None))?;
    let mut scene = match extension.to_str() {
      // glTF file.
      Some("gltf") => HalaGltfLoader::load(path),
      // Unsupported file.
      _ => Err(HalaRendererError::new(&format!("Unsupported file \"{:?}\".", path), None)),
    }?;
    scene.update_node_hierarchies();

    log::debug!("A HalaScene created.");
    Ok(scene)
  }

  /// Check if the scene has light.
  /// return: True if the scene has light, false otherwise.
  pub fn has_light(&self) -> bool {
    !self.lights.is_empty()
  }

  /// Check if the scene has camera.
  /// return: True if the scene has camera, false otherwise.
  pub fn has_medium(&self) -> bool {
    for material in self.materials.iter() {
      if material.medium._type != HalaMediumType::NONE {
        return true;
      }
    }
    false
  }

  /// Check if the scene has medium with the specified type.
  /// param medium_type: The medium type.
  /// return: True if the scene has medium with the specified type, false otherwise.
  pub fn has_medium_with(&self, medium_type: HalaMediumType) -> bool {
    for material in self.materials.iter() {
      if material.medium._type == medium_type {
        return true;
      }
    }
    false
  }

  /// Check if the scene has transparent material.
  /// return: True if the scene has transparent material, false otherwise.
  pub fn has_transparent(&self) -> bool {
    for material in self.materials.iter() {
      if material.opacity < 1.0 - f32::EPSILON {
        return true;
      }
    }
    false
  }

  /// Update the node hierarchies.
  /// Set the children and world transform of each node.
  fn update_node_hierarchies(&mut self) {
    let mut temp_children = vec![vec![]; self.nodes.len()];
    let mut temp_world_transforms = vec![glam::Mat4::IDENTITY; self.nodes.len()];
    for (idx, node) in self.nodes.iter().enumerate() {
      if let Some(parent_idx) = node.parent {
        temp_children[parent_idx as usize].push(idx as u32);
        temp_world_transforms[idx] = temp_world_transforms[parent_idx as usize] * node.local_transform;
      } else {
        temp_world_transforms[idx] = node.local_transform;
      }
    }
    for (idx, node) in self.nodes.iter_mut().enumerate() {
      node.children = temp_children[idx].clone();
      node.world_transform = temp_world_transforms[idx];
    }
  }
}