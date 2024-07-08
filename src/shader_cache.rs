use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Once;
use std::path::Path;

use hala_gfx::{
  HalaLogicalDevice,
  HalaShaderStageFlags,
  HalaRayTracingShaderGroupType,
  HalaShader,
};

use crate::error::HalaRendererError;

/// The shader cache.
pub struct HalaShaderCache {
  shader_dir: String,
  cache: HashMap<String, Rc<RefCell<HalaShader>>>,
}

/// The implementation of the shader cache.
impl HalaShaderCache {

  /// Create a new shader cache.
  /// return: The shader cache.
  fn new() -> Self {
    Self {
      shader_dir: String::from("./"),
      cache: HashMap::new(),
    }
  }

  /// Get the shader cache singleton instance.
  pub fn get_instance() -> Rc<RefCell<HalaShaderCache>> {
    static mut SINGLETON: Option<Rc<RefCell<HalaShaderCache>>> = None;
    static ONCE: Once = Once::new();

    unsafe {
      ONCE.call_once(|| {
        let singleton = HalaShaderCache::new();
        SINGLETON = Some(Rc::new(RefCell::new(singleton)));
      });
      SINGLETON.clone().unwrap()
    }
  }

  /// Set the shader directory.
  pub fn set_shader_dir<P: AsRef<Path>>(&mut self, shader_dir: P) {
    self.shader_dir = shader_dir.as_ref().to_string_lossy().to_string();
  }

  /// Load a shader.
  /// param logical_device: The logical device.
  /// param file_path: The shader file path.
  /// param stage: The shader stage.
  /// param rt_group_type: The ray tracing shader group type.
  /// param debug_name: The debug name.
  /// return: The shader.
  pub fn load<P: AsRef<Path>>(
    &mut self,
    logical_device: Rc<RefCell<HalaLogicalDevice>>,
    file_path: P,
    stage: HalaShaderStageFlags,
    rt_group_type: HalaRayTracingShaderGroupType,
    debug_name: &str,
  ) -> Result<Rc<RefCell<HalaShader>>, HalaRendererError> {
    let file_path = file_path.as_ref();
    let file_path = if file_path.is_absolute() {
      file_path.to_string_lossy().to_string()
    } else {
      format!("{}/{}", self.shader_dir, file_path.to_string_lossy())
    };

    if let Some(shader) = self.cache.get(&file_path) {
      return Ok(Rc::clone(shader));
    }

    let shader = Rc::new(RefCell::new(
      HalaShader::with_file(
        logical_device,
        &file_path,
        stage,
        rt_group_type,
        debug_name,
      )?
    ));
    self.cache.insert(file_path, Rc::clone(&shader));

    Ok(shader)
  }

  /// Create a shader from memory.
  /// param logical_device: The logical device.
  /// param code: The compiled shader code.
  /// param stage: The shader stage.
  /// param rt_group_type: The ray tracing shader group type.
  /// param debug_name: The debug name. The debug name will be used as the key of the shader.
  pub fn from_memory(
    &mut self,
    logical_device: Rc<RefCell<HalaLogicalDevice>>,
    code: &[u8],
    stage: HalaShaderStageFlags,
    rt_group_type: HalaRayTracingShaderGroupType,
    debug_name: &str,
  ) -> Result<Rc<RefCell<HalaShader>>, HalaRendererError> {
    let shader = Rc::new(RefCell::new(
      HalaShader::new(
        logical_device,
        code,
        stage,
        rt_group_type,
        debug_name,
      )?
    ));

    if let Some(shader) = self.cache.get(debug_name) {
      return Ok(Rc::clone(shader));
    }

    self.cache.insert(debug_name.to_string(), Rc::clone(&shader));

    Ok(shader)
  }

  /// Try to get a loaded shader.
  /// param file_path: The shader file path.
  /// return: The shader or None.
  pub fn get(&self, file_path: &str) -> Option<Rc<RefCell<HalaShader>>> {
    self.cache.get(file_path).map(|shader| Rc::clone(shader))
  }

  /// Remove the specified shader.
  /// param file_path: The shader file path.
  pub fn remove(&mut self, file_path: &str) {
    self.cache.remove(file_path);
  }

  /// Clear all loaded shaders.
  pub fn clear(&mut self) {
    self.cache.clear();
  }

}