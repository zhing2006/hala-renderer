use thiserror::Error;

/// The error type of the hala-renderer crate.
#[derive(Error, Debug)]
pub struct HalaRendererError {
  msg: String,
  #[source]
  source: Option<Box<dyn std::error::Error + Send + Sync>>,
}

/// The implementation of the error type of the hala-renderer crate.
impl HalaRendererError {
  /// Create a new error.
  /// param msg: The message of the error.
  /// param source: The source of the error.
  /// return: The error.
  pub fn new(msg: &str, source: Option<Box<dyn std::error::Error + Send + Sync>>) -> Self {
    Self {
      msg: msg.to_string(),
      source,
    }
  }
  pub fn message(&self) -> &str {
    &self.msg
  }
}

impl std::convert::From<hala_gfx::HalaGfxError> for HalaRendererError {
  fn from(err: hala_gfx::HalaGfxError) -> Self {
    Self {
      msg: err.message().to_string(),
      source: Some(Box::new(err)),
    }
  }
}

/// The implementation Display trait for the error type of the hala-renderer crate.
impl std::fmt::Display for HalaRendererError {
  /// Format the error.
  /// param f: The formatter.
  /// return: The result.
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{}", self.msg)
  }
}