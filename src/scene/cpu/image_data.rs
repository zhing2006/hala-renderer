use std::path::Path;

use image::GenericImageView;

use hala_gfx::HalaFormat;

use crate::error::HalaRendererError;

pub enum HalaImageDataType {
  ByteData(Vec<u8>),
  FloatData(Vec<f32>),
}

pub struct HalaImageData {
  pub format: HalaFormat,
  pub width: u32,
  pub height: u32,
  pub data_type: HalaImageDataType,
  pub num_of_bytes: usize,
}

impl HalaImageData {
  /// Create a new texture with the given file path.
  /// param path: The file path.
  /// return: The result.
  pub fn new_with_file<P: AsRef<Path>>(path: P) -> Result<Self, HalaRendererError> {
    let path = path.as_ref();

    let img = image::open(path)
      .map_err(|e| HalaRendererError::new(&format!("Failed to open image \"{}\".", path.to_string_lossy()), Some(Box::new(e))))?;
    let (width, height) = img.dimensions();

    let (format, data, num_of_bytes) = match img.color() {
      image::ColorType::Rgb8 => {
        let data = img.into_bytes();
        let num_of_bytes = data.len();
        (HalaFormat::R8G8B8_UNORM, HalaImageDataType::ByteData(data), num_of_bytes)
      },
      image::ColorType::Rgba8 => {
        let data = img.into_bytes();
        let num_of_bytes = data.len();
        (HalaFormat::B8G8R8A8_UNORM, HalaImageDataType::ByteData(data), num_of_bytes)
      },
      image::ColorType::Rgba32F => {
        let data = img.into_rgba32f().into_vec();
        let num_of_bytes = data.len() * std::mem::size_of::<f32>();
        (HalaFormat::R32G32B32A32_SFLOAT, HalaImageDataType::FloatData(data), num_of_bytes)
      },
      color_type => return Err(HalaRendererError::new(&format!("Unsupported color type: {:?}", color_type), None)),
    };

    Ok(Self {
      format,
      width,
      height,
      data_type: data,
      num_of_bytes,
    })
  }
}