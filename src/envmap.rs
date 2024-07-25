use std::rc::Rc;
use std::path::Path;

use image::GenericImageView;
use image::ImageReader;

use rayon::prelude::*;

use hala_gfx::{
  HalaContext,
  HalaFormat,
  HalaBuffer,
  HalaCommandBufferSet,
  HalaImage,
  HalaSampler,
};

use crate::error::HalaRendererError;

/// Environment map.
pub struct EnvMap {
  pub total_luminance: f32,
  pub image: HalaImage,
  pub sampler: HalaSampler,
  pub marginal_distribution_image: HalaImage,
  pub conditional_distribution_image: HalaImage,
  pub distribution_sampler: HalaSampler,
}

impl EnvMap {

  /// Create a new environment map with the given file path.
  /// param path: The file path.
  /// param context: The GFX context.
  /// param transfer_staging_buffer: The transfer staging buffer.
  /// param transfer_command_buffers: The transfer command buffers.
  /// return: The result.
  pub fn new_with_file<P: AsRef<Path>>(
    path: P,
    context: &HalaContext,
    transfer_staging_buffer: &HalaBuffer,
    transfer_command_buffers: &HalaCommandBufferSet,
  ) -> Result<Self, HalaRendererError> {
    let path = path.as_ref();
    let file_name = path.file_stem().ok_or(HalaRendererError::new("The file name is none!", None))?;

    // Open the image.
    let img = ImageReader::open(path)
      .map_err(|e| HalaRendererError::new(&format!("Failed to open image \"{}\".", path.to_string_lossy()), Some(Box::new(e))))?
      .with_guessed_format()
      .map_err(|e| HalaRendererError::new(&format!("Failed to guess the format of image \"{}\".", path.to_string_lossy()), Some(Box::new(e))))?
      .decode()
      .map_err(|e| HalaRendererError::new(&format!("Failed to decode image \"{}\".", path.to_string_lossy()), Some(Box::new(e))))?;
    let (width, height) = img.dimensions();

    // Check the color type.
    let format = match img.color() {
      image::ColorType::Rgba32F | image::ColorType::Rgb32F => HalaFormat::R32G32B32A32_SFLOAT,
      color_type => return Err(HalaRendererError::new(&format!("Unsupported color type \"{:?}\" for environment map.", color_type), None)),
    };

    // Perpare the image data.
    let validate_pixel_ch = |v: f32| -> Result<f32, HalaRendererError> {
      if v.is_nan() {
        return Err(HalaRendererError::new("The pixel value is NaN!", None));
      }
      if v.is_infinite() {
        return Err(HalaRendererError::new("The pixel value is infinite!", None));
      }
      Ok(v)
    };
    let img_buf = img.into_rgba32f();
    let mut data = Vec::new();
    for y in 0..height {
      for x in 0..width {
        let pixel = img_buf.get_pixel(x, y);

        let r = validate_pixel_ch(pixel[0])?;
        data.push(r);

        let g = validate_pixel_ch(pixel[1])?;
        data.push(g);

        let b = validate_pixel_ch(pixel[2])?;
        data.push(b);

        data.push(1.0);//pixel[3]);
      }
    }
    let cache_file_path = format!("./out/{}.dist_cache", file_name.to_string_lossy());
    let (total_sum, marginal_distribution, conditional_distribution) = if Path::new(&cache_file_path).exists() {
      let mut marginal_distribution: Vec<f32> = vec![0f32; height as usize];
      let mut conditional_distribution = vec![0f32; width as usize * height as usize];

      let file = std::fs::File::open(&cache_file_path)
        .map_err(|e| HalaRendererError::new(&format!("Failed to open file \"{}\".", &cache_file_path), Some(Box::new(e))))?;
      let mut reader = std::io::BufReader::new(file);

      let mut total_sum_buf = [0u8; 4];
      std::io::Read::read_exact(&mut reader, &mut total_sum_buf)
        .map_err(|e| HalaRendererError::new("Failed to read from file.", Some(Box::new(e))))?;
      let total_sum = f32::from_ne_bytes(total_sum_buf);

      for i in 0..height {
        let mut x = [0u8; 4];
        std::io::Read::read_exact(&mut reader, &mut x)
          .map_err(|e| HalaRendererError::new("Failed to read from file.", Some(Box::new(e))))?;
        marginal_distribution[i as usize] = f32::from_ne_bytes(x);
      }

      for i in 0..(width * height) {
        let mut x = [0u8; 4];
        std::io::Read::read_exact(&mut reader, &mut x)
          .map_err(|e| HalaRendererError::new("Failed to read from file.", Some(Box::new(e))))?;
        conditional_distribution[i as usize] = f32::from_ne_bytes(x);
      }
      (total_sum, marginal_distribution, conditional_distribution)
    } else {
      let (total_sum, marginal_distribution, conditional_distribution) = Self::build_distribution_maps(
        width as usize,
        height as usize,
        &img_buf
      )?;
      let file = std::fs::File::create(&cache_file_path)
        .map_err(|e| HalaRendererError::new(&format!("Failed to create file \"{}\".", &cache_file_path), Some(Box::new(e))))?;
      let mut writer = std::io::BufWriter::new(file);

      std::io::Write::write_all(&mut writer, &total_sum.to_ne_bytes())
        .map_err(|e| HalaRendererError::new("Failed to write to file.", Some(Box::new(e))))?;

      for v in marginal_distribution.iter() {
        std::io::Write::write_all(&mut writer, &v.to_ne_bytes())
          .map_err(|e| HalaRendererError::new("Failed to write to file.", Some(Box::new(e))))?;
      }

      for v in conditional_distribution.iter() {
        std::io::Write::write_all(&mut writer, &v.to_ne_bytes())
          .map_err(|e| HalaRendererError::new("Failed to write to file.", Some(Box::new(e))))?;
      }

      (total_sum, marginal_distribution, conditional_distribution)
    };

    // Create and upload the image.
    let image = HalaImage::new_2d(
      Rc::clone(&context.logical_device),
      hala_gfx::HalaImageUsageFlags::SAMPLED | hala_gfx::HalaImageUsageFlags::TRANSFER_DST,
      format,
      width,
      height,
      1,
      1,
      hala_gfx::HalaMemoryLocation::GpuOnly,
      &format!("env_texture_{}.image", file_name.to_string_lossy())
    )?;
    image.update_gpu_memory_with_buffer(
      data.as_slice(),
      hala_gfx::HalaPipelineStageFlags2::RAY_TRACING_SHADER,
      transfer_staging_buffer,
      transfer_command_buffers)?;
    let marginal_distribution_image = HalaImage::new_2d(
      Rc::clone(&context.logical_device),
      hala_gfx::HalaImageUsageFlags::SAMPLED | hala_gfx::HalaImageUsageFlags::TRANSFER_DST,
      HalaFormat::R32_SFLOAT,
      1,
      height,
      1,
      1,
      hala_gfx::HalaMemoryLocation::GpuOnly,
      &format!("env_texture_{}_marginal_distribution.image", file_name.to_string_lossy())
    )?;
    marginal_distribution_image.update_gpu_memory_with_buffer(
      marginal_distribution.as_slice(),
      hala_gfx::HalaPipelineStageFlags2::RAY_TRACING_SHADER,
      transfer_staging_buffer,
      transfer_command_buffers)?;
    let conditional_distribution_image = HalaImage::new_2d(
      Rc::clone(&context.logical_device),
      hala_gfx::HalaImageUsageFlags::SAMPLED | hala_gfx::HalaImageUsageFlags::TRANSFER_DST,
      HalaFormat::R32_SFLOAT,
      width,
      height,
      1,
      1,
      hala_gfx::HalaMemoryLocation::GpuOnly,
      &format!("env_texture_{}_conditional_distribution.image", file_name.to_string_lossy())
    )?;
    conditional_distribution_image.update_gpu_memory_with_buffer(
      conditional_distribution.as_slice(),
      hala_gfx::HalaPipelineStageFlags2::RAY_TRACING_SHADER,
      transfer_staging_buffer,
      transfer_command_buffers)?;

    // Create the sampler.
    let sampler = HalaSampler::new(
      Rc::clone(&context.logical_device),
      (hala_gfx::HalaFilter::LINEAR, hala_gfx::HalaFilter::LINEAR),
      hala_gfx::HalaSamplerMipmapMode::LINEAR,
      (hala_gfx::HalaSamplerAddressMode::REPEAT, hala_gfx::HalaSamplerAddressMode::REPEAT, hala_gfx::HalaSamplerAddressMode::REPEAT),
      0.0,
      false,
      0.0,
      (0.0, 0.0),
      &format!("env_texture_{}.sampler", file_name.to_string_lossy())
    )?;
    let distribution_sampler = HalaSampler::new(
      Rc::clone(&context.logical_device),
      (hala_gfx::HalaFilter::NEAREST, hala_gfx::HalaFilter::NEAREST),
      hala_gfx::HalaSamplerMipmapMode::NEAREST,
      (hala_gfx::HalaSamplerAddressMode::REPEAT, hala_gfx::HalaSamplerAddressMode::REPEAT, hala_gfx::HalaSamplerAddressMode::REPEAT),
      0.0,
      false,
      0.0,
      (0.0, 0.0),
      &format!("env_distribution_texture_{}.sampler", file_name.to_string_lossy())
    )?;

    Ok(Self {
      total_luminance: total_sum,
      image,
      sampler,
      marginal_distribution_image,
      conditional_distribution_image,
      distribution_sampler,
    })
  }

  /// Build the marginal and conditional distribution maps.
  /// param width: The width of the image.
  /// param height: The height of the image.
  /// param img_buf: The image buffer.
  /// return: The result.
  fn build_distribution_maps(
    width: usize,
    height: usize,
    img_buf: &image::ImageBuffer<image::Rgba<f32>, Vec<f32>>
  ) -> Result<(f32, Vec<f32>, Vec<f32>), HalaRendererError> {
    // TV BT.601 for SDR.
    // let luminance = |r: f32, g: f32, b: f32| -> f32 {
    //   (0.299 * r * r + 0.587 * g * g + 0.114 * b * b).sqrt()
    // };
    // TV BT.709 for HDR.
    let luminance = |r: f32, g: f32, b: f32| -> f32 {
     0.212671 * r + 0.715160 * g + 0.072169 * b
    };
    let lower_bound = |array: &[f32], lower: usize, upper: usize, value: f32| -> usize {
      let mut lower = lower;
      let mut upper = upper;
      while lower < upper {
        let mid = (lower + upper) / 2;
        if array[mid] < value {
          lower = mid + 1;
        } else {
          upper = mid;
        }
      }

      lower
    };

    let mut pdf_2d = vec![0f32; width * height];
    let mut cdf_2d = vec![0f32; width * height];
    let mut pdf_1d = vec![0f32; height];
    let mut cdf_1d = vec![0f32; height];

    let mut marginal_distribution = vec![0f32; height];
    let mut conditional_distribution = vec![0f32; width * height];

    let total_sum = img_buf.pixels().fold(0f32, |acc, pixel| acc + luminance(pixel[0], pixel[1], pixel[2]));

    pdf_2d.par_chunks_mut(width)
      .zip(cdf_2d.par_chunks_mut(width))
      .zip(pdf_1d.par_iter_mut())
      .enumerate()
      .for_each(|(v, ((pdf_2d_row, cdf_2d_row), pdf_1d_value))| {
        let mut row_weight_sum = 0f32;
        for u in 0..width {
          let pixel = img_buf.get_pixel(u as u32, v as u32);
          let weight = luminance(pixel[0], pixel[1], pixel[2]);

          pdf_2d_row[u] = weight;
          row_weight_sum += weight;
          cdf_2d_row[u] = row_weight_sum;
        }

        // convert to range [0, 1].
        for u in 0..width {
          pdf_2d_row[u] /= row_weight_sum;
          cdf_2d_row[u] /= row_weight_sum;
        }

        *pdf_1d_value = row_weight_sum;
      });
    cdf_1d.iter_mut().enumerate().fold(0f32, |col_weight_sum, (v, cdf)| {
      *cdf = col_weight_sum + pdf_1d[v];
      *cdf
    });
    let col_weight_sum = cdf_1d[height - 1];
    pdf_1d.par_iter_mut().zip(cdf_1d.par_iter_mut()).for_each(|(pdf, cdf)| {
      *pdf /= col_weight_sum;
      *cdf /= col_weight_sum;
    });

    // precalculate row and col to avoid binary search during lookup in the shader.
    marginal_distribution.par_iter_mut().enumerate().for_each(|(v, marginal)| {
      let inv_height = 1.0 / height as f32;
      let row = lower_bound(
        &cdf_1d,
        0,
        height,
        (v + 1) as f32 * inv_height);
      *marginal = row as f32 * inv_height;
    });

    conditional_distribution.par_chunks_mut(width).enumerate().for_each(|(v, conditional)| {
      (0..width).for_each(|u| {
        let inv_width = 1.0 / width as f32;
        let col = lower_bound(
          &cdf_2d,
          v * width,
          (v + 1) * width,
          (u + 1) as f32 * inv_width) - v * width;
        conditional[u] = col as f32 * inv_width;
      });
    });

    // let mut col_weight_sum = 0f32;
    // for v in 0..height {
    //   let mut row_weight_sum = 0f32;
    //   for u in 0..width {
    //     let pixel = img_buf.get_pixel(u as u32, v as u32);
    //     let weight = luminance(pixel[0] as f32, pixel[1] as f32, pixel[2] as f32);

    //     pdf_2d[(v * width + u) as usize] = weight;
    //     row_weight_sum += weight;
    //     cdf_2d[(v * width + u) as usize] = row_weight_sum;
    //   }

    //   // convert to range [0, 1].
    //   for u in 0..width {
    //     pdf_2d[(v * width + u) as usize] /= row_weight_sum;
    //     cdf_2d[(v * width + u) as usize] /= row_weight_sum;
    //   }

    //   pdf_1d[v as usize] = row_weight_sum;
    //   col_weight_sum += row_weight_sum;
    //   cdf_1d[v as usize] = col_weight_sum;
    // }

    // // convert to range [0, 1].
    // for v in 0..height {
    //   pdf_1d[v as usize] /= col_weight_sum;
    //   cdf_1d[v as usize] /= col_weight_sum;
    // }

    // // precalculate row and col to avoid binary search during lookup in the shader.
    // for v in 0..height {
    //   let inv_height = 1.0 / height as f32;
    //   let row = lower_bound(
    //     &cdf_1d,
    //     0,
    //     height,
    //     (v + 1) as f32 * inv_height);
    //   marginal_distribution[v] = row as f32 * inv_height;
    //   //marginal_distribution[v].y = pdf_1d[v];
    // }

    // for v in 0..height {
    //   for u in 0..width {
    //     let inv_width = 1.0 / width as f32;
    //     let col = lower_bound(
    //       &cdf_2d,
    //       v * width,
    //       (v + 1) * width,
    //       (u + 1) as f32 * inv_width) - v * width;
    //     conditional_distribution[v * width + u] = col as f32 * inv_width;
    //     //conditional_distribution[v * width + u].y = pdf_2d[(v * width + u) as usize];
    //   }
    // }

    Ok((total_sum, marginal_distribution, conditional_distribution))
  }

}