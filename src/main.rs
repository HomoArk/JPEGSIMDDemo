use image::codecs::jpeg::JpegEncoder;
use image::{ExtendedColorType, GenericImage, ImageBuffer, Rgb, RgbImage};
use std::fs::File;
use std::time::Instant;
use rayon::prelude::*;
use anyhow::Result;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
use std::io::Write;

// Image dimensions constants
const ORIGINAL_WIDTH: u32 = 19000;
const ORIGINAL_HEIGHT: u32 = 19000;
const NEW_WIDTH: u32 = 20000;
const NEW_HEIGHT: u32 = 20000;
const OFFSET_X: u32 = 500;
const OFFSET_Y: u32 = 500;
const JPEG_QUALITY: u8 = 100;
const BYTES_PER_PIXEL: u32 = 3;

#[cfg(target_arch = "x86_64")]
const SIMD_VECTOR_SIZE: usize = 32; // 256-bit AVX2
#[cfg(target_arch = "aarch64")]
const SIMD_VECTOR_SIZE: usize = 16; // 128-bit NEON
const MIN_BYTES_FOR_SIMD: usize = SIMD_VECTOR_SIZE * 2;

#[derive(Debug)]
struct BenchmarkResults {
    copy_time: std::time::Duration,
    encode_time: std::time::Duration,
    total_time: std::time::Duration,
}

#[derive(Debug)]
enum SimdSupport {
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Sse4,
    #[cfg(target_arch = "aarch64")]
    Neon,
    None,
}

fn get_simd_support() -> SimdSupport {
    #[cfg(target_arch = "x86_64")] {
        if is_x86_feature_detected!("avx2") {
            SimdSupport::Avx2
        } else if is_x86_feature_detected!("sse4.1") {
            SimdSupport::Sse4
        } else {
            SimdSupport::None
        }
    }
    #[cfg(target_arch = "aarch64")] {
        if std::arch::is_aarch64_feature_detected!("neon") {
            SimdSupport::Neon
        } else {
            SimdSupport::None
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))] {
        SimdSupport::None
    }
}

// SIMD implementations for different architectures
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn process_pixels_neon(pixels: &mut [u8], brightness: f32) {
    let brightness_f32 = vdupq_n_f32(brightness);
    let mut i = 0;

    while i + 16 <= pixels.len() {
        // Load 16 pixels
        let pixels_u8 = vld1q_u8(pixels.as_ptr().add(i));

        // Convert first 8 pixels to f32
        let pixels_u16_low = vmovl_u8(vget_low_u8(pixels_u8));
        let pixels_u32_low = vmovl_u16(vget_low_u16(pixels_u16_low));
        let mut pixels_f32_low = vcvtq_f32_u32(pixels_u32_low);

        // Convert second 8 pixels to f32
        let pixels_u16_high = vmovl_u8(vget_high_u8(pixels_u8));
        let pixels_u32_high = vmovl_u16(vget_low_u16(pixels_u16_high));
        let mut pixels_f32_high = vcvtq_f32_u32(pixels_u32_high);

        // Apply brightness
        pixels_f32_low = vmulq_f32(pixels_f32_low, brightness_f32);
        pixels_f32_high = vmulq_f32(pixels_f32_high, brightness_f32);

        // Convert back to u8 with saturation
        let result_u32_low = vcvtq_u32_f32(pixels_f32_low);
        let result_u32_high = vcvtq_u32_f32(pixels_f32_high);
        let result_u16 = vcombine_u16(vqmovn_u32(result_u32_low), vqmovn_u32(result_u32_high));
        let result_u8 = vqmovn_u16(result_u16);

        // Store results
        vst1q_u8(pixels.as_mut_ptr().add(i), vcombine_u8(result_u8, result_u8));
        i += 16;
    }

    // Handle remaining pixels
    while i < pixels.len() {
        pixels[i] = (pixels[i] as f32 * brightness).clamp(0.0, 255.0) as u8;
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn process_pixels_avx2(pixels: &mut [u8], brightness: f32) {
    let brightness_factor = _mm256_set1_ps(brightness);
    let mut i = 0;

    while i + 32 <= pixels.len() {
        let pixels_ptr = pixels.as_ptr().add(i);
        let pixels_avx = _mm256_loadu_si256(pixels_ptr as *const __m256i);

        // Process low 16 bytes
        let pixels_low = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(
            _mm256_extracti128_si256(pixels_avx, 0)
        ));
        let processed_low = _mm256_mul_ps(pixels_low, brightness_factor);

        // Process high 16 bytes
        let pixels_high = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(
            _mm256_extracti128_si256(pixels_avx, 1)
        ));
        let processed_high = _mm256_mul_ps(pixels_high, brightness_factor);

        // Convert back and combine
        let result_low = _mm256_cvtps_epi32(processed_low);
        let result_high = _mm256_cvtps_epi32(processed_high);
        let result = _mm256_packus_epi32(result_low, result_high);
        let final_result = _mm256_packus_epi16(result, result);

        _mm256_storeu_si256(pixels.as_mut_ptr().add(i) as *mut __m256i, final_result);
        i += 32;
    }

    // Handle remaining pixels
    while i < pixels.len() {
        pixels[i] = (pixels[i] as f32 * brightness).clamp(0.0, 255.0) as u8;
        i += 1;
    }
}

#[inline(always)]
unsafe fn copy_row_simd(src: &[u8], dst: &mut [u8], width: usize, simd_support: &SimdSupport) {
    match simd_support {
        #[cfg(target_arch = "aarch64")]
        SimdSupport::Neon => {
            let mut i = 0;
            while i + 16 <= width {
                let src_ptr = src.as_ptr().add(i);
                let dst_ptr = dst.as_mut_ptr().add(i);
                let data = vld1q_u8(src_ptr);
                vst1q_u8(dst_ptr, data);
                i += 16;
            }
            while i < width {
                dst[i] = src[i];
                i += 1;
            }
        },
        #[cfg(target_arch = "x86_64")]
        SimdSupport::Avx2 => {
            let mut i = 0;
            while i + 32 <= width {
                let src_ptr = src.as_ptr().add(i);
                let dst_ptr = dst.as_mut_ptr().add(i);
                let data = _mm256_loadu_si256(src_ptr as *const __m256i);
                _mm256_storeu_si256(dst_ptr as *mut __m256i, data);
                i += 32;
            }
            while i < width {
                dst[i] = src[i];
                i += 1;
            }
        },
        _ => dst.copy_from_slice(src),
    }
}

fn process_image() -> Result<BenchmarkResults> {
    let start_time = Instant::now();
    let simd_support = get_simd_support();

    let mut original_img = RgbImage::new(ORIGINAL_WIDTH, ORIGINAL_HEIGHT);
    let mut new_img = RgbImage::new(NEW_WIDTH, NEW_HEIGHT);

    for pixel in original_img.pixels_mut() {
        pixel[0] = 128;
        pixel[1] = 128;
        pixel[2] = 128;
    }

    let copy_start = Instant::now();
    let row_bytes = (ORIGINAL_WIDTH * BYTES_PER_PIXEL) as usize;
    let offset_bytes = (OFFSET_X * BYTES_PER_PIXEL) as usize;

    // Parallel processing with SIMD
    new_img.par_chunks_mut((NEW_WIDTH * BYTES_PER_PIXEL) as usize)
        .enumerate()
        .for_each(|(y, row)| {
            if y >= OFFSET_Y as usize && y < (OFFSET_Y + ORIGINAL_HEIGHT) as usize {
                let src_y = y - OFFSET_Y as usize;
                if src_y < ORIGINAL_HEIGHT as usize {
                    let src_row = &original_img.as_raw()[(src_y * row_bytes)..((src_y + 1) * row_bytes)];
                    let dst_slice = &mut row[offset_bytes..offset_bytes + row_bytes];

                    unsafe {
                        copy_row_simd(src_row, dst_slice, row_bytes, &simd_support);
                        // match simd_support {
                        //     #[cfg(target_arch = "aarch64")]
                        //     SimdSupport::Neon => process_pixels_neon(dst_slice, 1.1),
                        //     #[cfg(target_arch = "x86_64")]
                        //     SimdSupport::Avx2 => process_pixels_avx2(dst_slice, 1.1),
                        //     _ => dst_slice.iter_mut().for_each(|p| *p = (*p as f32 * 1.1).clamp(0.0, 255.0) as u8),
                        // }
                    }
                }
            }
        });

    let copy_time = copy_start.elapsed();
    let encode_start = Instant::now();

    // Encode to JPEG
    let out_file = File::create("output.jpg")?;
    let mut buffered_writer = std::io::BufWriter::with_capacity(16384, out_file);

    let jpeg_data = turbojpeg::compress_image(&new_img, JPEG_QUALITY as i32, turbojpeg::Subsamp::None)?;

    buffered_writer.write(&jpeg_data)?;
    buffered_writer.flush()?;

    Ok(BenchmarkResults {
        copy_time,
        encode_time: encode_start.elapsed(),
        total_time: start_time.elapsed(),
    })
}

fn main() -> Result<()> {
    println!("Architecture: {}", std::env::consts::ARCH);
    println!("SIMD Support: {:?}", get_simd_support());

    // Run benchmarks
    let iterations = 3;
    let mut total_results = Vec::with_capacity(iterations);

    println!("\nRunning {} iterations...", iterations);

    for i in 0..iterations {
        println!("\nIteration {}:", i + 1);
        let results = process_image()?;
        println!("  Copy time: {:?}", results.copy_time);
        println!("  Encode time: {:?}", results.encode_time);
        println!("  Total time: {:?}", results.total_time);
        total_results.push(results);
    }

    let avg_copy = total_results.iter().map(|r| r.copy_time).sum::<std::time::Duration>() / iterations as u32;
    let avg_encode = total_results.iter().map(|r| r.encode_time).sum::<std::time::Duration>() / iterations as u32;
    let avg_total = total_results.iter().map(|r| r.total_time).sum::<std::time::Duration>() / iterations as u32;

    println!("\nAverage times:");
    println!("  Copy: {:?}", avg_copy);
    println!("  Encode: {:?}", avg_encode);
    println!("  Total: {:?}", avg_total);

    Ok(())
}