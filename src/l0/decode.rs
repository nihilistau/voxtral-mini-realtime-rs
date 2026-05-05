//! L0-based decoder pipeline: USM KV cache + VHT2 + Q4 attention on iGPU.
//!
//! This module ties everything together for the zero-copy iGPU decode path:
//!
//! 1. KV cache lives in USM shared memory (zeMemAllocShared)
//! 2. VHT2 compress/decompress operates directly on USM pointers (CPU, zero-copy)
//! 3. Q4 attention kernel dispatches on the same USM buffers (GPU, zero-copy)
//!
//! The result: **zero** explicit data copies between VHT2 and attention.
//! On Intel UMA, CPU and GPU access the same DRAM — the only cost is cache
//! coherence (automatic on modern Intel), not DMA/PCIe/staging buffers.

use super::device::L0Context;
use super::kernel::{L0Kernel, L0Module};
use super::ocl_compile::OclCompiler;
use super::spirv_gen::OPENCL_Q4_MATMUL;
use super::usm::{UsmAllocator, UsmKvCache};
use crate::models::layers::shannon_prime::{compress_kv_vector, decompress_kv_vector, BandConfig};
use anyhow::Result;

/// Complete L0 decode context: device, allocator, compiled kernels, and KV cache.
pub struct L0DecodeContext {
    pub ctx: L0Context,
    pub allocator: UsmAllocator,
    pub kv_cache: UsmKvCache,
    pub q4_matmul_module: L0Module,
    pub band_config: BandConfig,
    /// Pre-created kernel pool — avoids zeKernelCreate overhead per dispatch.
    /// Pool size = max ops in a single batch (3 for QKV projections).
    kernel_pool: Vec<L0Kernel>,
}

/// Configuration for the L0 decoder.
pub struct L0DecodeConfig {
    pub batch: usize,
    pub kv_heads: usize,
    pub max_seq_len: usize,
    pub head_dim: usize,
    pub band_config: BandConfig,
}

impl L0DecodeContext {
    /// Initialize the full L0 decode pipeline.
    ///
    /// This:
    /// 1. Creates L0 context (discovers iGPU)
    /// 2. Compiles Q4 matmul kernel via OpenCL
    /// 3. Allocates USM KV cache
    pub fn new(config: L0DecodeConfig) -> Result<Self> {
        let ctx = L0Context::new()?;
        let allocator = UsmAllocator::new(&ctx);

        // Compile Q4 matmul kernel
        tracing::info!("Compiling Q4 matmul kernel for L0...");
        let compiler = OclCompiler::new()?;
        let binary = compiler.compile_to_binary(OPENCL_Q4_MATMUL, "-cl-std=CL2.0 -cl-fast-relaxed-math")?;
        let q4_matmul_module = L0Module::from_native(&ctx, &binary)?;
        tracing::info!("Q4 matmul kernel compiled ({} bytes)", binary.len());

        // Allocate KV cache in USM shared memory
        let kv_cache = UsmKvCache::new(
            &allocator,
            config.batch,
            config.kv_heads,
            config.max_seq_len,
            config.head_dim,
        )?;

        // Pre-create kernel pool (max 3 for QKV batch, avoids 156× zeKernelCreate per token)
        let pool_size = 3;
        let mut kernel_pool = Vec::with_capacity(pool_size);
        for _ in 0..pool_size {
            let k = q4_matmul_module.create_kernel("q4_matmul")?;
            k.set_group_size(16, 16, 1)?;
            kernel_pool.push(k);
        }
        tracing::info!("Pre-created kernel pool ({} kernels)", pool_size);

        Ok(L0DecodeContext {
            ctx,
            allocator,
            kv_cache,
            q4_matmul_module,
            band_config: config.band_config,
            kernel_pool,
        })
    }

    /// Append new KV vectors and apply Shannon-Prime VHT2 compression.
    ///
    /// This is the zero-copy path:
    /// 1. Write new K/V vectors into USM buffer at current_len position
    /// 2. Apply VHT2 compress on the raw USM pointer (CPU, no copy)
    /// 3. Apply VHT2 decompress (lossy reconstruction, CPU, no copy)
    /// 4. The GPU can now read the compressed cache directly for attention
    ///
    /// # Safety
    /// Must be called after any GPU work on the KV cache has completed.
    pub unsafe fn append_kv_with_vht2(
        &mut self,
        key_data: &[f32],   // [batch, kv_heads, 1, head_dim]
        value_data: &[f32], // [batch, kv_heads, 1, head_dim]
    ) {
        let batch = self.kv_cache.batch;
        let kv_heads = self.kv_cache.kv_heads;
        let head_dim = self.kv_cache.head_dim;
        let seq_idx = self.kv_cache.current_len;

        debug_assert_eq!(key_data.len(), batch * kv_heads * head_dim);
        debug_assert_eq!(value_data.len(), batch * kv_heads * head_dim);
        debug_assert!(seq_idx < self.kv_cache.max_seq_len);

        // Write new KV vectors into USM and apply VHT2 in-place
        for b in 0..batch {
            for h in 0..kv_heads {
                let src_offset = (b * kv_heads + h) * head_dim;

                // --- Key ---
                let key_slice = self.kv_cache.key_slice_mut(b, h, seq_idx);
                key_slice.copy_from_slice(&key_data[src_offset..src_offset + head_dim]);
                // VHT2 compress + decompress in-place on USM pointer — ZERO COPY
                compress_kv_vector(key_slice, &self.band_config);
                decompress_kv_vector(key_slice);

                // --- Value ---
                let val_slice = self.kv_cache.value_slice_mut(b, h, seq_idx);
                val_slice.copy_from_slice(&value_data[src_offset..src_offset + head_dim]);
                // VHT2 compress + decompress in-place on USM pointer — ZERO COPY
                compress_kv_vector(val_slice, &self.band_config);
                decompress_kv_vector(val_slice);
            }
        }

        self.kv_cache.current_len += 1;
    }

    /// Dispatch Q4 matmul on the iGPU using USM buffers.
    ///
    /// Computes: output = input × weights^T (with Q4 dequantization)
    /// All buffers must be USM allocations from this context's allocator.
    /// Uses pre-created kernel from pool (zero kernel creation overhead).
    pub fn q4_matmul(
        &self,
        weights_ptr: *const u8,
        input_ptr: *const f32,
        output_ptr: *mut f32,
        b: u32,
        m: u32,
        k: u32,
        n: u32,
    ) -> Result<()> {
        let blocks_per_row = k / 32;
        let kernel = &self.kernel_pool[0];

        kernel.set_arg_ptr(0, weights_ptr)?;
        kernel.set_arg_ptr(1, input_ptr)?;
        kernel.set_arg_ptr(2, output_ptr)?;
        kernel.set_arg_scalar(3, &b)?;
        kernel.set_arg_scalar(4, &m)?;
        kernel.set_arg_scalar(5, &k)?;
        kernel.set_arg_scalar(6, &n)?;
        kernel.set_arg_scalar(7, &blocks_per_row)?;

        let groups_x = (n + 15) / 16;
        let groups_y = (b * m + 15) / 16;

        kernel.dispatch(&self.ctx, groups_x, groups_y, 1)?;
        Ok(())
    }

    /// Dispatch multiple Q4 matmuls in a single command list (one fence sync).
    ///
    /// Each entry: (weights_ptr, input_ptr, output_ptr, B, M, K, N)
    /// All buffers must be USM. The GPU executes them sequentially in one submission.
    /// Uses pre-created kernel pool — each op gets its own kernel object since L0
    /// does NOT capture args at append time (the kernel IS the arg state).
    ///
    /// Max ops per call: 3 (pool size). Panics if exceeded.
    pub fn q4_matmul_batch(
        &self,
        ops: &[(* const u8, *const f32, *mut f32, u32, u32, u32, u32)],
    ) -> Result<()> {
        if ops.is_empty() {
            return Ok(());
        }
        assert!(ops.len() <= self.kernel_pool.len(),
            "Batch size {} exceeds kernel pool size {}", ops.len(), self.kernel_pool.len());

        let cmd_list = self.ctx.create_command_list()?;

        for (idx, &(w_ptr, i_ptr, o_ptr, b, m, k, n)) in ops.iter().enumerate() {
            let blocks_per_row = k / 32;
            let kernel = &self.kernel_pool[idx];

            kernel.set_arg_ptr(0, w_ptr)?;
            kernel.set_arg_ptr(1, i_ptr)?;
            kernel.set_arg_ptr(2, o_ptr)?;
            kernel.set_arg_scalar(3, &b)?;
            kernel.set_arg_scalar(4, &m)?;
            kernel.set_arg_scalar(5, &k)?;
            kernel.set_arg_scalar(6, &n)?;
            kernel.set_arg_scalar(7, &blocks_per_row)?;

            let groups_x = (n + 15) / 16;
            let groups_y = (b * m + 15) / 16;
            kernel.append_to_command_list(cmd_list, groups_x, groups_y, 1)?;
        }

        // Barrier + close + submit + sync
        unsafe {
            use super::sys;
            use std::ptr;
            sys::zeCommandListAppendBarrier(cmd_list, ptr::null_mut(), 0, ptr::null());
            sys::zeCommandListClose(cmd_list);
        }
        self.ctx.submit_and_sync(cmd_list)?;
        unsafe { super::sys::zeCommandListDestroy(cmd_list); }
        Ok(())
    }

    /// Reset KV cache for a new sequence.
    pub fn reset(&mut self) {
        self.kv_cache.reset();
    }

    /// Current sequence length in the KV cache.
    pub fn seq_len(&self) -> usize {
        self.kv_cache.current_len
    }
}
