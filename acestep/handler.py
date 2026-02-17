"""
Business Logic Handler
Encapsulates all data processing and business logic as a bridge between model and UI
"""
import os
import sys

# Disable tokenizers parallelism to avoid fork warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import math
from copy import deepcopy
import tempfile
import traceback
import re
import random
import uuid
import hashlib
import json
import threading
from typing import Optional, Dict, Any, Tuple, List, Union

import torch
import torchaudio
import soundfile as sf
import time
from tqdm import tqdm
from loguru import logger
import warnings

from acestep.constants import DEFAULT_DIT_INSTRUCTION, SFT_GEN_PROMPT
from acestep.core.generation.handler import (
    AudioCodesMixin,
    BatchPrepMixin,
    ConditioningBatchMixin,
    ConditioningEmbedMixin,
    ConditioningMaskMixin,
    ConditioningTargetMixin,
    ConditioningTextMixin,
    DiffusionMixin,
    GenerateMusicExecuteMixin,
    GenerateMusicRequestMixin,
    InitServiceMixin,
    IoAudioMixin,
    LyricScoreMixin,
    LyricTimestampMixin,
    LoraManagerMixin,
    MemoryUtilsMixin,
    MetadataMixin,
    PaddingMixin,
    ProgressMixin,
    PromptMixin,
    TaskUtilsMixin,
    VaeDecodeChunksMixin,
    VaeDecodeMixin,
    VaeEncodeChunksMixin,
    VaeEncodeMixin,
    ServiceGenerateRequestMixin,
    ServiceGenerateExecuteMixin,
    ServiceGenerateOutputsMixin,
)
from acestep.gpu_config import get_effective_free_vram_gb


warnings.filterwarnings("ignore")


class AceStepHandler(
    DiffusionMixin,
    GenerateMusicExecuteMixin,
    GenerateMusicRequestMixin,
    AudioCodesMixin,
    BatchPrepMixin,
    ConditioningBatchMixin,
    ConditioningEmbedMixin,
    ConditioningMaskMixin,
    ConditioningTargetMixin,
    ConditioningTextMixin,
    IoAudioMixin,
    InitServiceMixin,
    LyricScoreMixin,
    LyricTimestampMixin,
    LoraManagerMixin,
    MemoryUtilsMixin,
    MetadataMixin,
    PaddingMixin,
    ProgressMixin,
    PromptMixin,
    TaskUtilsMixin,
    VaeDecodeChunksMixin,
    VaeDecodeMixin,
    VaeEncodeChunksMixin,
    VaeEncodeMixin,
    ServiceGenerateRequestMixin,
    ServiceGenerateExecuteMixin,
    ServiceGenerateOutputsMixin,
):
    """ACE-Step Business Logic Handler"""
    
    def __init__(self):
        """Initialize runtime model handles, feature flags, and generation state."""
        self.model = None
        self.config = None
        self.device = "cpu"
        self.dtype = torch.float32  # Will be set based on device in initialize_service

        # VAE for audio encoding/decoding
        self.vae = None
        
        # Text encoder and tokenizer
        self.text_encoder = None
        self.text_tokenizer = None
        
        # Silence latent for initialization
        self.silence_latent = None
        
        # Sample rate
        self.sample_rate = 48000
        
        # Reward model (temporarily disabled)
        self.reward_model = None
        
        # Batch size
        self.batch_size = 2
        
        # Custom layers config
        self.custom_layers_config = {2: [6], 3: [10, 11], 4: [3], 5: [8, 9], 6: [8]}
        self.offload_to_cpu = False
        self.offload_dit_to_cpu = False
        self.compiled = False
        self.current_offload_cost = 0.0
        self.disable_tqdm = os.environ.get("ACESTEP_DISABLE_TQDM", "").lower() in ("1", "true", "yes") or not getattr(sys.stderr, 'isatty', lambda: False)()
        self.debug_stats = os.environ.get("ACESTEP_DEBUG_STATS", "").lower() in ("1", "true", "yes")
        self._last_diffusion_per_step_sec: Optional[float] = None
        self._progress_estimates_lock = threading.Lock()
        self._progress_estimates = {"records": []}
        self._progress_estimates_path = os.path.join(
            self._get_project_root(),
            ".cache",
            "acestep",
            "progress_estimates.json",
        )
        self._load_progress_estimates()
        self.last_init_params = None
        
        # Quantization state - tracks if model is quantized (int8_weight_only, fp8_weight_only, or w8a8_dynamic)
        # Populated during initialize_service, remains None if quantization is disabled
        self.quantization = None
        
        # LoRA state
        self.lora_loaded = False
        self.use_lora = False
        self.lora_scale = 1.0  # LoRA influence scale (0-1), mirrors active adapter's scale
        self._base_decoder = None  # Backup of original decoder state_dict (CPU) for memory efficiency
        self._active_loras = {}  # adapter_name -> scale (per-adapter)
        self._lora_adapter_registry = {}  # adapter_name -> explicit scaling targets
        self._lora_active_adapter = None

        # MLX DiT acceleration (macOS Apple Silicon only)
        self.mlx_decoder = None
        self.use_mlx_dit = False
        self.mlx_dit_compiled = False

        # MLX VAE acceleration (macOS Apple Silicon only)
        self.mlx_vae = None
        self.use_mlx_vae = False

    # ------------------------------------------------------------------
    # MLX DiT acceleration helpers
    # ------------------------------------------------------------------
    def _init_mlx_dit(self, compile_model: bool = False) -> bool:
        """Try to initialize the native MLX DiT decoder for Apple Silicon.

        Args:
            compile_model: If True, the diffusion step will be compiled with
                ``mx.compile`` for kernel fusion during generation.  The
                compilation itself happens lazily in ``mlx_generate_diffusion``.

        Returns True on success, False on failure (non-fatal).
        """
        try:
            from acestep.models.mlx import mlx_available
            if not mlx_available():
                logger.info("[MLX-DiT] MLX not available on this platform; skipping.")
                return False

            from acestep.models.mlx.dit_model import MLXDiTDecoder
            from acestep.models.mlx.dit_convert import convert_and_load

            mlx_decoder = MLXDiTDecoder.from_config(self.config)
            convert_and_load(self.model, mlx_decoder)
            self.mlx_decoder = mlx_decoder
            self.use_mlx_dit = True
            self.mlx_dit_compiled = compile_model
            logger.info(
                f"[MLX-DiT] Native MLX DiT decoder initialized successfully "
                f"(mx.compile={compile_model})."
            )
            return True
        except Exception as exc:
            logger.warning(f"[MLX-DiT] Failed to initialize MLX decoder (non-fatal): {exc}")
            self.mlx_decoder = None
            self.use_mlx_dit = False
            self.mlx_dit_compiled = False
            return False
    
    # ------------------------------------------------------------------
    # MLX VAE acceleration helpers
    # ------------------------------------------------------------------
    def _init_mlx_vae(self) -> bool:
        """Try to initialize the native MLX VAE for Apple Silicon.

        Converts the PyTorch ``AutoencoderOobleck`` weights into a pure-MLX
        re-implementation.  The PyTorch VAE is kept as a fallback.

        Performance optimizations applied:
        - Float16 inference: ~2x throughput from doubled memory bandwidth
          on Apple Silicon.  Snake1d uses mixed precision internally.
          Set ACESTEP_MLX_VAE_FP16=1 to enable float16 inference.
        - mx.compile(): kernel fusion reduces Metal dispatch overhead and
          improves data locality (used by mlx-lm, vllm-mlx, mlx-audio).

        Returns True on success, False on failure (non-fatal).
        """
        try:
            from acestep.models.mlx import mlx_available
            if not mlx_available():
                logger.info("[MLX-VAE] MLX not available on this platform; skipping.")
                return False

            import os
            import mlx.core as mx
            from mlx.utils import tree_map
            from acestep.models.mlx.vae_model import MLXAutoEncoderOobleck
            from acestep.models.mlx.vae_convert import convert_and_load

            mlx_vae = MLXAutoEncoderOobleck.from_pytorch_config(self.vae)
            convert_and_load(self.vae, mlx_vae)

            # --- Float16 conversion for faster inference ---
            # NOTE: Float16 causes audible quality degradation in the Oobleck
            # VAE decoder (the Snake activation and ConvTranspose1d chain
            # amplify rounding errors).  Default to float32 for quality.
            # Set ACESTEP_MLX_VAE_FP16=1 to enable float16 inference.
            use_fp16 = os.environ.get("ACESTEP_MLX_VAE_FP16", "0").lower() in (
                "1", "true", "yes",
            )
            vae_dtype = mx.float16 if use_fp16 else mx.float32

            if use_fp16:
                try:
                    def _to_fp16(x):
                        """Cast floating MLX arrays to float16 and keep other values unchanged."""
                        if isinstance(x, mx.array) and mx.issubdtype(x.dtype, mx.floating):
                            return x.astype(mx.float16)
                        return x
                    mlx_vae.update(tree_map(_to_fp16, mlx_vae.parameters()))
                    mx.eval(mlx_vae.parameters())
                    logger.info("[MLX-VAE] Model weights converted to float16.")
                except Exception as e:
                    logger.warning(f"[MLX-VAE] Float16 conversion failed ({e}); using float32.")
                    vae_dtype = mx.float32

            # --- Compile decode / encode for kernel fusion ---
            try:
                self._mlx_compiled_decode = mx.compile(mlx_vae.decode)
                self._mlx_compiled_encode_sample = mx.compile(mlx_vae.encode_and_sample)
                logger.info("[MLX-VAE] Decode/encode compiled with mx.compile().")
            except Exception as e:
                logger.warning(f"[MLX-VAE] mx.compile() failed ({e}); using uncompiled path.")
                self._mlx_compiled_decode = mlx_vae.decode
                self._mlx_compiled_encode_sample = mlx_vae.encode_and_sample

            self.mlx_vae = mlx_vae
            self.use_mlx_vae = True
            self._mlx_vae_dtype = vae_dtype
            logger.info(
                f"[MLX-VAE] Native MLX VAE initialized "
                f"(dtype={vae_dtype}, compiled=True)."
            )
            return True
        except Exception as exc:
            logger.warning(f"[MLX-VAE] Failed to initialize MLX VAE (non-fatal): {exc}")
            self.mlx_vae = None
            self.use_mlx_vae = False
            return False

    def _mlx_vae_decode(self, latents_torch):
        """Decode latents using native MLX VAE.
        
        Args:
            latents_torch: PyTorch tensor [B, C, T] (NCL format).
            
        Returns:
            PyTorch tensor [B, C_audio, T_audio] (NCL format).
        """
        import numpy as np
        import mlx.core as mx
        import time as _time

        t_start = _time.time()

        latents_np = latents_torch.detach().cpu().float().numpy()
        latents_nlc = np.transpose(latents_np, (0, 2, 1))  # NCL -> NLC

        B = latents_nlc.shape[0]
        T = latents_nlc.shape[1]

        # Convert to model dtype (float16 for speed, float32 fallback)
        vae_dtype = getattr(self, '_mlx_vae_dtype', mx.float32)
        latents_mx = mx.array(latents_nlc).astype(vae_dtype)

        t_convert = _time.time()

        # Use compiled decode (kernel-fused) when available
        decode_fn = getattr(self, '_mlx_compiled_decode', self.mlx_vae.decode)

        # Process batch items sequentially (peak memory stays constant)
        audio_parts = []
        for b in range(B):
            single = latents_mx[b : b + 1]  # [1, T, C]
            decoded = self._mlx_decode_single(single, decode_fn=decode_fn)
            # Cast back to float32 for downstream torch compatibility
            if decoded.dtype != mx.float32:
                decoded = decoded.astype(mx.float32)
            mx.eval(decoded)
            audio_parts.append(np.array(decoded))
            mx.clear_cache()  # Free intermediate buffers between samples

        t_decode = _time.time()

        audio_nlc = np.concatenate(audio_parts, axis=0)  # [B, T_audio, C_audio]
        audio_ncl = np.transpose(audio_nlc, (0, 2, 1))   # NLC -> NCL

        t_elapsed = _time.time() - t_start
        logger.info(
            f"[MLX-VAE] Decoded {B} sample(s), {T} latent frames -> "
            f"audio in {t_elapsed:.2f}s "
            f"(convert={t_convert - t_start:.3f}s, decode={t_decode - t_convert:.2f}s, "
            f"dtype={vae_dtype})"
        )

        return torch.from_numpy(audio_ncl)

    def _mlx_decode_single(self, z_nlc, decode_fn=None):
        """Decode a single sample with optional tiling for very long sequences.

        Args:
            z_nlc: MLX array [1, T, C] in NLC format.
            decode_fn: Compiled or plain decode callable.  Falls back to
                       ``self._mlx_compiled_decode`` or ``self.mlx_vae.decode``.
        
        Returns:
            MLX array [1, T_audio, C_audio] in NLC format.
        """
        import mlx.core as mx

        if decode_fn is None:
            decode_fn = getattr(self, '_mlx_compiled_decode', self.mlx_vae.decode)

        T = z_nlc.shape[1]
        # MLX unified memory: much larger chunk OK than PyTorch MPS.
        # 2048 latent frames ~= 87 seconds of audio; covers nearly all use cases.
        MLX_CHUNK = 2048
        MLX_OVERLAP = 64

        if T <= MLX_CHUNK:
            # No tiling needed; caller handles mx.eval()
            return decode_fn(z_nlc)

        # Overlap-discard tiling for very long sequences
        stride = MLX_CHUNK - 2 * MLX_OVERLAP
        num_steps = math.ceil(T / stride)
        decoded_parts = []
        upsample_factor = None

        for i in tqdm(range(num_steps), desc="Decoding audio chunks", disable=self.disable_tqdm):
            core_start = i * stride
            core_end = min(core_start + stride, T)
            win_start = max(0, core_start - MLX_OVERLAP)
            win_end = min(T, core_end + MLX_OVERLAP)

            chunk = z_nlc[:, win_start:win_end, :]
            audio_chunk = decode_fn(chunk)
            mx.eval(audio_chunk)

            if upsample_factor is None:
                upsample_factor = audio_chunk.shape[1] / chunk.shape[1]

            added_start = core_start - win_start
            trim_start = int(round(added_start * upsample_factor))
            added_end = win_end - core_end
            trim_end = int(round(added_end * upsample_factor))

            audio_len = audio_chunk.shape[1]
            end_idx = audio_len - trim_end if trim_end > 0 else audio_len
            decoded_parts.append(audio_chunk[:, trim_start:end_idx, :])

        return mx.concatenate(decoded_parts, axis=1)

    def _mlx_vae_encode_sample(self, audio_torch):
        """Encode audio and sample latent using native MLX VAE.
        
        Args:
            audio_torch: PyTorch tensor [B, C, S] (NCL format).
            
        Returns:
            PyTorch tensor [B, C_latent, T_latent] (NCL format).
        """
        import numpy as np
        import mlx.core as mx
        import time as _time

        audio_np = audio_torch.detach().cpu().float().numpy()
        audio_nlc = np.transpose(audio_np, (0, 2, 1))  # NCL -> NLC

        B = audio_nlc.shape[0]
        S = audio_nlc.shape[1]

        # Determine total work units for progress bar
        MLX_ENCODE_CHUNK = 48000 * 30
        MLX_ENCODE_OVERLAP = 48000 * 2
        if S <= MLX_ENCODE_CHUNK:
            chunks_per_sample = 1
        else:
            stride = MLX_ENCODE_CHUNK - 2 * MLX_ENCODE_OVERLAP
            chunks_per_sample = math.ceil(S / stride)
        total_work = B * chunks_per_sample

        t_start = _time.time()

        # Convert to model dtype (float16 for speed)
        vae_dtype = getattr(self, '_mlx_vae_dtype', mx.float32)
        # Use compiled encode when available
        encode_fn = getattr(self, '_mlx_compiled_encode_sample', self.mlx_vae.encode_and_sample)

        latent_parts = []
        pbar = tqdm(
            total=total_work,
            desc=f"MLX VAE Encode (native, n={B})",
            disable=self.disable_tqdm,
            unit="chunk",
        )
        for b in range(B):
            single = mx.array(audio_nlc[b : b + 1])  # [1, S, C_audio]
            if single.dtype != vae_dtype:
                single = single.astype(vae_dtype)
            latent = self._mlx_encode_single(single, pbar=pbar, encode_fn=encode_fn)
            # Cast back to float32 for downstream torch compatibility
            if latent.dtype != mx.float32:
                latent = latent.astype(mx.float32)
            mx.eval(latent)
            latent_parts.append(np.array(latent))
            mx.clear_cache()
        pbar.close()

        t_elapsed = _time.time() - t_start
        logger.info(
            f"[MLX-VAE] Encoded {B} sample(s), {S} audio frames -> "
            f"latent in {t_elapsed:.2f}s (dtype={vae_dtype})"
        )

        latent_nlc = np.concatenate(latent_parts, axis=0)  # [B, T, C_latent]
        latent_ncl = np.transpose(latent_nlc, (0, 2, 1))   # NLC -> NCL
        return torch.from_numpy(latent_ncl)

    def _mlx_encode_single(self, audio_nlc, pbar=None, encode_fn=None):
        """Encode a single audio sample with optional tiling.
        
        Args:
            audio_nlc: MLX array [1, S, C_audio] in NLC format.
            pbar: Optional tqdm progress bar to update.
            encode_fn: Compiled or plain encode callable.  Falls back to
                       ``self._mlx_compiled_encode_sample`` or
                       ``self.mlx_vae.encode_and_sample``.
            
        Returns:
            MLX array [1, T_latent, C_latent] in NLC format.
        """
        import mlx.core as mx

        if encode_fn is None:
            encode_fn = getattr(
                self, '_mlx_compiled_encode_sample', self.mlx_vae.encode_and_sample,
            )

        S = audio_nlc.shape[1]
        # ~30 sec at 48 kHz (generous for MLX unified memory)
        MLX_ENCODE_CHUNK = 48000 * 30
        MLX_ENCODE_OVERLAP = 48000 * 2

        if S <= MLX_ENCODE_CHUNK:
            result = encode_fn(audio_nlc)
            mx.eval(result)
            if pbar is not None:
                pbar.update(1)
            return result

        # Overlap-discard tiling
        stride = MLX_ENCODE_CHUNK - 2 * MLX_ENCODE_OVERLAP
        num_steps = math.ceil(S / stride)
        encoded_parts = []
        downsample_factor = None

        for i in range(num_steps):
            core_start = i * stride
            core_end = min(core_start + stride, S)
            win_start = max(0, core_start - MLX_ENCODE_OVERLAP)
            win_end = min(S, core_end + MLX_ENCODE_OVERLAP)

            chunk = audio_nlc[:, win_start:win_end, :]
            latent_chunk = encode_fn(chunk)
            mx.eval(latent_chunk)

            if downsample_factor is None:
                downsample_factor = chunk.shape[1] / latent_chunk.shape[1]

            added_start = core_start - win_start
            trim_start = int(round(added_start / downsample_factor))
            added_end = win_end - core_end
            trim_end = int(round(added_end / downsample_factor))

            latent_len = latent_chunk.shape[1]
            end_idx = latent_len - trim_end if trim_end > 0 else latent_len
            encoded_parts.append(latent_chunk[:, trim_start:end_idx, :])

            if pbar is not None:
                pbar.update(1)

        return mx.concatenate(encoded_parts, axis=1)
    
    def switch_to_training_preset(self) -> Tuple[str, bool]:
        """Best-effort switch to a training-safe preset (non-quantized DiT)."""
        if self.quantization is None:
            return "Already in training-safe preset (quantization disabled).", True

        if not self.last_init_params:
            return "Cannot switch preset automatically: no previous init parameters found.", False

        params = dict(self.last_init_params)
        params["quantization"] = None

        status, ok = self.initialize_service(
            project_root=params["project_root"],
            config_path=params["config_path"],
            device=params["device"],
            use_flash_attention=params["use_flash_attention"],
            compile_model=params["compile_model"],
            offload_to_cpu=params["offload_to_cpu"],
            offload_dit_to_cpu=params["offload_dit_to_cpu"],
            quantization=None,
            prefer_source=params.get("prefer_source"),
        )
        if ok:
            return f"Switched to training preset (quantization disabled).\n{status}", True
        return f"Failed to switch to training preset.\n{status}", False

    @torch.inference_mode()
    def service_generate(
        self,
        captions: Union[str, List[str]],
        lyrics: Union[str, List[str]],
        keys: Optional[Union[str, List[str]]] = None,
        target_wavs: Optional[torch.Tensor] = None,
        refer_audios: Optional[List[List[torch.Tensor]]] = None,
        metas: Optional[Union[str, Dict[str, Any], List[Union[str, Dict[str, Any]]]]] = None,
        vocal_languages: Optional[Union[str, List[str]]] = None,
        infer_steps: int = 60,
        guidance_scale: float = 7.0,
        seed: Optional[Union[int, List[int]]] = None,
        return_intermediate: bool = False,
        repainting_start: Optional[Union[float, List[float]]] = None,
        repainting_end: Optional[Union[float, List[float]]] = None,
        instructions: Optional[Union[str, List[str]]] = None,
        audio_cover_strength: float = 1.0,
        cover_noise_strength: float = 0.0,
        use_adg: bool = False,
        cfg_interval_start: float = 0.0,
        cfg_interval_end: float = 1.0,
        shift: float = 1.0,
        audio_code_hints: Optional[Union[str, List[str]]] = None,
        infer_method: str = "ode",
        timesteps: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Generate music latents from text/audio conditioning inputs.

        Args:
            captions: Caption text(s) describing target music.
            lyrics: Lyric text(s) used for lyric conditioning.
            keys: Optional sample identifiers.
            target_wavs: Optional target audio tensor for repaint/cover.
            refer_audios: Optional reference audio tensors for style conditioning.
            metas: Optional metadata strings/dicts per sample.
            vocal_languages: Optional lyric language code(s).
            infer_steps: Diffusion inference steps.
            guidance_scale: Classifier-free guidance scale.
            seed: Optional single seed or per-sample seed list.
            return_intermediate: Reserved compatibility flag (handled by caller flow).
            repainting_start: Optional repaint start time(s) in seconds.
            repainting_end: Optional repaint end time(s) in seconds.
            instructions: Optional instruction text(s) per sample.
            audio_cover_strength: Blend strength for cover mode.
            cover_noise_strength: Initial-noise blend strength for cover mode.
            use_adg: Whether to enable adaptive diffusion guidance.
            cfg_interval_start: CFG schedule start ratio.
            cfg_interval_end: CFG schedule end ratio.
            shift: Diffusion time-shift parameter.
            audio_code_hints: Optional serialized audio-code hints.
            infer_method: Diffusion method selector.
            timesteps: Optional custom timestep schedule.

        Returns:
            Dict[str, Any]: Model output payload with latents, masks, spans, timing, and cached
            condition tensors required by downstream result handlers.
        """
        _ = return_intermediate
        normalized = self._normalize_service_generate_inputs(
            captions=captions,
            lyrics=lyrics,
            keys=keys,
            metas=metas,
            vocal_languages=vocal_languages,
            repainting_start=repainting_start,
            repainting_end=repainting_end,
            instructions=instructions,
            audio_code_hints=audio_code_hints,
            infer_steps=infer_steps,
            seed=seed,
        )
        batch = self._prepare_batch(
            captions=normalized["captions"],
            lyrics=normalized["lyrics"],
            keys=normalized["keys"],
            target_wavs=target_wavs,
            refer_audios=refer_audios,
            metas=normalized["metas"],
            vocal_languages=normalized["vocal_languages"],
            repainting_start=normalized["repainting_start"],
            repainting_end=normalized["repainting_end"],
            instructions=normalized["instructions"],
            audio_code_hints=normalized["audio_code_hints"],
            audio_cover_strength=audio_cover_strength,
            cover_noise_strength=cover_noise_strength,
        )
        payload = self._unpack_service_processed_data(self.preprocess_batch(batch))
        seed_param = self._resolve_service_seed_param(normalized["seed_list"])
        self._ensure_silence_latent_on_device()
        generate_kwargs = self._build_service_generate_kwargs(
            payload=payload,
            seed_param=seed_param,
            infer_steps=normalized["infer_steps"],
            guidance_scale=guidance_scale,
            audio_cover_strength=audio_cover_strength,
            cover_noise_strength=cover_noise_strength,
            infer_method=infer_method,
            use_adg=use_adg,
            cfg_interval_start=cfg_interval_start,
            cfg_interval_end=cfg_interval_end,
            shift=shift,
            timesteps=timesteps,
        )
        outputs, encoder_hidden_states, encoder_attention_mask, context_latents = (
            self._execute_service_generate_diffusion(
                payload=payload,
                generate_kwargs=generate_kwargs,
                seed_param=seed_param,
                infer_method=infer_method,
                shift=shift,
                audio_cover_strength=audio_cover_strength,
            )
        )
        return self._attach_service_generate_outputs(
            outputs=outputs,
            payload=payload,
            batch=batch,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            context_latents=context_latents,
        )

    def generate_music(
        self,
        captions: str,
        lyrics: str,
        bpm: Optional[int] = None,
        key_scale: str = "",
        time_signature: str = "",
        vocal_language: str = "en",
        inference_steps: int = 8,
        guidance_scale: float = 7.0,
        use_random_seed: bool = True,
        seed: Optional[Union[str, float, int]] = -1,
        reference_audio=None,
        audio_duration: Optional[float] = None,
        batch_size: Optional[int] = None,
        src_audio=None,
        audio_code_string: Union[str, List[str]] = "",
        repainting_start: float = 0.0,
        repainting_end: Optional[float] = None,
        instruction: str = DEFAULT_DIT_INSTRUCTION,
        audio_cover_strength: float = 1.0,
        cover_noise_strength: float = 0.0,
        task_type: str = "text2music",
        use_adg: bool = False,
        cfg_interval_start: float = 0.0,
        cfg_interval_end: float = 1.0,
        shift: float = 1.0,
        infer_method: str = "ode",
        use_tiled_decode: bool = True,
        timesteps: Optional[List[float]] = None,
        latent_shift: float = 0.0,
        latent_rescale: float = 1.0,
        progress=None
    ) -> Dict[str, Any]:
        """
        Main interface for music generation
        
        Returns:
            Dictionary containing:
            - audios: List of audio dictionaries with path, key, params
            - generation_info: Markdown-formatted generation information
            - status_message: Status message
            - extra_outputs: Dictionary with latents, masks, time_costs, etc.
            - success: Whether generation completed successfully
            - error: Error message if generation failed
        """
        progress = self._resolve_generate_music_progress(progress)

        if self.model is None or self.vae is None or self.text_tokenizer is None or self.text_encoder is None:
            readiness_error = self._validate_generate_music_readiness()
            return readiness_error

        task_type, instruction = self._resolve_generate_music_task(
            task_type=task_type,
            audio_code_string=audio_code_string,
            instruction=instruction,
        )

        logger.info("[generate_music] Starting generation...")
        if progress:
            progress(0.51, desc="Preparing inputs...")
        logger.info("[generate_music] Preparing inputs...")
        
        runtime = self._prepare_generate_music_runtime(
            batch_size=batch_size,
            audio_duration=audio_duration,
            repainting_end=repainting_end,
            seed=seed,
            use_random_seed=use_random_seed,
        )
        actual_batch_size = runtime["actual_batch_size"]
        actual_seed_list = runtime["actual_seed_list"]
        seed_value_for_ui = runtime["seed_value_for_ui"]
        audio_duration = runtime["audio_duration"]
        repainting_end = runtime["repainting_end"]
            
        try:
            refer_audios, processed_src_audio, audio_error = self._prepare_reference_and_source_audio(
                reference_audio=reference_audio,
                src_audio=src_audio,
                audio_code_string=audio_code_string,
                actual_batch_size=actual_batch_size,
                task_type=task_type,
            )
            if audio_error is not None:
                return audio_error

            service_inputs = self._prepare_generate_music_service_inputs(
                actual_batch_size=actual_batch_size,
                processed_src_audio=processed_src_audio,
                audio_duration=audio_duration,
                captions=captions,
                lyrics=lyrics,
                vocal_language=vocal_language,
                instruction=instruction,
                bpm=bpm,
                key_scale=key_scale,
                time_signature=time_signature,
                task_type=task_type,
                audio_code_string=audio_code_string,
                repainting_start=repainting_start,
                repainting_end=repainting_end,
            )
            service_run = self._run_generate_music_service_with_progress(
                progress=progress,
                actual_batch_size=actual_batch_size,
                audio_duration=audio_duration,
                inference_steps=inference_steps,
                timesteps=timesteps,
                service_inputs=service_inputs,
                refer_audios=refer_audios,
                guidance_scale=guidance_scale,
                actual_seed_list=actual_seed_list,
                audio_cover_strength=audio_cover_strength,
                cover_noise_strength=cover_noise_strength,
                use_adg=use_adg,
                cfg_interval_start=cfg_interval_start,
                cfg_interval_end=cfg_interval_end,
                shift=shift,
                infer_method=infer_method,
            )
            outputs = service_run["outputs"]
            infer_steps_for_progress = service_run["infer_steps_for_progress"]
            
            logger.info("[generate_music] Model generation completed. Decoding latents...")
            pred_latents = outputs["target_latents"]  # [batch, latent_length, latent_dim]
            time_costs = outputs["time_costs"]
            time_costs["offload_time_cost"] = self.current_offload_cost
            per_step = time_costs.get("diffusion_per_step_time_cost")
            if isinstance(per_step, (int, float)) and per_step > 0:
                self._last_diffusion_per_step_sec = float(per_step)
                self._update_progress_estimate(
                    per_step_sec=float(per_step),
                    infer_steps=infer_steps_for_progress,
                    batch_size=actual_batch_size,
                    duration_sec=audio_duration if audio_duration and audio_duration > 0 else None,
                )
            if self.debug_stats:
                logger.debug(
                    f"[generate_music] pred_latents: {pred_latents.shape}, dtype={pred_latents.dtype} "
                    f"{pred_latents.min()=}, {pred_latents.max()=}, {pred_latents.mean()=} {pred_latents.std()=}"
                )
            else:
                logger.debug(f"[generate_music] pred_latents: {pred_latents.shape}, dtype={pred_latents.dtype}")
            logger.debug(f"[generate_music] time_costs: {time_costs}")

            if torch.isnan(pred_latents).any() or torch.isinf(pred_latents).any():
                raise RuntimeError(
                    "Generation produced NaN or Inf latents. "
                    "This usually indicates a checkpoint/config mismatch "
                    "or unsupported quantization/backend combination. "
                    "Try running with --backend pt or verify your model checkpoints match this release."
                )
            if pred_latents.numel() > 0 and pred_latents.abs().sum() == 0:
                raise RuntimeError(
                    "Generation produced zero latents. "
                    "This usually indicates a checkpoint/config mismatch or unsupported setup."
                )

            if progress:
                progress(0.8, desc="Decoding audio...")
            logger.info("[generate_music] Decoding latents with VAE...")
            
            # Apply latent shift and rescale before VAE decode (for anti-clipping control)
            if latent_shift != 0.0 or latent_rescale != 1.0:
                logger.info(f"[generate_music] Applying latent post-processing: shift={latent_shift}, rescale={latent_rescale}")
                if self.debug_stats:
                    logger.debug(f"[generate_music] Latent BEFORE shift/rescale: min={pred_latents.min():.4f}, max={pred_latents.max():.4f}, mean={pred_latents.mean():.4f}, std={pred_latents.std():.4f}")
                pred_latents = pred_latents * latent_rescale + latent_shift
                if self.debug_stats:
                    logger.debug(f"[generate_music] Latent AFTER shift/rescale: min={pred_latents.min():.4f}, max={pred_latents.max():.4f}, mean={pred_latents.mean():.4f}, std={pred_latents.std():.4f}")
            
            # Decode latents to audio
            start_time = time.time()
            with torch.inference_mode():
                with self._load_model_context("vae"):
                    # Move pred_latents to CPU early to save VRAM (will be used in extra_outputs later)
                    pred_latents_cpu = pred_latents.detach().cpu()
                    
                    # Transpose for VAE decode: [batch, latent_length, latent_dim] -> [batch, latent_dim, latent_length]
                    pred_latents_for_decode = pred_latents.transpose(1, 2).contiguous()
                    # Ensure input is in VAE's dtype
                    pred_latents_for_decode = pred_latents_for_decode.to(self.vae.dtype)
                    
                    # Release original pred_latents to free VRAM before VAE decode
                    del pred_latents
                    self._empty_cache()
                    
                    logger.debug(f"[generate_music] Before VAE decode: allocated={self._memory_allocated()/1024**3:.2f}GB, max={self._max_memory_allocated()/1024**3:.2f}GB")
                    
                    # When native MLX VAE is active, bypass VRAM checks and CPU
                    # offload entirely; MLX uses unified memory, not PyTorch VRAM.
                    _using_mlx_vae = self.use_mlx_vae and self.mlx_vae is not None
                    _vae_cpu = False

                    if not _using_mlx_vae:
                        # Check effective free VRAM and auto-enable CPU decode if extremely tight
                        import os as _os
                        _vae_cpu = _os.environ.get("ACESTEP_VAE_ON_CPU", "0").lower() in ("1", "true", "yes")
                        if not _vae_cpu:
                            # MPS (Apple Silicon) uses unified memory; get_effective_free_vram_gb()
                            # relies on CUDA and always returns 0 on Mac, which would incorrectly
                            # force VAE decode onto the CPU.  Skip the auto-CPU logic for MPS.
                            if self.device == "mps":
                                logger.info("[generate_music] MPS device: skipping VRAM check (unified memory), keeping VAE on MPS")
                            else:
                                _effective_free = get_effective_free_vram_gb()
                                logger.info(f"[generate_music] Effective free VRAM before VAE decode: {_effective_free:.2f} GB")
                                # If less than 0.5 GB free, VAE decode on GPU will almost certainly OOM
                                if _effective_free < 0.5:
                                    logger.warning(f"[generate_music] Only {_effective_free:.2f} GB free VRAM; auto-enabling CPU VAE decode")
                                    _vae_cpu = True
                        if _vae_cpu:
                            logger.info("[generate_music] Moving VAE to CPU for decode (ACESTEP_VAE_ON_CPU=1)...")
                            _vae_device = next(self.vae.parameters()).device
                            self.vae = self.vae.cpu()
                            pred_latents_for_decode = pred_latents_for_decode.cpu()
                            self._empty_cache()
                    
                    if use_tiled_decode:
                        logger.info("[generate_music] Using tiled VAE decode to reduce VRAM usage...")
                        pred_wavs = self.tiled_decode(pred_latents_for_decode)  # [batch, channels, samples]
                    elif _using_mlx_vae:
                        # Direct decode via native MLX (no tiling needed)
                        try:
                            pred_wavs = self._mlx_vae_decode(pred_latents_for_decode)
                        except Exception as exc:
                            logger.warning(f"[generate_music] MLX direct decode failed ({exc}), falling back to PyTorch")
                            decoder_output = self.vae.decode(pred_latents_for_decode)
                            pred_wavs = decoder_output.sample
                            del decoder_output
                    else:
                        decoder_output = self.vae.decode(pred_latents_for_decode)
                        pred_wavs = decoder_output.sample
                        del decoder_output
                    
                    if _vae_cpu:
                        logger.info("[generate_music] VAE decode on CPU complete, restoring to GPU...")
                        self.vae = self.vae.to(_vae_device)
                        if pred_wavs.device.type != 'cpu':
                            pass  # already on right device
                        # pred_wavs stays on CPU - fine for audio post-processing
                    
                    logger.debug(f"[generate_music] After VAE decode: allocated={self._memory_allocated()/1024**3:.2f}GB, max={self._max_memory_allocated()/1024**3:.2f}GB")
                    
                    # Release pred_latents_for_decode after decode
                    del pred_latents_for_decode
                    
                    # Cast output to float32 for audio processing/saving (in-place if possible)
                    if pred_wavs.dtype != torch.float32:
                        pred_wavs = pred_wavs.float()
                    
                    # Anti-clipping normalization: only scale if peak exceeds [-1, 1].
                    peak = pred_wavs.abs().amax(dim=[1, 2], keepdim=True)
                    if torch.any(peak > 1.0):
                        pred_wavs = pred_wavs / peak.clamp(min=1.0)
                    self._empty_cache()
            end_time = time.time()
            time_costs["vae_decode_time_cost"] = end_time - start_time
            time_costs["total_time_cost"] = time_costs["total_time_cost"] + time_costs["vae_decode_time_cost"]
            
            # Update offload cost one last time to include VAE offloading
            time_costs["offload_time_cost"] = self.current_offload_cost
            
            logger.info("[generate_music] VAE decode completed. Preparing audio tensors...")
            if progress:
                progress(0.99, desc="Preparing audio data...")
            
            # Prepare audio tensors (no file I/O here, no UUID generation)
            # pred_wavs is already [batch, channels, samples] format
            # Move to CPU and convert to float32 for return
            audio_tensors = []
            
            for i in range(actual_batch_size):
                # Extract audio tensor: [channels, samples] format, CPU, float32
                audio_tensor = pred_wavs[i].cpu()
                audio_tensors.append(audio_tensor)
            
            status_message = "Generation completed successfully!"
            logger.info(f"[generate_music] Done! Generated {len(audio_tensors)} audio tensors.")
            
            # Extract intermediate information from outputs
            src_latents = outputs.get("src_latents")  # [batch, T, D]
            target_latents_input = outputs.get("target_latents_input")  # [batch, T, D]
            chunk_masks = outputs.get("chunk_masks")  # [batch, T]
            spans = outputs.get("spans", [])  # List of tuples
            latent_masks = outputs.get("latent_masks")  # [batch, T]
            
            # Extract condition tensors for LRC timestamp generation
            encoder_hidden_states = outputs.get("encoder_hidden_states")
            encoder_attention_mask = outputs.get("encoder_attention_mask")
            context_latents = outputs.get("context_latents")
            lyric_token_idss = outputs.get("lyric_token_idss")
            
            # Move all tensors to CPU to save VRAM (detach to release computation graph)
            extra_outputs = {
                "pred_latents": pred_latents_cpu,  # Already moved to CPU earlier to save VRAM during VAE decode
                "target_latents": target_latents_input.detach().cpu() if target_latents_input is not None else None,
                "src_latents": src_latents.detach().cpu() if src_latents is not None else None,
                "chunk_masks": chunk_masks.detach().cpu() if chunk_masks is not None else None,
                "latent_masks": latent_masks.detach().cpu() if latent_masks is not None else None,
                "spans": spans,
                "time_costs": time_costs,
                "seed_value": seed_value_for_ui,
                # Condition tensors for LRC timestamp generation
                "encoder_hidden_states": encoder_hidden_states.detach().cpu() if encoder_hidden_states is not None else None,
                "encoder_attention_mask": encoder_attention_mask.detach().cpu() if encoder_attention_mask is not None else None,
                "context_latents": context_latents.detach().cpu() if context_latents is not None else None,
                "lyric_token_idss": lyric_token_idss.detach().cpu() if lyric_token_idss is not None else None,
            }
            
            # Build audios list with tensor data (no file paths, no UUIDs, handled outside)
            audios = []
            for idx, audio_tensor in enumerate(audio_tensors):
                audio_dict = {
                    "tensor": audio_tensor,  # torch.Tensor [channels, samples], CPU, float32
                    "sample_rate": self.sample_rate,
                }
                audios.append(audio_dict)
            
            return {
                "audios": audios,
                "status_message": status_message,
                "extra_outputs": extra_outputs,
                "success": True,
                "error": None,
            }

        except Exception as e:
            error_msg = f"âŒ Error: {str(e)}\n{traceback.format_exc()}"
            logger.exception("[generate_music] Generation failed")
            return {
                "audios": [],
                "status_message": error_msg,
                "extra_outputs": {},
                "success": False,
                "error": str(e),
            }
