"""Top-level initialization orchestration for the handler."""

import os
import traceback
from pathlib import Path
from typing import Optional, Tuple

import torch
from loguru import logger

from acestep import gpu_config

_ROCM_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

_DIT_DTYPE_MAP = {
    "auto": None,  # None is automode
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _detect_cuda_dtype() -> torch.dtype:
    """Auto-detect optimal dtype based on GPU architecture.
    
    Returns:
        torch.dtype: float32 for safety fallback, float16 for pre-Ampere,
                     bfloat16 for Ampere and newer
    """
    try:
        if not torch.cuda.is_available():
            logger.info("[detect_cuda_dtype] CUDA not available -> using float32")
            return torch.float32
        
        # Get compute capability as a tuple (major, minor)
        # Pascal = 6.x (GTX 10xx: 1060, 1070, 1080, etc.)
        # Turing = 7.5 (GTX 16xx: 1650, 1660, 1660 Ti, etc.)
        # Ampere = 8.x (RTX 30xx, A100, etc.)
        # Ada = 8.9 (RTX 40xx, etc.)
        # Hopper = 9.0 (H100, etc.)
        major, minor = torch.cuda.get_device_capability(0)
        compute_capability = major * 10 + minor
        
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"[detect_cuda_dtype] Detected GPU: {gpu_name} (CC {major}.{minor})")
        
        # Pascal (60-69) and older - float16 for compatibility
        if compute_capability < 75:
            logger.info(f"[detect_cuda_dtype] Pascal or older (CC < 7.5) -> using float16")
            return torch.float16
        
        # Turing (75) - check if GTX 16xx or RTX 20xx
        elif compute_capability == 75:
            gpu_name_lower = gpu_name.lower()
            # GTX 16xx - use bfloat16
            if 'gtx 16' in gpu_name_lower:
                logger.info(f"[detect_cuda_dtype] GTX 16xx series detected -> using bfloat16")
                return torch.bfloat16
            # RTX 20xx - has bfloat16 but conservative float16 for safety
            # Change to bfloat16 if you want to test RTX 20xx capabilities
            logger.info(f"[detect_cuda_dtype] RTX 20xx series detected -> using float16 (conservative)")
            return torch.bfloat16
        
        # Ampere (80+) and newer - native bfloat16 support
        elif compute_capability >= 80:
            logger.info(f"[detect_cuda_dtype] Ampere or newer (CC >= 8.0) -> using bfloat16")
            return torch.bfloat16
        
        # Fallback for unknown cases
        else:
            logger.warning(f"[detect_cuda_dtype] Unknown compute capability {compute_capability} -> using float16 (safe fallback)")
            return torch.float16
        
    except Exception as e:
        logger.warning(f"[detect_cuda_dtype] Failed to detect GPU architecture: {e} -> using float16")
        return torch.float16


def _resolve_rocm_dtype() -> torch.dtype:
    """Return a safe model dtype for ROCm/HIP devices.

    Uses ``float32`` by default to avoid segfaults from incomplete
    ``bfloat16`` kernel support on some ROCm GPU configurations (e.g.
    AMD iGPUs on Strix Halo).  Set the ``ACESTEP_ROCM_DTYPE`` environment
    variable to ``float16`` or ``bfloat16`` to override for hardware that
    fully supports those formats.
    """
    raw = os.environ.get("ACESTEP_ROCM_DTYPE", "float32").strip().lower()
    dtype = _ROCM_DTYPE_MAP.get(raw)
    if dtype is None:
        logger.warning(
            f"[initialize_service] Unknown ACESTEP_ROCM_DTYPE={raw!r}; "
            "falling back to float32."
        )
        dtype = torch.float32
    return dtype


def _resolve_dtype(
    device: str,
    dit_dtype_mode: str = "auto",
) -> torch.dtype:
    """Resolve final dtype based on device and user-selected mode.
    
    Args:
        device: Target device (cuda, xpu, cpu, etc.)
        dit_dtype_mode: Selected mode from dropdown ("auto", "float32", "float16", "bfloat16")
    
    Returns:
        torch.dtype: Final dtype to use for DiT model
    """
    # Forcing mode
    forced_dtype = _DIT_DTYPE_MAP.get(dit_dtype_mode)
    if forced_dtype is not None:
        logger.info(
            f"[_resolve_dtype] User forced dtype mode '{dit_dtype_mode}' -> using {forced_dtype}"
        )
        return forced_dtype
    
    # Auto mode
    if device == "cuda" and gpu_config.is_rocm_available():
        dtype = _resolve_rocm_dtype()
        logger.info(
            f"[_resolve_dtype] ROCm/HIP device detected (auto mode): using dtype={dtype}"
        )
        return dtype
    elif device == "cuda":
        dtype = _detect_cuda_dtype()
        logger.info(f"[_resolve_dtype] CUDA device (auto mode): using dtype={dtype}")
        return dtype
    else:
        # XPU and rest devices
        dtype = torch.bfloat16 if device == "xpu" else torch.float32
        logger.info(f"[_resolve_dtype] Device {device} (auto mode): using dtype={dtype}")
        return dtype


class InitServiceOrchestratorMixin:
    """Public ``initialize_service`` orchestration entrypoint."""

    def initialize_service(
        self,
        project_root: str,
        config_path: str,
        device: str = "auto",
        use_flash_attention: bool = False,
        compile_model: bool = False,
        offload_to_cpu: bool = False,
        offload_dit_to_cpu: bool = False,
        quantization: Optional[str] = None,
        prefer_source: Optional[str] = None,
        use_mlx_dit: bool = True,
        dit_dtype_mode: str = "auto",
    ) -> Tuple[str, bool]:
        """Initialize model artifacts and runtime backends for generation.

        This method intentionally supports repeated calls to reinitialize models
        with new settings; it does not short-circuit when components are already loaded.
        """
        try:
            if config_path is None:
                config_path = "acestep-v15-turbo"
                logger.warning(
                    "[initialize_service] config_path not set; defaulting to 'acestep-v15-turbo'."
                )

            resolved_device = self._resolve_initialize_device(device)
            self.device = resolved_device
            self.offload_to_cpu = offload_to_cpu
            self.offload_dit_to_cpu = offload_dit_to_cpu

            normalized_compile, normalized_quantization, mlx_compile_requested = self._configure_initialize_runtime(
                device=resolved_device,
                compile_model=compile_model,
                quantization=quantization,
            )
            self.compiled = normalized_compile

            self.dtype = _resolve_dtype(resolved_device, dit_dtype_mode)
            self.dit_dtype_mode = dit_dtype_mode  # saving for info
            
            self.quantization = normalized_quantization
            try:
                self._validate_quantization_setup(
                    quantization=self.quantization,
                    compile_model=normalized_compile,
                )
            except ImportError as exc:
                if self.quantization is not None:
                    logger.warning(
                        "[initialize_service] Quantization disabled: {}",
                        exc,
                    )
                    self.quantization = None
                else:
                    raise

            base_root = project_root or self._get_project_root()
            checkpoint_dir = os.path.join(base_root, "checkpoints")
            checkpoint_path = Path(checkpoint_dir)

            precheck_failure = self._ensure_models_present(
                checkpoint_path=checkpoint_path,
                config_path=config_path,
                prefer_source=prefer_source,
            )
            if precheck_failure is not None:
                self.model = None
                self.vae = None
                self.text_encoder = None
                self.text_tokenizer = None
                self.config = None
                self.silence_latent = None
                return precheck_failure

            self._sync_model_code_if_needed(config_path, checkpoint_path)

            model_path = os.path.join(checkpoint_dir, config_path)
            self._load_main_model_from_checkpoint(
                model_checkpoint_path=model_path,
                device=resolved_device,
                use_flash_attention=use_flash_attention,
                compile_model=normalized_compile,
                quantization=self.quantization,
            )
            vae_path = self._load_vae_model(
                checkpoint_dir=checkpoint_dir,
                device=resolved_device,
                compile_model=normalized_compile,
            )
            text_encoder_path = self._load_text_encoder_and_tokenizer(
                checkpoint_dir=checkpoint_dir,
                device=resolved_device,
            )

            mlx_dit_status, mlx_vae_status = self._initialize_mlx_backends(
                device=resolved_device,
                use_mlx_dit=use_mlx_dit,
                mlx_compile_requested=mlx_compile_requested,
            )

            status_msg = self._build_initialize_status_message(
                device=resolved_device,
                model_path=model_path,
                vae_path=vae_path,
                text_encoder_path=text_encoder_path,
                dtype=self.dtype,
                attention=getattr(self.config, "_attn_implementation", "eager"),
                compile_model=normalized_compile,
                mlx_compile_requested=mlx_compile_requested,
                offload_to_cpu=offload_to_cpu,
                offload_dit_to_cpu=offload_dit_to_cpu,
                quantization=self.quantization,
                mlx_dit_status=mlx_dit_status,
                mlx_vae_status=mlx_vae_status,
                dit_dtype_mode=dit_dtype_mode,
            )

            self.last_init_params = {
                "project_root": project_root,
                "config_path": config_path,
                "device": resolved_device,
                "use_flash_attention": use_flash_attention,
                "compile_model": normalized_compile,
                "offload_to_cpu": offload_to_cpu,
                "offload_dit_to_cpu": offload_dit_to_cpu,
                "quantization": self.quantization,
                "use_mlx_dit": use_mlx_dit,
                "prefer_source": prefer_source,
                "dit_dtype_mode": dit_dtype_mode,
            }

            return status_msg, True
        except Exception as exc:
            self.model = None
            self.vae = None
            self.text_encoder = None
            self.text_tokenizer = None
            self.config = None
            self.silence_latent = None
            error_msg = f"Error initializing model: {str(exc)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.exception(error_msg)
            return error_msg, False

