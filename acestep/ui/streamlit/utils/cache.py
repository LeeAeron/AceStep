"""
Handler caching for Streamlit.

Creates / caches AceStepHandler and LLMHandler instances and exposes
a single ``initialize_models()`` that loads weights on demand.
"""
import os
import sys
import torch
import gc
from typing import Optional, Tuple
from pathlib import Path

import streamlit as st
from loguru import logger

# Ensure ACE-Step repo is on Python path
_project_root = Path(__file__).parent.parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def get_available_vram_mb() -> int:
    """Get available VRAM in MB. Returns 0 if no GPU or CUDA not available."""
    try:
        if not torch.cuda.is_available():
            return 0

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        gpu_id = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
        reserved = torch.cuda.memory_reserved(gpu_id)
        allocated = torch.cuda.memory_allocated(gpu_id)
        free_memory = total_memory - reserved
        free_mb = free_memory // (1024 * 1024)
        return free_mb
    except Exception as exc:
        logger.warning(f"Could not detect VRAM: {exc}")
        return 0


def get_total_vram_mb() -> int:
    """Get total VRAM in MB. Returns 0 if no GPU."""
    try:
        if not torch.cuda.is_available():
            return 0
        gpu_id = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(gpu_id).total_memory
        return total // (1024 * 1024)
    except Exception:
        return 0


def force_clear_memory():
    """Aggressive flush GPU."""
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()


@st.cache_resource
def get_dit_handler():
    """Return a cached AceStepHandler instance (uninitialised)."""
    try:
        from acestep.handler import AceStepHandler
        logger.info("Creating AceStepHandler instance...")
        return AceStepHandler()
    except Exception as exc:
        logger.error(f"Failed to create AceStepHandler: {exc}")
        return None


@st.cache_resource
def get_llm_handler():
    """Return a cached LLMHandler instance (uninitialised)."""
    try:
        from acestep.llm_inference import LLMHandler
        logger.info("Creating LLMHandler instance...")
        return LLMHandler()
    except Exception as exc:
        logger.error(f"Failed to create LLMHandler: {exc}")
        return None


@st.cache_resource
def get_dataset_handler():
    """Return a cached DatasetHandler instance."""
    try:
        from acestep.dataset_handler import DatasetHandler
        logger.info("Creating DatasetHandler instance...")
        return DatasetHandler()
    except Exception as exc:
        logger.error(f"Failed to create DatasetHandler: {exc}")
        return None


def is_dit_ready() -> bool:
    """Check whether DiT model weights are loaded."""
    handler = get_dit_handler()
    return handler is not None and handler.model is not None


def is_llm_ready() -> bool:
    """Check whether LLM model weights are loaded."""
    handler = get_llm_handler()
    return handler is not None and handler.llm_initialized


def initialize_dit(
    config_path: str = "acestep-v15-sft",
    device: str = "cuda",
    offload_to_cpu: bool = True,
    compile_model: bool = False,
    min_vram_mb: int = 2048,
) -> Tuple[str, bool]:
    """Load DiT model weights with smart offload."""
    handler = get_dit_handler()
    if handler is None:
        return "AceStepHandler could not be created", False

    force_clear_memory()

    project_root = str(_project_root)
    
    total_vram = get_total_vram_mb()
    use_flash = (total_vram >= 8192) and not offload_to_cpu
    if use_flash:
        use_flash = handler.is_flash_attention_available(device)

    logger.info(f"Initializing DiT: device={device}, offload={offload_to_cpu}, compile={compile_model}")

    status, ok = handler.initialize_service(
        project_root=project_root,
        config_path=config_path,
        device=device,
        use_flash_attention=use_flash,
        compile_model=compile_model,
        offload_to_cpu=offload_to_cpu,
    )
    
    # Patching handler for force GPU decode
    if ok and device == "cuda":
        _patch_handler_for_gpu_decode(handler)
    
    if ok:
        mode_str = "CUDA" if device == "cuda" else "CPU"
        offload_str = "+offload" if offload_to_cpu else ""
        status = f"{status} ({mode_str}{offload_str})"
    
    return status, ok


def _patch_handler_for_gpu_decode(handler):
    """
    Patch handler to prevent fallback onto CPU decode.
    """
    import acestep.core.generation.handler.generate_music_decode as decode_module
    
    # saving original funcion
    if hasattr(decode_module, '_check_vram_and_offload'):
        original_check = decode_module._check_vram_and_offload
        
        def forced_gpu_check(*args, **kwargs):
            """Always give False (not offload to CPU) and fake chunk_size"""
            return False, 256  # False = not on CPU, 256 = chunk_size
        
        decode_module._check_vram_and_offload = forced_gpu_check
        logger.info("Patched _check_vram_and_offload for forced GPU decode")
    
    # Patch handler method
    if hasattr(handler, '_check_vram_for_decode'):
        handler._check_vram_for_decode = lambda *args, **kwargs: (True, 1.0)
    
    logger.info("Handler patched for GPU decode")


def initialize_llm(
    lm_model_path: str = "acestep-5Hz-lm-1.7B",
    backend: str = "pt",
    device: str = "cuda",
    offload_to_cpu: bool = True,
    min_vram_mb: int = 1536,
) -> Tuple[str, bool]:
    """Load LLM model weights."""
    handler = get_llm_handler()
    if handler is None:
        return "LLMHandler could not be created", False

    checkpoint_dir = str(_project_root / "checkpoints")

    try:
        from acestep.model_downloader import ensure_lm_model
        dl_ok, dl_msg = ensure_lm_model(
            model_name=lm_model_path,
            checkpoints_dir=checkpoint_dir,
        )
        if not dl_ok:
            logger.warning(f"LM model download issue: {dl_msg}")
    except Exception as exc:
        logger.warning(f"LM model download check failed: {exc}")

    status, ok = handler.initialize(
        checkpoint_dir=checkpoint_dir,
        lm_model_path=lm_model_path,
        backend=backend,
        device=device,
        offload_to_cpu=offload_to_cpu,
    )
    
    if ok:
        mode_str = "CUDA" if device == "cuda" else "CPU"
        offload_str = "+offload" if offload_to_cpu else ""
        status = f"{status} ({mode_str}{offload_str})"
    
    return status, ok


def get_memory_info() -> dict:
    """Get current memory status for display in UI."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda",
        "total_vram_mb": 0,
        "available_vram_mb": 0,
        "offload_recommended": True,
    }
    
    if torch.cuda.is_available():
        info["total_vram_mb"] = get_total_vram_mb()
        info["available_vram_mb"] = get_available_vram_mb()
    
    return info


def clear_handlers() -> None:
    """Clear all cached handlers (forces re-creation)."""
    st.cache_resource.clear()
    force_clear_memory()
    logger.info("CUDA cache cleared")