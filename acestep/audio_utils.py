"""
Audio saving and transcoding utility module

Independent audio file operations outside of handler, supporting:
- Save audio tensor/numpy to files (default FLAC format, fast)
- Format conversion (FLAC/WAV/MP3)
- Batch processing
"""


import io
import json
import os
import subprocess
import hashlib
from pathlib import Path
from typing import Union, Optional, List, Tuple
import torch
import numpy as np
import torchaudio
from loguru import logger


def normalize_audio(audio_data: Union[torch.Tensor, np.ndarray], target_db: float = -1.0) -> Union[torch.Tensor, np.ndarray]:
    """
    Apply peak normalization to audio data.
    
    Args:
        audio_data: Audio data as torch.Tensor or numpy.ndarray
        target_db: Target peak level in dB (default: -1.0)
        
    Returns:
        Normalized audio data in the same format as input
    """
    # Create a copy to avoid modifying original in-place
    if isinstance(audio_data, torch.Tensor):
        audio = audio_data.clone()
        is_tensor = True
    else:
        audio = audio_data.copy()
        is_tensor = False
        
    # Calculate current peak
    if is_tensor:
        peak = torch.max(torch.abs(audio))
    else:
        peak = np.max(np.abs(audio))
        
    # Handle silence/near-silence to avoid division by zero or extreme gain
    if peak < 1e-6:
        return audio_data
        
    # Convert target dB to linear amplitude
    target_amp = 10 ** (target_db / 20.0)
    
    # Calculate needed gain
    gain = target_amp / peak
    
    # Apply gain
    audio = audio * gain
    
    return audio


import soundfile as sf

class AudioSaver:
    """Audio saving and transcoding utility class"""

    def __init__(self, default_format: str = "wav32"):
        """
        Initialize audio saver

        Args:
            default_format: Default save format ('flac', 'mp3', 'wav32')
        """
        self.default_format = default_format.lower()
        if self.default_format not in ["flac", "mp3", "wav32"]:
            logger.warning(f"Unsupported format {default_format}, using 'wav32'")
            self.default_format = "wav32"

    def save_audio(
        self,
        audio_data: Union[torch.Tensor, np.ndarray],
        output_path: Union[str, Path],
        sample_rate: int = 48000,
        format: Optional[str] = None,
        channels_first: bool = True,
    ) -> str:
        """
        Save audio data to file
        """
        format = (format or self.default_format).lower()
        output_path = Path(output_path)

        # Convert to torch tensor
        if isinstance(audio_data, np.ndarray):
            audio_tensor = torch.from_numpy(audio_data).float()
        else:
            audio_tensor = audio_data.cpu().float()

        audio_tensor = audio_tensor.contiguous()

        try:
            # --- WAV32 PCM_32 через soundfile ---
            if format == "wav32":
                audio_np = audio_tensor.transpose(0, 1).numpy()
                sf.write(str(output_path), audio_np, sample_rate, subtype="PCM_32", format="WAV")
                logger.debug(f"[AudioSaver] Saved audio to {output_path} (wav32 PCM_32, {sample_rate}Hz)")
                return str(output_path)

            # --- FLAC / MP3 через torchaudio+soundfile ---
            if format in ["flac", "mp3"]:
                torchaudio.save(
                    str(output_path),
                    audio_tensor,
                    sample_rate,
                    channels_first=True,
                    backend="soundfile",
                )
                logger.debug(f"[AudioSaver] Saved audio to {output_path} ({format}, {sample_rate}Hz)")
                return str(output_path)

        except Exception as e:
            try:
                audio_np = audio_tensor.transpose(0, 1).numpy()
                if format == "wav32":
                    sf.write(str(output_path), audio_np, sample_rate, format="WAV", subtype="PCM_32")
                else:
                    sf.write(str(output_path), audio_np, sample_rate, format=format.upper())
                logger.debug(f"[AudioSaver] Fallback soundfile Saved audio to {output_path} ({format}, {sample_rate}Hz)")
                return str(output_path)
            except Exception as inner_e:
                logger.error(f"[AudioSaver] Failed to save audio: {e} -> Fallback failed: {inner_e}")
                raise
    
    def convert_audio(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        output_format: str,
        remove_input: bool = False,
    ) -> str:
        """
        Convert audio format
        
        Args:
            input_path: Input audio file path
            output_path: Output audio file path
            output_format: Target format ('flac', 'mp3', 'wav32')
            remove_input: Whether to delete input file
        
        Returns:
            Output file path
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Load audio
        audio_tensor, sample_rate = torchaudio.load(str(input_path))
        
        # Save as new format
        output_path = self.save_audio(
            audio_tensor,
            output_path,
            sample_rate=sample_rate,
            format=output_format,
            channels_first=True
        )
        
        # Delete input file if needed
        if remove_input:
            input_path.unlink()
            logger.debug(f"[AudioSaver] Removed input file: {input_path}")
        
        return output_path
    
    def save_batch(
        self,
        audio_batch: Union[List[torch.Tensor], torch.Tensor],
        output_dir: Union[str, Path],
        file_prefix: str = "audio",
        sample_rate: int = 48000,
        format: Optional[str] = None,
        channels_first: bool = True,
    ) -> List[str]:
        """
        Save audio batch
        
        Args:
            audio_batch: Audio batch, List[tensor] or tensor [batch, channels, samples]
            output_dir: Output directory
            file_prefix: File prefix
            sample_rate: Sample rate
            format: Audio format
            channels_first: Tensor format flag
        
        Returns:
            List of saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process batch
        if isinstance(audio_batch, torch.Tensor) and audio_batch.dim() == 3:
            # [batch, channels, samples]
            audio_list = [audio_batch[i] for i in range(audio_batch.shape[0])]
        elif isinstance(audio_batch, list):
            audio_list = audio_batch
        else:
            audio_list = [audio_batch]
        
        saved_paths = []
        for i, audio in enumerate(audio_list):
            output_path = output_dir / f"{file_prefix}_{i:04d}"
            saved_path = self.save_audio(
                audio,
                output_path,
                sample_rate=sample_rate,
                format=format,
                channels_first=channels_first
            )
            saved_paths.append(saved_path)
        
        return saved_paths


def get_lora_weights_hash(dit_handler) -> str:
    """Compute an MD5 hash identifying the currently loaded LoRA adapter weights.

    Iterates over the handler's LoRA service registry to find adapter weight
    file paths, then hashes each file to produce a combined fingerprint.

    Args:
        dit_handler: DiT handler instance with LoRA state attributes.

    Returns:
        Hex digest string uniquely identifying the loaded LoRA weights,
        or empty string if no LoRA is active.
    """
    if not getattr(dit_handler, "lora_loaded", False):
        return ""
    if not getattr(dit_handler, "use_lora", False):
        return ""

    lora_service = getattr(dit_handler, "_lora_service", None)
    if lora_service is None or not lora_service.registry:
        return ""

    hash_obj = hashlib.sha256()
    found_any = False

    for adapter_name in sorted(lora_service.registry.keys()):
        meta = lora_service.registry[adapter_name]
        lora_path = meta.get("path")
        if not lora_path:
            continue

        # Try common weight file names at lora_path
        candidates = []
        if os.path.isfile(lora_path):
            candidates.append(lora_path)
        elif os.path.isdir(lora_path):
            for fname in (
                "adapter_model.safetensors",
                "adapter_model.bin",
                "lokr_weights.safetensors",
            ):
                fpath = os.path.join(lora_path, fname)
                if os.path.isfile(fpath):
                    candidates.append(fpath)

        for fpath in candidates:
            try:
                with open(fpath, "rb") as f:
                    while True:
                        chunk = f.read(1 << 20)  # 1 MB chunks
                        if not chunk:
                            break
                        hash_obj.update(chunk)
                found_any = True
            except OSError:
                continue

    return hash_obj.hexdigest() if found_any else ""


def get_audio_file_hash(audio_file) -> str:
    """
    Get hash identifier for an audio file.
    
    Args:
        audio_file: Path to audio file (str) or file-like object
    
    Returns:
        Hash string or empty string
    """
    if audio_file is None:
        return ""
    
    try:
        if isinstance(audio_file, str):
            if os.path.exists(audio_file):
                with open(audio_file, 'rb') as f:
                    return hashlib.sha256(f.read()).hexdigest()
            return hashlib.sha256(audio_file.encode('utf-8')).hexdigest()
        elif hasattr(audio_file, 'name'):
            return hashlib.sha256(str(audio_file.name).encode('utf-8')).hexdigest()
        return hashlib.sha256(str(audio_file).encode('utf-8')).hexdigest()
    except Exception:
        return hashlib.sha256(str(audio_file).encode('utf-8')).hexdigest()


def generate_uuid_from_params(params_dict) -> str:
    """
    Generate deterministic UUID from generation parameters.
    Same parameters will always generate the same UUID.
    
    Args:
        params_dict: Dictionary of parameters
    
    Returns:
        UUID string
    """
    
    params_json = json.dumps(params_dict, sort_keys=True, ensure_ascii=False)
    hash_obj = hashlib.sha256(params_json.encode('utf-8'))
    hash_hex = hash_obj.hexdigest()
    uuid_str = f"{hash_hex[0:8]}-{hash_hex[8:12]}-{hash_hex[12:16]}-{hash_hex[16:20]}-{hash_hex[20:32]}"
    return uuid_str


def generate_uuid_from_audio_data(
    audio_data: Union[torch.Tensor, np.ndarray],
    seed: Optional[int] = None
) -> str:
    """
    Generate UUID from audio data (for caching/deduplication)
    
    Args:
        audio_data: Audio data
        seed: Optional seed value
    
    Returns:
        UUID string
    """
    if isinstance(audio_data, torch.Tensor):
        # Convert to numpy and calculate hash
        audio_np = audio_data.cpu().numpy()
    else:
        audio_np = audio_data
    
    # Calculate data hash
    data_hash = hashlib.sha256(audio_np.tobytes()).hexdigest()
    
    if seed is not None:
        combined = f"{data_hash}_{seed}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    return data_hash


# Global default instance
_default_saver = AudioSaver(default_format="flac")


def save_audio(
    audio_data: Union[torch.Tensor, np.ndarray],
    output_path: Union[str, Path],
    sample_rate: int = 48000,
    format: Optional[str] = None,
    channels_first: bool = True,
) -> str:
    """
    Convenience function: save audio (using default configuration)
    
    Args:
        audio_data: Audio data
        output_path: Output path
        sample_rate: Sample rate
        format: Format (default flac)
        channels_first: Tensor format flag
    
    Returns:
        Saved file path
    """
    return _default_saver.save_audio(
        audio_data, output_path, sample_rate, format, channels_first
    )

