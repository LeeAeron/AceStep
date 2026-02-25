"""Audio IO and normalization helpers for handler decomposition."""

import math
import random
from typing import Optional

import torch
from loguru import logger


class IoAudioMixin:
    """Mixin containing audio file loading and normalization helpers.

    Depends on host members:
    - Method: ``is_silence`` (provided by ``MemoryUtilsMixin`` in this decomposition).
    """

    def _normalize_audio_to_stereo_48k(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        """Normalize audio tensor to stereo at 48kHz.

        Args:
            audio: Tensor in [channels, samples] or [samples] format.
            sr: Source sample rate.

        Returns:
            Tensor in [2, samples] at 48kHz, clamped to [-1.0, 1.0] (float32).
        """
        # Ensure channel dim
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        if audio.shape[0] == 1:
            audio = torch.cat([audio, audio], dim=0)

        audio = audio[:2]

        if sr != 48000:
            import torchaudio
            audio = torchaudio.transforms.Resample(sr, 48000)(audio)

        # Ensure float32 and clamp to [-1, 1]
        if audio.dtype != torch.float32:
            audio = audio.float()
        return torch.clamp(audio, -1.0, 1.0)

    def _to_pcm16(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert float32 audio in [-1,1] to int16 PCM.

        Args:
            audio: float32 tensor in [-1,1], shape [2, samples]

        Returns:
            int16 tensor with same shape.
        """
        if audio.dtype != torch.float32:
            audio = audio.float()
        # clamp again for safety, scale and convert
        return (torch.clamp(audio, -1.0, 1.0) * 32767.0).to(torch.int16)

    def process_reference_audio(self, audio_file: Optional[str]) -> Optional[torch.Tensor]:
        """Load and normalize reference audio, then sample 3x10s segments.

        Processing is done in float32 for resampling and silence detection,
        but the returned tensor is converted to 16-bit PCM (int16) at 48kHz.

        Returns:
            30-second stereo int16 tensor (shape [2, 30*48000]) or None.
        """
        if audio_file is None:
            return None

        try:
            import torchaudio
            audio, sr = torchaudio.load(audio_file)
            logger.debug(f"[process_reference_audio] Reference audio shape: {audio.shape}")
            logger.debug(f"[process_reference_audio] Reference audio sample rate: {sr}")
            logger.debug(
                f"[process_reference_audio] Reference audio duration: {audio.shape[-1] / sr:.6f} seconds"
            )

            # Normalize to float32 stereo 48kHz
            audio = self._normalize_audio_to_stereo_48k(audio, sr)

            # If silent, nothing to do
            if self.is_silence(audio):
                return None

            target_frames = 30 * 48000
            segment_frames = 10 * 48000

            # If shorter than 30s, repeat to reach target length (float32)
            if audio.shape[-1] < target_frames:
                repeat_times = math.ceil(target_frames / audio.shape[-1])
                audio = audio.repeat(1, repeat_times)

            total_frames = audio.shape[-1]
            segment_size = total_frames // 3

            front_start = random.randint(0, max(0, segment_size - segment_frames))
            front_audio = audio[:, front_start : front_start + segment_frames]

            middle_start = segment_size + random.randint(0, max(0, segment_size - segment_frames))
            middle_audio = audio[:, middle_start : middle_start + segment_frames]

            back_start = 2 * segment_size + random.randint(
                0, max(0, (total_frames - 2 * segment_size) - segment_frames)
            )
            back_audio = audio[:, back_start : back_start + segment_frames]

            concatenated = torch.cat([front_audio, middle_audio, back_audio], dim=-1)

            # Final safety clamp and convert to PCM16 for downstream (ACE-Step)
            return self._to_pcm16(concatenated)

        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning(f"[process_reference_audio] Invalid or unsupported reference audio: {exc}")
            return None

    def process_src_audio(self, audio_file: Optional[str]) -> Optional[torch.Tensor]:
        """Load and normalize source audio for remix/extract flows.

        Args:
            audio_file: Path to source audio file.

        Returns:
            Normalized stereo 48kHz tensor, or ``None`` on error/empty input.
        """
        if audio_file is None:
            return None

        try:
            import torchaudio
            audio, sr = torchaudio.load(audio_file)
            return self._normalize_audio_to_stereo_48k(audio, sr)
        except (OSError, RuntimeError, ValueError):
            logger.exception("[process_src_audio] Error processing source audio")
            return None
