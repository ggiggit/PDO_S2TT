"""Dependency-light wrapper for an exported streaming Zipformer encoder."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from torch import Tensor, nn


class TorchscriptStreamingZipformer(nn.Module):
    """Load the official icefall TorchScript model and expose encoder chunks.

    The WenetSpeech streaming-small export consumes 45 fbank frames per step:
    32 new frames plus 13 frames of right padding. It returns four 256-D
    encoder frames and updated left-context states.
    """

    def __init__(self, checkpoint: str | Path, device: str = "cpu") -> None:
        super().__init__()
        full_model = torch.jit.load(str(checkpoint), map_location=device)
        self.encoder = full_model.encoder
        self.encoder.eval()
        self.device_name = device
        self.chunk_length = int(self.encoder.chunk_size) * 2
        self.input_frames = self.chunk_length + int(self.encoder.pad_length)

    @torch.no_grad()
    def init_states(self, batch_size: int = 1) -> List[Tensor]:
        return self.encoder.get_init_states(
            batch_size=batch_size, device=torch.device(self.device_name)
        )

    @torch.no_grad()
    def forward_chunk(
        self, fbank_chunk: Tensor, states: List[Tensor]
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:
        if fbank_chunk.ndim != 3 or fbank_chunk.size(-1) != 80:
            raise ValueError("fbank_chunk must have shape (B, T, 80)")
        if fbank_chunk.size(1) != self.input_frames:
            raise ValueError(
                f"expected {self.input_frames} frames, got {fbank_chunk.size(1)}"
            )
        lengths = torch.full(
            (fbank_chunk.size(0),),
            self.input_frames,
            dtype=torch.int32,
            device=fbank_chunk.device,
        )
        return self.encoder(fbank_chunk, lengths, states)

    @torch.no_grad()
    def encode_fbank(self, fbank: Tensor) -> Tuple[List[Tensor], List[float]]:
        """Encode a complete fbank matrix using the real streaming state path."""
        if fbank.ndim != 2 or fbank.size(-1) != 80:
            raise ValueError("fbank must have shape (frames, 80)")
        fbank = fbank.to(self.device_name)
        # A short low-energy tail flushes the final partial chunk.
        tail = max(self.input_frames, self.chunk_length)
        fbank = torch.cat([fbank, fbank.new_full((tail, 80), -23.0258509)])
        states = self.init_states()
        chunks: List[Tensor] = []
        end_times: List[float] = []
        offset = 0
        while offset + self.input_frames <= fbank.size(0):
            window = fbank[offset : offset + self.input_frames].unsqueeze(0)
            encoder_out, _, states = self.forward_chunk(window, states)
            chunks.append(encoder_out)
            offset += self.chunk_length
            end_times.append(offset * 0.01)
        return chunks, end_times


def load_fbank(audio_path: str | Path, sample_rate: int = 16000) -> Tensor:
    """Compute 80-bin Kaldi-compatible fbank features for one audio file."""
    import torchaudio

    waveform, source_rate = torchaudio.load(str(audio_path))
    waveform = waveform.mean(dim=0, keepdim=True)
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)
    return torchaudio.compliance.kaldi.fbank(
        waveform,
        num_mel_bins=80,
        frame_length=25.0,
        frame_shift=10.0,
        dither=0.0,
        energy_floor=0.0,
        sample_frequency=float(sample_rate),
        snip_edges=False,
    )
