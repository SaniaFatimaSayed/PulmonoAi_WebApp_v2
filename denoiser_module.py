import os
import torch
import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio


_denoiser_model = None


def load_denoiser():
    global _denoiser_model

    if _denoiser_model is None:
        _denoiser_model = pretrained.dns64()
        _denoiser_model.eval()

    return _denoiser_model


def denoise_audio(input_path, output_dir="outputs/denoised_audio"):
    os.makedirs(output_dir, exist_ok=True)

    model = load_denoiser()

    wav, sr = torchaudio.load(input_path)

    wav = convert_audio(
        wav,
        sr,
        model.sample_rate,
        model.chin
    )

    with torch.no_grad():
        denoised = model(wav[None])[0]

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_denoised.wav")

    torchaudio.save(
        output_path,
        denoised.cpu(),
        model.sample_rate
    )

    return output_path