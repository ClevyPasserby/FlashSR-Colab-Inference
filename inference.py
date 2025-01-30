import argparse
import torch
import numpy as np
from TorchJaekwon.Util.UtilAudio import UtilAudio
from TorchJaekwon.Util.UtilData import UtilData
from tqdm import tqdm
from FlashSR.FlashSR import FlashSR
import warnings
import math
import os
from pathlib import Path
import glob

warnings.filterwarnings("ignore")


def _getWindowingArray(window_size, fade_size):
    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)
    window = torch.ones(window_size)
    window[-fade_size:] *= fadeout
    window[:fade_size] *= fadein
    return window

def process_audio(input_path, output_path, overlap, flashsr, device):
    audio, sr = UtilAudio.read(input_path, sample_rate=48000)
    audio = audio.to(device)

    C = 245760  # chunk_size
    N = overlap
    step = C // N
    fade_size = C // 10
    print(f"N = {N} | C = {C} | step = {step} | fade_size = {fade_size}")

    border = C - step

    if len(audio.shape) == 1:
        audio = audio.unsqueeze(0)

    if audio.shape[1] > 2 * border and (border > 0):
        audio = torch.nn.functional.pad(audio, (border, border), mode='reflect')

    total_chunks = math.ceil(audio.size(1) / step)
    print(total_chunks)

    windowingArray = _getWindowingArray(C, fade_size)

    result = torch.zeros((1,) + tuple(audio.shape), dtype=torch.float32)
    counter = torch.zeros((1,) + tuple(audio.shape), dtype=torch.float32)

    i = 0
    progress_bar = tqdm(total=total_chunks, desc="Processing audio chunks", leave=False, unit="chunk")

    while i < audio.shape[1]:
        part = audio[:, i:i + C]
        length = part.shape[-1]
        if length < C:
            if length > C // 2 + 1:
                part = torch.nn.functional.pad(input=part, pad=(0, C - length), mode='reflect')
            else:
                part = torch.nn.functional.pad(input=part, pad=(0, C - length, 0, 0), mode='constant', value=0)

        out = flashsr(part, lowpass_input=True).cpu()

        window = windowingArray
        if i == 0:
            window[:fade_size] = 1
        elif i + C >= audio.shape[1]:
            window[-fade_size:] = 1

        result[..., i:i + length] += out[..., :length] * window[..., :length]
        counter[..., i:i + length] += window[..., :length]

        i += step
        progress_bar.update(1)

    progress_bar.close()

    final_output = result / counter
    final_output = final_output.squeeze(0).numpy()
    np.nan_to_num(final_output, copy=False, nan=0.0)

    if audio.shape[1] > 2 * border and (border > 0):
        final_output = final_output[..., border:-border]

    UtilAudio.write(output_path, final_output, 48000)
    print(f'Success! Output file saved as {output_path}')


def main(input, output, overlap):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    student_ldm_ckpt_path = './ckpts/student_ldm.pth'
    sr_vocoder_ckpt_path = './ckpts/sr_vocoder.pth'
    vae_ckpt_path = './ckpts/vae.pth'
    flashsr = FlashSR(student_ldm_ckpt_path, sr_vocoder_ckpt_path, vae_ckpt_path)
    flashsr = flashsr.to(device)

    if Path(input).is_file():
        file_path = input
        filename = Path(input).name
        Path(output).mkdir(parents=True, exist_ok=True)
        process_audio(file_path, os.path.join(output, filename), overlap, flashsr, device)
    else:
        for file_path in sorted(glob.glob(os.path.join(input, "*"))):
            filename = Path(file_path).name
            Path(output).mkdir(parents=True, exist_ok=True)
            process_audio(file_path, os.path.join(output, filename), overlap, flashsr, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Inference Script")
    parser.add_argument("--input", type=str, required=True, help="Path to input wav file or folder")
    parser.add_argument("--output", type=str, required=True, help="Path to output folder")
    parser.add_argument("--overlap", type=int, help="Overlap", default=2)

    args = parser.parse_args()

    main(args.input, args.output, args.overlap)
