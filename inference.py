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
import gc
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")

def _getWindowingArray(window_size, fade_size):
    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)
    window = torch.ones(window_size)
    window[-fade_size:] *= fadeout
    window[:fade_size] *= fadein
    return window

def optimize_memory(aggressive=False):
    """Aggressive memory optimization"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    if aggressive and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.99)
        torch.cuda.empty_cache()

def get_system_memory_info():
    """Get system and GPU memory information"""
    system_memory = psutil.virtual_memory()
    memory_info = {
        'system_available_gb': system_memory.available / (1024**3),
        'system_used_gb': system_memory.used / (1024**3),
        'system_total_gb': system_memory.total / (1024**3)
    }
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        memory_info['gpu_used_gb'] = torch.cuda.memory_allocated() / (1024**3)
        memory_info['gpu_reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)
        memory_info['gpu_available_gb'] = 15 - memory_info['gpu_used_gb']
    
    return memory_info

class ChunkProcessor:
    def __init__(self, flashsr, device, chunk_size, fade_size):
        self.flashsr = flashsr
        self.device = device
        self.chunk_size = chunk_size
        self.fade_size = fade_size
        self.windowing_array = _getWindowingArray(chunk_size, fade_size)
        
        # Pre-allocate buffers to avoid repeated allocations
        self.current_chunk = None
        self.next_chunk = None
        
    def prepare_chunk(self, audio, start_idx):
        """Prepare chunk with minimal operations"""
        end_idx = min(start_idx + self.chunk_size, audio.shape[1])
        chunk = audio[:, start_idx:end_idx]
        
        if chunk.shape[-1] < self.chunk_size:
            pad_size = self.chunk_size - chunk.shape[-1]
            if chunk.shape[-1] > self.chunk_size // 2 + 1:
                chunk = torch.nn.functional.pad(chunk, (0, pad_size), mode='reflect')
            else:
                chunk = torch.nn.functional.pad(chunk, (0, pad_size), mode='constant', value=0)
        
        return chunk

def process_audio_fast(input_path, output_path, overlap, flashsr, device):
    # Skip if output already exists
    if os.path.exists(output_path):
        print(f"Skipping {input_path} - output already exists: {output_path}")
        return
    
    optimize_memory(aggressive=True)
    
    memory_info = get_system_memory_info()
    print(f"Memory before processing - System: {memory_info['system_available_gb']:.1f}GB available, "
          f"GPU: {memory_info.get('gpu_available_gb', 0):.1f}GB available")
    
    # Load audio - keep on GPU as much as possible
    audio, sr = UtilAudio.read(input_path, sample_rate=48000)
    audio = audio.to(device, non_blocking=True)

    C = 1228800  # chunk_size
    N = overlap
    step = C // N
    fade_size = C // 8
    print(f"N = {N} | C = {C} | step = {step} | fade_size = {fade_size}")

    border = C - step

    if len(audio.shape) == 1:
        audio = audio.unsqueeze(0)

    if audio.shape[1] > 2 * border and (border > 0):
        audio = torch.nn.functional.pad(audio, (border, border), mode='reflect')

    total_chunks = math.ceil(audio.size(1) / step)
    print(f"Total chunks: {total_chunks}")

    # Initialize processor
    processor = ChunkProcessor(flashsr, device, C, fade_size)
    
    # Pre-allocate result tensors on CPU with pinned memory for fast transfers
    result_shape = (audio.shape[0], audio.shape[1])
    result = torch.zeros(result_shape, dtype=torch.float32, device='cpu', pin_memory=True)
    counter = torch.zeros(result_shape, dtype=torch.float32, device='cpu', pin_memory=True)

    i = 0
    processed_chunks = 0
    
    # Pre-load first two chunks for pipelining
    current_chunk = processor.prepare_chunk(audio, i)
    next_start = i + step
    next_chunk = processor.prepare_chunk(audio, next_start) if next_start < audio.shape[1] else None

    progress_bar = tqdm(total=total_chunks, desc="Processing audio chunks", leave=False, unit="chunk")

    with torch.no_grad():
        # Enable all optimizations
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        
        if device.type == 'cuda':
            torch.cuda.set_per_process_memory_fraction(0.99)
        
        while i < audio.shape[1]:
            # Process current chunk
            chunk_start_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
            chunk_end_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
            
            if device.type == 'cuda':
                chunk_start_time.record()
            
            try:
                # Use mixed precision for maximum speed
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                    out = flashsr(current_chunk, lowpass_input=True)
                
                # Move output to CPU asynchronously
                out = out.cpu().float()
                
                if device.type == 'cuda':
                    chunk_end_time.record()
                    torch.cuda.synchronize()
                    # Uncomment to see chunk processing time
                    # chunk_time = chunk_start_time.elapsed_time(chunk_end_time)
                    # print(f"Chunk processed in {chunk_time:.2f}ms")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("GPU OOM, clearing cache and retrying...")
                    optimize_memory(aggressive=True)
                    # Retry without mixed precision
                    out = flashsr(current_chunk, lowpass_input=True)
                    out = out.cpu().float()
                else:
                    raise e
            
            # Apply windowing and accumulate results
            window = processor.windowing_array.clone()
            
            if i == 0:
                window[:fade_size] = 1
            elif i + C >= audio.shape[1]:
                window[-fade_size:] = 1
            
            end_idx = min(i + current_chunk.shape[-1], result.shape[1])
            actual_length = end_idx - i
            
            result[:, i:end_idx] += out[:, :actual_length] * window[:actual_length]
            counter[:, i:end_idx] += window[:actual_length]
            
            processed_chunks += 1
            progress_bar.update(1)
            
            # Prepare next chunk while processing current results
            i += step
            
            # Switch to next chunk
            current_chunk = next_chunk
            next_start = i + step
            
            # Pre-load next chunk if available
            if next_start < audio.shape[1]:
                next_chunk = processor.prepare_chunk(audio, next_start)
            else:
                next_chunk = None
            
            # Aggressive but smart memory management
            del out
            if processed_chunks % 3 == 0:  # More frequent cleanup
                optimize_memory(aggressive=False)  # Less aggressive during processing

    progress_bar.close()
    
    # Final processing
    final_output = result / counter
    final_output = final_output.numpy()
    np.nan_to_num(final_output, copy=False, nan=0.0)

    if audio.shape[1] > 2 * border and (border > 0):
        final_output = final_output[:, border:-border]

    # Write as MP3
    UtilAudio.write(output_path, final_output, 48000)
    
    # Final cleanup
    optimize_memory(aggressive=True)
    
    final_memory = get_system_memory_info()
    print(f"Memory after processing - System: {final_memory['system_available_gb']:.1f}GB available, "
          f"GPU: {final_memory.get('gpu_available_gb', 0):.1f}GB available")
    print(f'Success! Output file saved as {output_path}')

def main(input, output, overlap):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Maximum performance settings
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.set_per_process_memory_fraction(0.99)
        torch.cuda.empty_cache()
        
        # Set higher GPU clock speeds if available
        try:
            torch.cuda.set_device(device)
            # Force GPU to maximum performance state
            os.environ['CUDA_CACHE_DISABLE'] = '0'
        except:
            pass
        
        print("Enabled maximum performance CUDA optimizations")

    # Load model
    student_ldm_ckpt_path = './ckpts/student_ldm.pth'
    sr_vocoder_ckpt_path = './ckpts/sr_vocoder.pth'
    vae_ckpt_path = './ckpts/vae.pth'
    
    print("Loading FlashSR model with performance optimizations...")
    
    # Pre-allocate GPU memory
    if device.type == 'cuda':
        large_dummy = torch.zeros((1000, 1000, 50), device=device)
        del large_dummy
        torch.cuda.empty_cache()
    
    flashsr = FlashSR(student_ldm_ckpt_path, sr_vocoder_ckpt_path, vae_ckpt_path)
    flashsr = flashsr.to(device)
    flashsr.eval()
    
    # Safe warmup with proper dimensions
    if device.type == 'cuda':
        try:
            C = 245760
            # Use correct tensor shape for warmup
            dummy_input = torch.randn(1, C, device=device)
            print("Performing model warmup...")
            with torch.no_grad():
                _ = flashsr(dummy_input, lowpass_input=True)
            del dummy_input
            torch.cuda.empty_cache()
            print("Model warmup successful")
        except Exception as e:
            print(f"Model warmup skipped: {e}")
            optimize_memory(aggressive=True)
    
    # Suppress dynamo errors
    try:
        torch._dynamo.config.suppress_errors = True
    except:
        pass

    # Process files
    if Path(input).is_file():
        file_path = input
        filename = Path(input).name
        output_filename = Path(filename).stem + '.mp3'
        output_file_path = os.path.join(output, output_filename)
        Path(output).mkdir(parents=True, exist_ok=True)
        
        print(f"Processing single file: {filename}")
        process_audio_fast(file_path, output_file_path, overlap, flashsr, device)
    else:
        files = sorted(glob.glob(os.path.join(input, "*")))
        print(f"Found {len(files)} files. Checking which need processing...")
        
        # Identify files to process
        files_to_process = []
        for file_path in files:
            filename = Path(file_path).name
            output_filename = Path(filename).stem + '.mp3'
            output_file_path = os.path.join(output, output_filename)
            
            if not os.path.exists(output_file_path):
                files_to_process.append((file_path, output_file_path, filename))
            else:
                print(f"Skipping {filename} - output already exists")
        
        print(f"Processing {len(files_to_process)} files...")
        
        for file_idx, (file_path, output_file_path, filename) in enumerate(files_to_process):
            Path(output).mkdir(parents=True, exist_ok=True)
            print(f"Processing file {file_idx + 1}/{len(files_to_process)}: {filename}")
            
            optimize_memory(aggressive=True)
            process_audio_fast(file_path, output_file_path, overlap, flashsr, device)
            optimize_memory(aggressive=True)
            gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ultra-Fast Audio Inference Script")
    parser.add_argument("--input", type=str, required=True, help="Path to input wav file or folder")
    parser.add_argument("--output", type=str, required=True, help="Path to output folder")
    parser.add_argument("--overlap", type=int, help="Overlap", default=2)

    args = parser.parse_args()

    main(args.input, args.output, args.overlap)
