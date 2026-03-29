#!/usr/bin/env python3
import os
import torch
import numpy as np
import argparse
import random
import time
import multiprocessing
import tempfile
import json
import pickle
import sys
from typing import Optional, List, Dict, Any, Union
from tqdm import tqdm
from OCC.Extend.DataExchange import write_step_file
from model import ARModel
from utils import (
    reconstruct_cad_from_sequence, 
    load_se_vqvae_model,
)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load JSON config for generation script.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_files_with_timeout_multiprocess(solid, step_path, stl_path, write_timeout=30):

    def write_worker(temp_step_path, final_step_path, final_stl_path, result_file_path):
        import json  # Must import at the top of the function to avoid free variable issues
        try:
            # Try importing OpenCASCADE utilities
            try:
                from OCC.Extend.DataExchange import write_stl_file, read_step_file  # type: ignore
                use_step_reader = False
            except ImportError:
                try:
                    from OCC.Extend.DataExchange import write_stl_file  # type: ignore
                    from OCC.Core.STEPControl import STEPControl_Reader  # type: ignore
                    from OCC.Core.IFSelect import IFSelect_RetDone       # type: ignore
                    use_step_reader = True
                except ImportError:
                    with open(result_file_path, 'w') as f:
                        json.dump({'status': 'error', 'error': 'Failed to import OpenCASCADE modules'}, f)
                    return

            # Read STEP
            if not use_step_reader:
                try:
                    shape = read_step_file(temp_step_path)
                except Exception:
                    # Fallback to STEPControl_Reader
                    use_step_reader = True

            if use_step_reader:
                from OCC.Core.STEPControl import STEPControl_Reader  # type: ignore
                from OCC.Core.IFSelect import IFSelect_RetDone       # type: ignore
                step_reader = STEPControl_Reader()
                status = step_reader.ReadFile(temp_step_path)
                if status != IFSelect_RetDone:
                    with open(result_file_path, 'w') as f:
                        json.dump({'status': 'error', 'error': f'Failed to read temporary STEP file: {temp_step_path}'}, f)
                    return
                step_reader.TransferRoots()
                shape = step_reader.OneShape()

            if shape.IsNull():
                with open(result_file_path, 'w') as f:
                    json.dump({'status': 'error', 'error': 'Loaded shape from STEP file is null'}, f)
                return

            # 1) Write STL first
            try:
                write_stl_file(shape, final_stl_path, linear_deflection=0.001, angular_deflection=0.5)
            except Exception as e:
                with open(result_file_path, 'w') as f:
                    json.dump({'status': 'stl_failed', 'error': f'Failed to write STL: {e}'}, f)
                return

            # 2) Copy STEP only after STL is successfully written
            import shutil
            shutil.copy2(temp_step_path, final_step_path)

            with open(result_file_path, 'w') as f:
                json.dump({'status': 'success', 'error': None}, f)

        except Exception as e:
            import traceback
            error_str = f"Child process error: {e}\n{traceback.format_exc()}"
            with open(result_file_path, 'w') as f:
                json.dump({'status': 'error', 'error': error_str}, f)

    try:
        # Unique temporary file name
        timestamp = int(time.time() * 1000000)
        process_id = os.getpid()
        unique_id = f"{timestamp}_{process_id}"

        if os.name == 'nt':
            temp_dir = tempfile.gettempdir()
            temp_step_path = os.path.join(temp_dir, f"brep_step_{unique_id}.step")
            result_file_path = os.path.join(temp_dir, f"brep_result_{unique_id}.json")
        else:
            temp_step_path = f"/tmp/brep_step_{unique_id}.step"
            result_file_path = f"/tmp/brep_result_{unique_id}.json"

        # Write STEP in main process (usually will not hang)
        from OCC.Extend.DataExchange import write_step_file
        write_step_file(solid, temp_step_path)

        # Start child process for STL writing and STEP copying
        import multiprocessing
        process = multiprocessing.Process(
            target=write_worker,
            args=(temp_step_path, step_path, stl_path, result_file_path)
        )
        process.start()
        process.join(timeout=write_timeout)

        # Timeout handling
        if process.is_alive():
            print(f"File writing timed out ({write_timeout}s), terminating child process...")
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()

            # Clean up temporary files
            try:
                if os.path.exists(temp_step_path):
                    os.unlink(temp_step_path)
                if os.path.exists(result_file_path):
                    os.unlink(result_file_path)
            except Exception:
                pass

            return 'timeout', f"File writing timed out ({write_timeout}s)"

        # Read result from child process
        if os.path.exists(result_file_path):
            import json
            with open(result_file_path, 'r') as f:
                result = json.load(f)
            status = result.get('status', 'error')
            error = result.get('error', 'Unknown error')
        else:
            status = 'error'
            error = 'Child process did not create a result file'

        # Cleanup temporary files
        try:
            if os.path.exists(temp_step_path):
                os.unlink(temp_step_path)
            if os.path.exists(result_file_path):
                os.unlink(result_file_path)
        except Exception:
            pass

        return status, error

    except Exception as e:
        return 'error', f"Unexpected error during multiprocessing write: {e}"

def load_checkpoint(model_path: str) -> Dict:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"Successfully loaded checkpoint: {model_path}")
        return checkpoint
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")

def load_model_config(config: Dict[str, Any], dataset_type: str) -> Dict[str, Any]:
    """
    Load model configuration from the config dictionary and calculate derived values.
    """
    if dataset_type not in config:
        raise ValueError(f"Dataset type '{dataset_type}' not found in config.")
    
    model_config = config[dataset_type]

    # 1. Load required base parameters
    required_keys = ['face_index_size', 'se_codebook_size', 'bbox_index_size']
    missing_keys = [key for key in required_keys if key not in model_config]
    if missing_keys:
        raise ValueError(f"Config missing required keys: {', '.join(missing_keys)}")

    # 2. Set fixed parameters
    model_config['special_token_size'] = 4
    model_config['bbox_tokens_per_element'] = 6
    
    # 3. Calculate offsets and tokens
    # Layout: [Face Indices] [SE Codebook] [BBox Indices] [Special Tokens]
    
    face_index_size = model_config['face_index_size']
    se_codebook_size = model_config['se_codebook_size']
    bbox_index_size = model_config['bbox_index_size']
    
    model_config['face_index_offset'] = 0
    model_config['se_token_offset'] = model_config['face_index_offset'] + face_index_size
    model_config['bbox_token_offset'] = model_config['se_token_offset'] + se_codebook_size
    
    special_token_start = model_config['bbox_token_offset'] + bbox_index_size
    
    model_config['START_TOKEN'] = special_token_start
    model_config['SEP_TOKEN'] = special_token_start + 1
    model_config['END_TOKEN'] = special_token_start + 2
    model_config['PAD_TOKEN'] = special_token_start + 3
    
    model_config['vocab_size'] = special_token_start + model_config['special_token_size']

    print(f"Successfully loaded and calculated model config for dataset type: {dataset_type}")
    print(f"Vocab size: {model_config['vocab_size']}")
    print(f"Special tokens: START={model_config['START_TOKEN']}, SEP={model_config['SEP_TOKEN']}, END={model_config['END_TOKEN']}, PAD={model_config['PAD_TOKEN']}")
          
    return model_config

def infer_se_tokens_per_element(se_vqvae_model: Any, device: str) -> int:
    """
    Infer the number of SE tokens per element from the VQ-VAE encoder output.
    """
    try:
        in_channels = se_vqvae_model.encoder.conv_in.weight.shape[1]
    except Exception:
        in_channels = 3

    se_random_data = np.random.rand(in_channels, 32, 32).astype(np.float32)

    with torch.no_grad():
        x = torch.tensor(se_random_data, dtype=torch.float32).unsqueeze(0).to(device)
        h = se_vqvae_model.encoder(x)
        h = se_vqvae_model.quant_conv(h)
        _, _, indices = se_vqvae_model.quantize(h)
        token_indices = (
            indices[2] if isinstance(indices, tuple) and len(indices) > 2
            else indices[0] if isinstance(indices, tuple)
            else indices
        )

    return int(token_indices.numel())

def init_ar_model(vocab_size: int, pad_token_id: int, checkpoint: Dict = None, device: str = 'cpu') -> ARModel:
    model_config = {
        'vocab_size': vocab_size,
        'd_model': 256,
        'nhead': 8,
        'num_layers': 8,
        'dim_feedforward': 1024,
        'dropout': 0.0,
        'max_seq_len': 2048,
        'pad_token_id': pad_token_id
    }
    
    if checkpoint and 'config' in checkpoint and checkpoint['config']:
        config_dict = checkpoint['config']
        if 'n_embd' in config_dict:
            model_config['d_model'] = config_dict['n_embd']
        if 'n_head' in config_dict:
            model_config['nhead'] = config_dict['n_head']
        if 'n_layer' in config_dict:
            model_config['num_layers'] = config_dict['n_layer']
        if 'n_inner' in config_dict:
            model_config['dim_feedforward'] = config_dict['n_inner']
        if 'n_positions' in config_dict:
            model_config['max_seq_len'] = config_dict['n_positions']
    
    model = ARModel(**model_config).to(device)
    return model

def load_model_weights(model: ARModel, checkpoint: Dict):
    try:
        state_dict = checkpoint['model_state_dict']

        def strip_prefix(sd, prefix):
            if any(k.startswith(prefix) for k in sd.keys()):
                sd = { (k[len(prefix):] if k.startswith(prefix) else k): v for k, v in sd.items() }
            return sd

        state_dict = strip_prefix(state_dict, 'module.')
        state_dict = strip_prefix(state_dict, 'model.')

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if missing_keys:
            print(f"Missing keys: {missing_keys[:10]}{' ...' if len(missing_keys)>10 else ''}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys[:10]}{' ...' if len(unexpected_keys)>10 else ''}")

        model.eval()
        print("Model weights loaded successfully")

    except Exception as e:
        raise RuntimeError(f"Failed to load model weights: {e}")

def create_prompt(vocab_info: Dict, device: str, prompt_type: str = "auto") -> torch.Tensor:
    """
    Create initial prompt for generating CAD sequence
    """
    START_TOKEN = vocab_info['START_TOKEN']
    
    if prompt_type == "start_only":
        # Only include START token
        prompt = [START_TOKEN]
    elif prompt_type == "with_first_face":
        # Include START token and some random bbox and face tokens as example
        prompt = [START_TOKEN]
        bbox_tokens_per_element = vocab_info['bbox_tokens_per_element']
        bbox_token_offset = vocab_info['bbox_token_offset']
        bbox_codebook_size = vocab_info['bbox_index_size']
        se_tokens_per_element = vocab_info['se_tokens_per_element']
        se_token_offset = vocab_info['se_token_offset']
        se_codebook_size = vocab_info['se_codebook_size']
        face_index_offset = vocab_info['face_index_offset']
        
        for _ in range(bbox_tokens_per_element):
            prompt.append(bbox_token_offset + np.random.randint(0, bbox_codebook_size))
        for _ in range(se_tokens_per_element):
            prompt.append(se_token_offset + np.random.randint(0, se_codebook_size))
        prompt.append(face_index_offset)  # First face index
    else:  # auto
        # Automatically select the simplest prompt
        prompt = [START_TOKEN]
    
    return torch.tensor(prompt, dtype=torch.long, device=device).unsqueeze(0)

def generate_sequence(
    model: ARModel,
    vocab_info: Dict,
    device: str,
    prompt: Optional[torch.Tensor] = None,
    max_length: int = 2048,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.6,
    repetition_penalty: float = 1,
    no_repeat_ngram_size: int = 0,
) -> List[int]:
    
    if prompt is None:
        prompt = create_prompt(vocab_info, device)
    
    if prompt.dim() == 1:
        prompt = prompt.unsqueeze(0)
    
    model.eval()
    
    # Ensure max_length does not exceed model capabilities
    effective_max_length = min(max_length, getattr(model.config, 'n_positions', 2048))
    
    START_TOKEN = vocab_info['START_TOKEN']
    END_TOKEN = vocab_info['END_TOKEN']
    PAD_TOKEN = vocab_info['PAD_TOKEN']
    
    with torch.no_grad():
        try:
            generated = model.generate(
                input_ids=prompt,
                max_length=effective_max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                pad_token_id=PAD_TOKEN,
                eos_token_id=END_TOKEN,
                bos_token_id=START_TOKEN,
                num_beams=1,
                early_stopping=False,
                do_sample=True,
                use_cache=True,
                output_scores=False,
                return_dict_in_generate=False
            )
            
        except Exception as e:
            print(f"Exception occurred: {e}")
            # When an exception occurs, return only the sequence containing START_TOKEN
            generated = [torch.cat([prompt[0], torch.tensor([END_TOKEN], device=device)])]
    
    return generated[0].cpu().tolist()

def generate_and_reconstruct_single(
    ar_model: ARModel,
    se_vqvae_model: Any,
    vocab_info: Dict,
    device: str,
    max_length: int = 2048,
    temperature: float = 1,
    top_k: int = 0,
    top_p: float = 0.6,
    repetition_penalty: float = 1,
    no_repeat_ngram_size: int = 0,
    scale_factor: float = 1.0,
    save_step: bool = True,
    output_dir: str = "result/generated_brep",
    filename_prefix: str = "generated",
    silent: bool = False,
    write_timeout: int = 30
) -> Dict[str, Any]:
    """
    Single sample generation and reconstruction.
    """
    if not silent:
        print("\n=== Generate Single CAD Model ===")

    result = {
        'sequence': None,
        'solid': None,
        'saved_path': None,   # STEP path
        'stl_path': None,     # STL path
        'stl_saved': False,
        'step_saved': False,
        'error': None
    }

    try:
        # 1) Sequence generation
        if not silent:
            print("1. Generating CAD sequence...")
        try:
            sequence = generate_sequence(
                model=ar_model,
                vocab_info=vocab_info,
                device=device,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
        except Exception as e:
            result['error'] = f"Sequence generation error: {e}"
            if not silent:
                print(f"Sequence generation error: {e}")
            return result
        
        # Check whether the sequence is empty or invalid
        if not sequence:
            result['error'] = "Sequence generation failed: returned an empty sequence"
            return result
        
        # Check whether the sequence is too short (could be only START_TOKEN + END_TOKEN, indicating failure)
        min_valid_length = 3  # At least START_TOKEN + some content + END_TOKEN
        if len(sequence) < min_valid_length:
            result['error'] = f"Sequence generation failed: sequence too short (length={len(sequence)}, minimum={min_valid_length})"
            return result
        
        START_TOKEN = vocab_info['START_TOKEN']
        END_TOKEN = vocab_info['END_TOKEN']
        if len(sequence) == 2 and sequence[0] == START_TOKEN and sequence[1] == END_TOKEN:
            result['error'] = "Sequence generation failed: only start and end tokens were produced"
            return result

        result['sequence'] = sequence
        if not silent:
            print(f"Generated sequence length: {len(sequence)}")

        # 2) BREP reconstruction
        if not silent:
            print("2. Reconstructing BREP model...")
        solid = reconstruct_cad_from_sequence(
            sequence=sequence,
            vocab_info=vocab_info,
            se_vqvae_model=se_vqvae_model,
            device=device,
            scale_factor=scale_factor,
            verbose=False
        )
        if solid is None:
            result['error'] = "BREP reconstruction failed"
            return result

        result['solid'] = solid
        if not silent:
            print("BREP reconstruction succeeded")

        # 3) Save (STL first, then STEP)
        if save_step:
            if not silent:
                print("3. Saving STEP/STL files (STL first, then STEP)...")
            timestamp = f"{int(time.time())}_{int((time.time() % 1) * 1000000):06d}"
            stepname = f"{filename_prefix}_{timestamp}.step"
            stlname  = f"{filename_prefix}_{timestamp}.stl"

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            steppath = os.path.join(output_dir, stepname)
            stlpath  = os.path.join(output_dir, stlname)

            try:
                status, error = write_files_with_timeout_multiprocess(
                    solid, steppath, stlpath, write_timeout=write_timeout
                )

                if status == 'success':
                    result['saved_path'] = steppath
                    result['stl_path']   = stlpath
                    result['stl_saved']  = True
                    result['step_saved'] = True
                    if not silent:
                        print(f"Saved STL: {stlpath}")
                        print(f"Saved STEP: {steppath}")

                elif status == 'stl_failed':
                    result['error'] = f"STL write failed: {error}"
                    if not silent:
                        print("STL write failed, skipping this sample (excluded from main statistics)")

                elif status == 'timeout':
                    result['error'] = error
                    if not silent:
                        print("File write timed out, skipping this sample (excluded from main statistics)")

                else:
                    result['error'] = f"Save failed: {error}"
                    if not silent:
                        print(f"Save failed: {error}")

            except Exception as e:
                result['error'] = f"Unexpected write error: {e}"
                if not silent:
                    print(f"Unexpected write error: {e}")

        return result

    except Exception as e:
        result['error'] = str(e)
        if not silent:
            print(f"Generation process failed: {e}")
        return result

def generate_and_reconstruct_batch(
    ar_model: ARModel,
    se_vqvae_model: Any,
    vocab_info: Dict,
    device: str,
    num_samples: int = 100,
    max_length: int = 2048,
    temperature: float = 1,
    top_k: int = 0,
    top_p: float = 0.6,
    repetition_penalty: float = 1,
    no_repeat_ngram_size: int = 0,
    scale_factor: float = 1.0,
    save_step: bool = True,
    output_dir: str = "result/generated_brep",
    filename_prefix: str = "generated",
    write_timeout: int = 30
) -> Dict[str, Any]:
    """
    Batch generation.
    """
    
    # Main statistics
    saved_count = 0
    total_attempts = 0

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with tqdm(total=num_samples, desc="Generating BREP files", unit="item") as pbar:
        while saved_count < num_samples:
            result = generate_and_reconstruct_single(
                ar_model=ar_model,
                se_vqvae_model=se_vqvae_model,
                vocab_info=vocab_info,
                device=device,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                scale_factor=scale_factor,
                save_step=save_step,
                output_dir=output_dir,
                filename_prefix=f"{filename_prefix}_{saved_count:04d}",
                silent=True,
                write_timeout=write_timeout
            )

            total_attempts += 1

            if result.get('stl_saved') and result.get('step_saved'):
                saved_count += 1
                pbar.update(1)

            # Update progress bar description
            pbar.set_description(
                f"Generating BREP files (saved:{saved_count}, attempts:{total_attempts})"
            )

    # Consistency checks
    assert total_attempts >= saved_count, "total_attempts must be greater than or equal to saved_count"

    print("\n=== Generation Summary ===")
    print(f"total_attempts : {total_attempts}")
    print(f"saved_count    : {saved_count} (both STL and STEP saved successfully)")

    return {
        'total_attempts': total_attempts,
        'saved_count': saved_count,
    }

def main():
    
    parser = argparse.ArgumentParser(description="CAD generation")
    
    # Model path arguments
    parser.add_argument("--ar_model", type=str, default="checkpoint/ar/v9/9.9,128,256,1024,8,8/9.9_epoch_500.pt")
    parser.add_argument("--se_vqvae", type=str, default="checkpoint/se/deepcad_se_vqvae_epoch_2680.pt")
    parser.add_argument("--config", type=str, default="config.json", help="Path to the generation config file (JSON)")
    parser.add_argument("--dataset_type", type=str, choices=["abc", "deepcad"], default="deepcad", help="Dataset type (abc or deepcad)")
    
    # Generation arguments
    parser.add_argument("--num_samples", type=int, default=10, help="Number of successfully saved BREP files to generate")
    parser.add_argument("--mode", type=str, default="batch", choices=["single", "batch"], help="Run mode: single for one model, batch for multiple models")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--temperature", type=float, default=1, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling parameter")
    parser.add_argument("--repetition_penalty", type=float, default=1, help="Repetition penalty")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0, help="No-repeat n-gram size")
    parser.add_argument("--scale_factor", type=float, default=1.0, help="Data scaling factor")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="result/generated_brep/final_test", help="Output directory")
    parser.add_argument("--no_save_step", default=False, help="Disable STEP file saving")
    parser.add_argument("--filename_prefix", type=str, default="deepcad", help="Output filename prefix")
    
    # Other arguments
    parser.add_argument("--device", type=str, default='cuda', help="Compute device (cuda/cpu)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID (for example: 0, 1, 2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--write_timeout", type=int, default=30, help="File write timeout in seconds")
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    if args.device == "cuda" and torch.cuda.is_available():
        gpu_index = 0 if args.gpu is None else args.gpu
        if gpu_index < 0 or gpu_index >= torch.cuda.device_count():
            print(f"Requested GPU {args.gpu} is unavailable. Available GPU count: {torch.cuda.device_count()}")
            gpu_index = 0
            print(f"Falling back to default GPU: cuda:{gpu_index}")
        device = torch.device(f"cuda:{gpu_index}")
        torch.cuda.set_device(device)  # Explicitly bind the GPU
        print(f"Using GPU: {device}")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            print("CUDA is unavailable, falling back to CPU")
        device = torch.device("cpu")
        print("Using CPU")

    try:
        print("=== Initializing CAD generation pipeline ===")
        
        # 1. Load config and model
        print("1. Loading configuration...")
        cfg = load_config(args.config)
        print(f"   Using config file: {args.config}")
        
        print("2. Loading model configuration...")
        vocab_info = load_model_config(cfg, args.dataset_type)
        
        print("3. Loading autoregressive model...")
        checkpoint = load_checkpoint(args.ar_model)
        ar_model = init_ar_model(
            vocab_size=vocab_info['vocab_size'],
            pad_token_id=vocab_info['PAD_TOKEN'],
            checkpoint=checkpoint,
            device=device
        )
        load_model_weights(ar_model, checkpoint)
        
        print("4. Loading SE VQ-VAE model...")
        se_vqvae_model = load_se_vqvae_model(args.se_vqvae, False, args.dataset_type, device)
        if se_vqvae_model is None:
            raise RuntimeError(f"Failed to load SE VQ-VAE model. Please check the path: {args.se_vqvae}")
        vocab_info['se_tokens_per_element'] = infer_se_tokens_per_element(se_vqvae_model, device)
        print(f"Detected SE tokens per element: {vocab_info['se_tokens_per_element']}")
            
        print("Initialization complete")

        # 2. Run generation
        if args.mode == "single":
            result = generate_and_reconstruct_single(
                ar_model=ar_model,
                se_vqvae_model=se_vqvae_model,
                vocab_info=vocab_info,
                device=device,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                scale_factor=args.scale_factor,
                save_step=not args.no_save_step,
                output_dir=args.output_dir,
                filename_prefix=args.filename_prefix,
                write_timeout=args.write_timeout
            )
            
            print("\n=== Single Generation Result ===")
            print(f"Sequence length: {len(result['sequence']) if result['sequence'] else 0}")
            if result['saved_path']:
                print(f"Saved: {result['saved_path']}")
            if result['error']:
                print(f"Error: {result['error']}")
            
        else:
            result = generate_and_reconstruct_batch(
                ar_model=ar_model,
                se_vqvae_model=se_vqvae_model,
                vocab_info=vocab_info,
                device=device,
                num_samples=args.num_samples,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                scale_factor=args.scale_factor,
                save_step=not args.no_save_step,
                output_dir=args.output_dir,
                filename_prefix=args.filename_prefix,
                write_timeout=args.write_timeout
            )
            
    except Exception as e:
        print(f"Program execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
