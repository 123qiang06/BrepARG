import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import VQModel
import time
import math
from model import ARModel
from torch.utils.data import DataLoader
from quantise import VectorQuantiser
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from model import ARModel  # Ensure your ARModel path is correct

class VQVAE(VQModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        old_quant = self.quantize
        self.quantize = VectorQuantiser(
            num_embed=old_quant.n_e,
            embed_dim=old_quant.vq_embed_dim,
            beta=old_quant.beta,
            distance='cos',
            anchor='probrandom',
            first_batch=False, 
            contras_loss=True
        )
        self.quantize.embedding.weight.data.copy_(old_quant.embedding.weight.data)

class VQVAETrainer():
    "Surface and Edge VQ-VAE Trainer"

    def __init__(self, args, train_dataset, val_dataset, multi_gpu=False):
        self.args = args
        self.iters = 0
        self.epoch = 1
        self.save_dir = args.save_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.multi_gpu = multi_gpu
        
        # Add best model tracking
        self.best_val_loss = float('inf')
        self.best_path = None

        # Ensure save/log directories exist, and distinguish log directories for multi-process
        os.makedirs(self.save_dir, exist_ok=True)
        base_tb_dir = args.tb_log_dir
        rank_env = int(os.environ.get('RANK', '0')) if 'RANK' in os.environ else 0
        self.rank = rank_env
        tb_dir = base_tb_dir if rank_env == 0 else f"{base_tb_dir}_rank{rank_env}"
        os.makedirs(tb_dir, exist_ok=True)
        # Initialize TensorBoard writer (different ranks write to different directories to avoid conflicts)
        self.writer = SummaryWriter(log_dir=tb_dir)

        # Initialize data loader configuration
        num_workers = 4
        effective_batch_size = args.batch_size
        
        if self.multi_gpu and self.rank == 0:
            print(f"Data loader configuration:")
            print(f"  Batch size per GPU: {effective_batch_size}")
            print(f"  Num workers: {num_workers}")
        
        # Create distributed sampler for DDP
        train_sampler = None
        val_sampler = None
        train_shuffle = True
        
        if self.multi_gpu and 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            # DDP mode - use distributed sampler
            from torch.utils.data.distributed import DistributedSampler
            
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=int(os.environ['WORLD_SIZE']),
                rank=int(os.environ['RANK']),
                shuffle=True
            )
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=int(os.environ['WORLD_SIZE']),
                rank=int(os.environ['RANK']),
                shuffle=False
            )
            train_shuffle = False  # Cannot use shuffle when using sampler
            if self.rank == 0:
                print(f"Using DistributedSampler, World Size: {os.environ['WORLD_SIZE']}, Rank: {os.environ['RANK']}")
        
        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset, 
            shuffle=train_shuffle, 
            batch_size=effective_batch_size, 
            sampler=train_sampler,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()  # Enable pin_memory to accelerate GPU transfer
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            val_dataset, 
            shuffle=False, 
            batch_size=effective_batch_size, 
            sampler=val_sampler,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        # Save sampler reference to reset random seed between epochs
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler

        # Determine input channels based on whether to use type flag
        self.use_type_flag = args.use_type_flag
        in_channels = 4 if self.use_type_flag else 3
        out_channels = 3  # Output is always 3-channel coordinates

        self.model = VQVAE(
            in_channels=in_channels,  # Set input channels based on flag
            out_channels=out_channels,
            down_block_types=['DownEncoderBlock2D'] * 5,
            up_block_types=['UpDecoderBlock2D'] * 5,
            block_out_channels=[32, 64, 128, 256, 512],
            layers_per_block=2,
            act_fn='silu',
            latent_channels=128,
            vq_embed_dim=64,
            num_vq_embeddings=4096,
            norm_num_groups=32,
            sample_size=512,
        ).to(self.device)
        
        # Multi-GPU support - DDP mode
        if self.multi_gpu and 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            # DDP mode - use distributed training
            import torch.distributed as dist
            from torch.nn.parallel import DistributedDataParallel as DDP
            
            # Initialize distributed process group
            if not dist.is_initialized():
                # Windows/CPU fallback to gloo; CUDA + non-Windows use nccl
                backend = 'nccl' if (torch.cuda.is_available() and os.name != 'nt') else 'gloo'
                dist.init_process_group(backend=backend)
            
            # Get distributed information
            local_rank = int(os.environ['LOCAL_RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            rank = int(os.environ['RANK'])
            
            # Set device
            torch.cuda.set_device(local_rank)
            self.device = torch.device(f'cuda:{local_rank}')
            
            # Move model to corresponding GPU and wrap as DDP
            self.model = self.model.to(self.device)
            self.model = DDP(self.model, device_ids=[local_rank])
            
            if rank == 0:
                print(f"DDP training initialization:")
                print(f"  Total processes: {world_size}")
                print(f"  Current device: {self.device}")
                gpu_name = torch.cuda.get_device_name(local_rank)
                gpu_memory = torch.cuda.get_device_properties(local_rank).total_memory / 1024**3
                print(f"  GPU: {gpu_name} ({gpu_memory:.1f}GB)")
                
            # Optimize training settings
            if hasattr(torch.backends.cudnn, 'benchmark'):
                torch.backends.cudnn.benchmark = True
        else:
            # Single GPU training mode
            if self.rank == 0:
                print(f"Single GPU training mode")
                print(f"  Device: {self.device}")
        
        # Use hasattr to check if model is wrapped by DDP/DataParallel, rather than relying on multi_gpu flag
        self.codebook_size = self.model.module.quantize.embedding.num_embeddings if hasattr(self.model, 'module') else self.model.quantize.embedding.num_embeddings
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=1e-6,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        # AMP only enabled when CUDA is available, improving cross-platform stability
        self.amp_enabled = torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

        if args.weight and os.path.exists(args.weight):
            self._load_checkpoint(args.weight)
        else:
            if args.weight:
                print(f"Specified weight path does not exist: {args.weight}")
            print("Training from scratch")

    def train_one_epoch(self):
        self.model.train()
        start_time = time.time()
        progress_bar = tqdm(total=len(self.train_dataloader), disable=(self.rank != 0))
        progress_bar.set_description(f"Epoch {self.epoch}")
        # Set epoch for DDP sampler to change shuffle
        if self.train_sampler is not None and hasattr(self.train_sampler, 'set_epoch'):
            self.train_sampler.set_epoch(self.epoch)
        if self.val_sampler is not None and hasattr(self.val_sampler, 'set_epoch'):
            self.val_sampler.set_epoch(self.epoch)
        
        # For accumulating statistics of this epoch
        epoch_recon_loss = 0.0
        epoch_vq_loss = 0.0
        epoch_total_loss = 0.0
        batch_count = 0
        total_usage_rate = 0.0
        total_perplexity = 0.0

        # Zero gradients outside the loop
        self.optimizer.zero_grad()
        

        for batch_idx, batch_data in enumerate(self.train_dataloader):
            try:
                # Move data to device and adjust channel order (B, 4, 4, 3/4) -> (B, 3/4, 4, 4)
                batch_data = batch_data.to(self.device, non_blocking=True).permute(0, 3, 1, 2)
                
                # Forward pass
                with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                    # Get model (handle DataParallel)
                    model = self.model.module if hasattr(self.model, 'module') else self.model
                    
                    # Encoding stage
                    h = model.encoder(batch_data)
                    h = model.quant_conv(h)
                    
                    # Quantization stage (get all information at once)
                    quant_out, vq_loss, indices = model.quantize(h)
                    # Get perplexity and codebook usage rate
                    if isinstance(indices, tuple) and len(indices) > 0:
                        perplexity = indices[0].item() if hasattr(indices[0], 'item') else float(indices[0])
                        encoding_indices = indices[2] if len(indices) > 2 else indices[1]
                        used_codes = torch.unique(encoding_indices).numel()
                        usage_rate = used_codes / self.codebook_size
                    else:
                        perplexity = 0.0
                        usage_rate = 0.0
                    # Decoding stage
                    recon = model.decoder(model.post_quant_conv(quant_out))
                    
                    # Calculate reconstruction loss
                    if self.use_type_flag and batch_data.shape[1] == 4:
                        # If input contains type flag, only calculate loss for first 3 channels (coordinates)
                        coords_input = batch_data[:, :3, :, :]  # Take first 3 channels
                        recon_loss = nn.functional.mse_loss(recon, coords_input)
                    else:
                        # If input does not contain type flag, calculate loss directly
                        recon_loss = nn.functional.mse_loss(recon, batch_data)
                
                # Total loss
                total_loss = recon_loss + vq_loss
                
                # Check if loss is valid
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"Warning: batch {batch_idx} detected invalid loss value, skipping this batch")
                    continue
                
                # Backward pass
                self.scaler.scale(total_loss).backward()
                
                # Gradient clipping to prevent gradient explosion
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                # Update statistics
                epoch_total_loss += total_loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_vq_loss += vq_loss.item()
                total_usage_rate += usage_rate
                total_perplexity += perplexity
                batch_count += 1
                
                # Update training iteration count
                self.iters += 1
                
                # Update progress bar
                if self.rank == 0:
                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        'loss': f"{epoch_total_loss/batch_count:.8f}", 
                        'recon': f"{epoch_recon_loss/batch_count:.8f}", 
                        'vq': f"{epoch_vq_loss/batch_count:.8f}",
                        'usage': f"{total_usage_rate/batch_count:.2f}",
                        'perp': f"{total_perplexity/batch_count:.2f}"
                    })
                
            except Exception as e:
                print(f"Warning: batch {batch_idx} processing failed: {e}")
                continue
        
        progress_bar.close()
        
        # Calculate and print statistics at end of epoch
        avg_recon_loss = epoch_recon_loss / batch_count if batch_count > 0 else 0
        avg_vq_loss = epoch_vq_loss / batch_count if batch_count > 0 else 0
        avg_total_loss = epoch_total_loss / batch_count if batch_count > 0 else 0
        avg_usage_rate = total_usage_rate / batch_count if batch_count > 0 else 0
        avg_perplexity = total_perplexity / batch_count if batch_count > 0 else 0
        epoch_time = time.time() - start_time

        # Synchronize with other DDP ranks to ensure all processes complete this epoch before validation/saving
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
        except Exception:
            pass
        
        if self.rank == 0:
            print(f"Epoch [{self.epoch}/{self.args.train_epoch}] Time: {epoch_time:.2f}s")
            print(f"  Loss - Recon: {avg_recon_loss:.8f} | VQ: {avg_vq_loss:.8f} | Total: {avg_total_loss:.8f}")
            print(f"  Metrics - Usage: {avg_usage_rate:.2f} | Perplexity: {avg_perplexity:.2f}")
        
        # Record to TensorBoard (only on rank 0)
        if 'RANK' not in os.environ or int(os.environ['RANK']) == 0:
            self.writer.add_scalar('train/total_loss', avg_total_loss, self.epoch)
            self.writer.add_scalar('train/recon_loss', avg_recon_loss, self.epoch)
            self.writer.add_scalar('train/vq_loss', avg_vq_loss, self.epoch)
            self.writer.add_scalar('train/codebook_usage', avg_usage_rate, self.epoch)
            self.writer.add_scalar('train/perplexity', avg_perplexity, self.epoch)
            self.writer.add_scalar('train/learning_rate', self.optimizer.param_groups[0]['lr'], self.epoch)
        self.epoch += 1

    def test_val(self):
        # Only execute validation on rank 0 to avoid DDP multi-process duplicate validation/printing/saving
        rank_env = int(os.environ.get('RANK', '0')) if 'RANK' in os.environ else 0
        if rank_env != 0:
            return 0.0
        self.model.eval()
        # Statistics
        total_recon_loss = 0.0
        total_vq_loss = 0.0
        total_loss = 0.0
        batch_count = 0
        
        with torch.no_grad():
            # Handle DataParallel/DDP-safe base model access
            model_to_use = self.model.module if hasattr(self.model, 'module') else self.model
            all_code_counts = torch.zeros(self.codebook_size, device=self.device)
            perplexity_sum = 0.0
            perplexity_count = 0
            for surf_uv in self.val_dataloader:
                surf_uv = surf_uv.to(self.device).permute(0, 3, 1, 2)  # (B, C, 32, 32)
                
                # Encoding stage
                h = model_to_use.encoder(surf_uv)
                h = model_to_use.quant_conv(h)
                
                # Quantization stage
                quant_out, vq_loss, indices = model_to_use.quantize(h)
                
                # Decoding stage
                recon = model_to_use.decoder(model_to_use.post_quant_conv(quant_out))
                
                # Calculate reconstruction loss
                if self.use_type_flag:
                    # If using type flag, only calculate loss for first 3 channels
                    coords = surf_uv[:, :3, :, :]  # (B, 3, 32, 32)
                else:
                    # If not using type flag, calculate loss for all input channels
                    coords = surf_uv  # (B, 3, 32, 32)

                # Calculate reconstruction loss
                recon_loss = F.mse_loss(recon, coords, reduction='mean')
                loss = recon_loss + vq_loss
                
                # Calculate codebook usage count (accumulate to all_code_counts)
                indices_tensor = indices[2] if len(indices) > 2 else indices[1]
                flat_indices = indices_tensor.view(-1)
                code_counts = torch.bincount(flat_indices, minlength=self.codebook_size).float()
                all_code_counts += code_counts

                # Accumulate statistics
                total_recon_loss += recon_loss.item()
                total_vq_loss += vq_loss.item()
                total_loss += loss.item()
                batch_count += 1
                # Get perplexity
                if isinstance(indices, tuple) and len(indices) > 0:
                    perplexity = indices[0].item() if hasattr(indices[0], 'item') else float(indices[0])
                else:
                    perplexity = 0.0
                perplexity_sum += perplexity
                perplexity_count += 1

        # Calculate average loss
        avg_recon_loss = total_recon_loss / batch_count if batch_count > 0 else 0
        avg_vq_loss = total_vq_loss / batch_count if batch_count > 0 else 0
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        # Statistics of codebook activity across entire validation set
        avg_usage = (all_code_counts > 0).sum().item() / self.codebook_size
        avg_perplexity = perplexity_sum / perplexity_count if perplexity_count > 0 else 0.0

        # Print validation results
        print(f"Validation loss: Total {avg_loss:.6f} = Reconstruction {avg_recon_loss:.6f} + VQ {avg_vq_loss:.6f} | Codebook activity: {avg_usage:.6f} | Perplexity: {avg_perplexity:.2f}")

        # Check if it's the best model
        # Use epoch-1 because train_one_epoch has already incremented epoch
        val_epoch = self.epoch - 1
        is_best = False
        if avg_recon_loss < self.best_val_loss:
            self.best_val_loss = avg_recon_loss
            is_best = True
            # Update best model filename
            self.best_path = f'{self.args.dataset_type}_se_vqvae_epoch_{val_epoch}.pt'
            print(f"New best model! Validation loss: {avg_recon_loss:.6f}")
            print(f"Best model filename: {self.best_path}")
        else:
            print(f"Current best validation loss: {self.best_val_loss:.6f}")
            print(f"Current best model filename: {self.best_path}")

        # Record to TensorBoard (only rank 0)
        if rank_env == 0:
            self.writer.add_scalar('val/total_loss', avg_loss, val_epoch)
            self.writer.add_scalar('val/recon_loss', avg_recon_loss, val_epoch)
            self.writer.add_scalar('val/vq_loss', avg_vq_loss, val_epoch)
            self.writer.add_scalar('val/codebook_usage', avg_usage, val_epoch)
            self.writer.add_scalar('val/perplexity', avg_perplexity, val_epoch)
            self.writer.add_scalar('val/best_loss', self.best_val_loss, val_epoch)

        # If it's the best model, save automatically
        if is_best:
            self.save_model(is_best=True, save_epoch=val_epoch)

        return avg_loss

    def save_model(self, is_best=False, save_epoch=None):
        # Only save model on rank 0 in DDP mode
        if 'RANK' in os.environ and int(os.environ['RANK']) != 0:
            return
        
        # If epoch to save is not specified, use current epoch-1 (because train_one_epoch has already incremented)
        if save_epoch is None:
            save_epoch = self.epoch - 1
        
        # Get base model in multi-card mode
        if hasattr(self.model, 'module'):  # DataParallel or DDP
            model_to_save = self.model.module
        else:
            model_to_save = self.model
        
        # Only save necessary hyperparameters
        args_to_save = {k: v for k, v in vars(self.args).items() if isinstance(v, (int, float, str, bool, type(None)))}
        checkpoint = {
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if hasattr(self, 'scaler') else None,
            'epoch': save_epoch + 1,  # Save next epoch to train (continue training directly after loading)
            'completed_epoch': save_epoch,  # Save completed epoch (for display)
            'iters': self.iters,  # Add training iteration count
            'best_val_loss': self.best_val_loss,
            'best_path': self.best_path,  # Save best model filename
            'current_learning_rate': self.optimizer.param_groups[0]['lr'],  # Add current learning rate
            'multi_gpu': self.multi_gpu,  # Save multi-GPU information
            'args': args_to_save,
        }

        # Save regular checkpoint
        save_path = os.path.join(self.save_dir, f'{self.args.dataset_type}_se_vqvae_epoch_{save_epoch}.pt')
        torch.save(checkpoint, save_path)
        print(f"Model saved to: {save_path}")
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dir, f'{self.args.dataset_type}_se_vqvae_best.pt')
            torch.save(checkpoint, best_path)
            print(f"Best model saved to: {best_path}")


    def _load_checkpoint(self, checkpoint_path):
        """Selectively load based on what is actually saved in the checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model weights
            if 'model_state_dict' in checkpoint:
                try:
                    # Get current model state dictionary
                    current_model = self.model.module if hasattr(self.model, 'module') else self.model
                    
                    # Handle state dictionary key name mismatch (DataParallel)
                    state_dict = checkpoint['model_state_dict']
                    
                    # If checkpoint was saved by DataParallel/DDP but current is not, remove 'module.' prefix
                    if not hasattr(self.model, 'module') and any(k.startswith('module.') for k in state_dict.keys()):
                        state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
                    
                    # If checkpoint was not saved by DataParallel/DDP but current is, add 'module.' prefix
                    elif hasattr(self.model, 'module') and not any(k.startswith('module.') for k in state_dict.keys()):
                        state_dict = {f'module.{k}': v for k, v in state_dict.items()}
                    
                    # Try strict loading first
                    missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                    if missing_keys:
                        print(f"Model loading warning - missing keys: {missing_keys}")
                    if unexpected_keys:
                        print(f"Model loading warning - unexpected keys: {unexpected_keys}")
                    if not missing_keys and not unexpected_keys:
                        print(f"Complete model state loaded: {checkpoint_path}")
                    else:
                        print(f"Partial model state loaded: {checkpoint_path}")
                except Exception as e:
                    print(f"Model state loading failed, trying compatibility mode: {e}")
                    # Compatibility mode: only load matching weights
                    model_dict = self.model.state_dict()
                    pretrained_dict = checkpoint['model_state_dict']
                    
                    # Check shape matching
                    matched_dict = {}
                    shape_mismatches = []
                    for k, v in pretrained_dict.items():
                        if k in model_dict:
                            if v.shape == model_dict[k].shape:
                                matched_dict[k] = v
                            else:
                                shape_mismatches.append(f"{k}: {v.shape} vs {model_dict[k].shape}")
                        else:
                            print(f"Skipping non-existent key: {k}")
                    
                    if shape_mismatches:
                        print(f"Shape mismatch keys: {shape_mismatches}")
                    
                    model_dict.update(matched_dict)
                    self.model.load_state_dict(model_dict)
                    print(f"Compatibility mode loading successful, loaded {len(matched_dict)}/{len(model_dict)} layers")
            else:
                print(f"model_state_dict not found, skipping model weight loading")
            
            # Load optimizer state
            if 'optimizer_state_dict' in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    
                    # Move optimizer state to correct device
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(self.device)
                    
                    print(f"Optimizer state loaded")
                except Exception as e:
                    print(f"Optimizer state loading failed: {e}")
            else:
                print(f"optimizer_state_dict not found, skipping optimizer state loading")
            
            # Load scaler state
            if hasattr(self, 'scaler') and 'scaler_state_dict' in checkpoint and self.scaler is not None:
                try:
                    self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                    print(f"Scaler state loaded")
                except Exception as e:
                    print(f"Scaler state loading failed: {e}")
            else:
                print(f"scaler_state_dict not found, skipping scaler state loading")
            
            # Load training progress
            if 'epoch' in checkpoint:
                self.epoch = checkpoint['epoch']
                completed_epoch = checkpoint.get('completed_epoch', self.epoch - 1)
                print(f"Completed training epoch: {completed_epoch}, will continue from epoch {self.epoch}")
            else:
                print(f"epoch not found, skipping epoch loading")
            
            # Load training iteration count
            if 'iters' in checkpoint:
                self.iters = checkpoint['iters']
                print(f"Restored training iteration count: {self.iters}")
            else:
                print(f"iters not found, using default value: {self.iters}")
            
            # Load best validation loss
            if 'best_val_loss' in checkpoint:
                self.best_val_loss = checkpoint['best_val_loss']
                print(f"Restored best validation loss: {self.best_val_loss:.6f}")
            else:
                print(f"best_val_loss not found, using default value")
            
            # Load best model filename
            if 'best_path' in checkpoint:
                self.best_path = checkpoint['best_path']
                print(f"Restored best model filename: {self.best_path}")
            else:
                print(f"best_path not found, using default value")
            
            # # Load learning rate state (ensure optimizer learning rate is correct)
            # if 'current_learning_rate' in checkpoint:
            #     current_lr = checkpoint['current_learning_rate']
            #     # Update optimizer learning rate (ensure consistency)
            #     for param_group in self.optimizer.param_groups:
            #         param_group['lr'] = current_lr
            #     print(f"Restored learning rate: {current_lr:.2e}")
            # else:
            #     current_lr = self.optimizer.param_groups[0]['lr']
            #     print(f"current_learning_rate not found, using optimizer current learning rate: {current_lr:.2e}")
            
            print(f"Checkpoint loading completed")
            
        except Exception as e:
            print(f"Checkpoint loading failed: {e}")
            print("Training from scratch")
        
        # Ensure model is on correct device
        self.model.to(self.device)

    def close_writer(self):
        self.writer.close()

class ARTrainer:
    """Autoregressive CAD generation model trainer"""

    def __init__(self, train_dataset, val_dataset, args, device=None, multi_gpu=False):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # === Parameters passed through args ===
        self.batch_size = args.batch_size
        self.train_epoch = args.train_epoch
        self.test_epoch = args.test_epoch
        self.save_epoch = args.save_epoch
        self.dataset_type = args.dataset_type
        self.save_dir = args.save_dir
        self.tb_log_dir = args.tb_log_dir
        self.weight_path = args.weight
        self.max_seq_len = args.max_seq_len
        self.args = args
        self.multi_gpu = multi_gpu
        self.rank = int(os.environ.get("RANK", 0))

        # === Training parameters ===
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.dropout = args.dropout
        self.label_smoothing = args.label_smoothing

        # === Model architecture parameters ===
        self.d_model = args.d_model
        self.nhead = args.nhead
        self.num_layers = args.num_layers
        self.dim_feedforward = args.dim_feedforward

        # === Get token configuration from dataset ===
        self.vocab_size = train_dataset.vocab_size
        self.special_token_size = train_dataset.special_token_size
        self.face_index_size = train_dataset.face_index_size
        self.codebook_size = train_dataset.codebook_size
        self.face_index_offset = train_dataset.face_index_offset
        self.se_token_offset = train_dataset.se_token_offset
        self.bbox_token_offset = train_dataset.bbox_token_offset
        self.se_tokens_per_element = train_dataset.se_tokens_per_element
        self.bbox_tokens_per_element = train_dataset.bbox_tokens_per_element

        # Special tokens
        self.START_TOKEN = train_dataset.START_TOKEN
        self.SEP_TOKEN = train_dataset.SEP_TOKEN
        self.END_TOKEN = train_dataset.END_TOKEN
        self.PAD_TOKEN = train_dataset.PAD_TOKEN

        # Training state
        self.epoch = 1
        self.global_step = 0
        self.iters = 0
        self.best_loss = float('inf')
        self.best_path = None

        # Device
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Directories/TB
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.tb_log_dir, exist_ok=True)
        # Only write TensorBoard on rank0 to avoid multi-process conflicts
        self.writer = SummaryWriter(log_dir=self.tb_log_dir) if self.rank == 0 else None

        # === DataLoader & Sampler ===
        if self.multi_gpu and torch.cuda.device_count() > 1 and dist.is_available() and dist.is_initialized():
            num_workers = min(8, torch.cuda.device_count() * 2)
            effective_batch_size = self.batch_size
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=True
            )
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=False
            )
            train_shuffle = False
            if self.rank == 0:
                print(f"Using DistributedSampler, World Size: {dist.get_world_size()}, Rank: {dist.get_rank()}")
                print(f"Batch size per GPU: {effective_batch_size} | num_workers: {num_workers}")
        else:
            num_workers = 2
            effective_batch_size = self.batch_size
            train_sampler = None
            val_sampler = None
            train_shuffle = True

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=effective_batch_size,
            shuffle=train_shuffle,
            sampler=train_sampler,
            drop_last=True,
            num_workers=num_workers,
            collate_fn=train_dataset.collate_fn,
            pin_memory=torch.cuda.is_available()
        )

        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=effective_batch_size,
            shuffle=False,
            sampler=val_sampler,
            drop_last=False,
            num_workers=num_workers,
            collate_fn=val_dataset.collate_fn,
            pin_memory=torch.cuda.is_available()
        )

        self.train_sampler = train_sampler
        self.val_sampler = val_sampler

        # Initialize model & optimizer
        self._init_model()
        self._init_optimizer()

        # Checkpoint
        if self.weight_path and os.path.exists(self.weight_path):
            self._load_checkpoint(self.weight_path)

        # AMP (disabled on CPU)
        self.amp_enabled = torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

        if self.rank == 0:
            print(f"Training configuration: will train for {self.train_epoch} complete epochs")

    # ----------------- Model / Optimizer -----------------

    def _init_model(self):
        self.model = ARModel(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            max_seq_len=self.max_seq_len,
            pad_token_id=self.PAD_TOKEN
        ).to(self.device)

        # Only wrap when DDP is initialized externally; do not initialize process group here & do not use DataParallel
        if self.multi_gpu and dist.is_available() and dist.is_initialized():
            from torch.nn.parallel import DistributedDataParallel as DDP
            local_rank = int(os.environ['LOCAL_RANK'])
            self.model = DDP(self.model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

        if self.rank == 0:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"Model initialized - d_model: {self.d_model}, layers: {self.num_layers}, heads: {self.nhead}")
            print(f"  Parameter count: {total_params:,} total, {trainable_params:,} trainable")
            model_size_mb = total_params * 4 / 1024 / 1024
            print(f"  Model memory: {model_size_mb:.1f}MB")

    def _init_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=self.weight_decay,
            eps=1e-8
        )
        if self.rank == 0:
            print("Optimizer configuration:")
            print(f"  Learning rate: {self.learning_rate}")
            print(f"  Weight decay: {self.weight_decay}")
            print(f"  Dropout: {self.dropout}")
            print(f"  Label smoothing: {self.label_smoothing}")

    # ----------------- Train / Validate -----------------

    def train(self):
        if self.rank == 0:
            print(f"Starting training - from epoch {self.epoch} to epoch {self.train_epoch}")
            print(f"Dataset: {len(self.train_dataloader.dataset)} training samples, {len(self.val_dataloader.dataset)} validation samples")

        try:
            for epoch in range(self.epoch, self.train_epoch + 1):
                self.epoch = epoch
                if self.rank == 0:
                    print(f"\n=== Epoch {self.epoch}/{self.train_epoch} ===")

                # DDP requires setting sampler epoch for each epoch
                if self.train_sampler is not None:
                    self.train_sampler.set_epoch(epoch)
                if self.val_sampler is not None:
                    self.val_sampler.set_epoch(epoch)

                _ = self.train_one_epoch()

                # Validation and saving
                if epoch % self.test_epoch == 0:
                    val_loss, _ = self.validate()  # val_loss is already global average
                    is_best = val_loss < self.best_loss
                    if is_best:
                        self.best_loss = val_loss  # Sync to member variable for saving and restoring
                        if self.rank == 0:
                            print(f"New best model! Validation loss: {val_loss:.6f}")
                    else:
                        if self.rank == 0:
                            print(f"Validation loss: {val_loss:.6f}")

                    if epoch % self.save_epoch == 0 or is_best:
                        self.save_checkpoint(is_best=is_best)

                elif epoch % self.save_epoch == 0:
                    self.save_checkpoint()

        except KeyboardInterrupt:
            print("Training interrupted by user")
        except Exception as e:
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()

        if self.rank == 0:
            print(f"Training completed! Completed all {self.train_epoch} epochs")
            print(f"Best validation loss: {self.best_loss:.6f}")
            print(f"Model saved to: {self.save_dir}")

    @torch.no_grad()
    def _all_reduce_sum(self, tensor):
        """Perform global sum on tensor (under DDP); non-DDP returns itself directly."""
        if self.multi_gpu and dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    def train_one_epoch(self):
        self.model.train()
        start_time = time.time()

        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.epoch} Training", disable=(self.rank != 0))

        # Accumulate "weighted loss" and "valid token count" for global averaging
        global_loss_num = torch.tensor(0.0, device=self.device)
        global_loss_den = torch.tensor(0.0, device=self.device)

        for _, batch in enumerate(progress_bar):
            if self.multi_gpu and torch.cuda.is_available():
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            else:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                labels = input_ids.clone()
                labels[labels == self.PAD_TOKEN] = -100

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                ce_loss = outputs.loss  # Already mean over tokens-of-batch (HF default)
                
                # If label smoothing is enabled or model did not return loss, calculate manually
                if ce_loss is None or self.label_smoothing > 0:
                    logits = outputs.logits
                    loss_fct = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=self.label_smoothing)
                    ce_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

            self.scaler.scale(ce_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.global_step += 1
            self.iters += 1

            # Use "valid token count" as weight for weighted average, convenient for DDP global reduction
            with torch.no_grad():
                valid_tokens = (labels != -100).sum().to(torch.float32)
                global_loss_num += ce_loss.detach() * valid_tokens
                global_loss_den += valid_tokens

            if self.rank == 0:
                # Only show local process rolling average (for display), final TB write uses global average
                progress_bar.set_postfix({
                    'CE Loss(local)': f'{ce_loss.item():.4f}',
                    'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })

        # ---- Calculate global average loss and write to TB ----
        global_loss_num = self._all_reduce_sum(global_loss_num)
        global_loss_den = self._all_reduce_sum(global_loss_den.clamp(min=1.0))
        global_avg_ce = (global_loss_num / global_loss_den).item()

        if self.rank == 0 and self.writer is not None:
            self.writer.add_scalar('Train/CE_Loss', global_avg_ce, self.epoch)
            self.writer.add_scalar('Train/Learning_Rate', self.optimizer.param_groups[0]['lr'], self.epoch)

        _ = time.time() - start_time
        return global_avg_ce

    def validate(self):
        self.model.eval()

        val_bar = tqdm(self.val_dataloader, desc="Validating", disable=(self.rank != 0))

        global_loss_num = torch.tensor(0.0, device=self.device)
        global_loss_den = torch.tensor(0.0, device=self.device)

        with torch.no_grad():
            for batch in val_bar:
                if self.multi_gpu and torch.cuda.is_available():
                    input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                    attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                else:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)

                with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                    labels = input_ids.clone()
                    labels[labels == self.PAD_TOKEN] = -100

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    ce_loss = outputs.loss
                    if ce_loss is None:
                        logits = outputs.logits
                        loss_fct = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=self.label_smoothing)
                        ce_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

                valid_tokens = (labels != -100).sum().to(torch.float32)
                global_loss_num += ce_loss.detach() * valid_tokens
                global_loss_den += valid_tokens

                if self.rank == 0:
                    val_bar.set_postfix({'CE(local)': f'{ce_loss.item():.4f}'})

        # Global reduction
        global_loss_num = self._all_reduce_sum(global_loss_num)
        global_loss_den = self._all_reduce_sum(global_loss_den.clamp(min=1.0))
        global_avg_ce = (global_loss_num / global_loss_den).item()
        perplexity = math.exp(min(global_avg_ce, 20.0))

        if self.rank == 0:
            print(f"Validation - CE Loss: {global_avg_ce:.6f}, Perplexity: {perplexity:.2f}")
            if self.writer is not None:
                self.writer.add_scalar('Val/CE_Loss', global_avg_ce, self.epoch)
                self.writer.add_scalar('Val/Perplexity', perplexity, self.epoch)

        return global_avg_ce, perplexity

    # ----------------- Checkpoint -----------------

    def save_checkpoint(self, is_best=False):
        # Only save on rank0
        if 'RANK' in os.environ and int(os.environ['RANK']) != 0:
            return

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model

        checkpoint = {
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if hasattr(self, 'scaler') else None,
            'epoch': self.epoch,
            'iters': self.iters,
            'best_loss': self.best_loss,
            'best_path': self.best_path,
            'current_learning_rate': self.optimizer.param_groups[0]['lr'],
            'args': self.args,
            'multi_gpu': self.multi_gpu and dist.is_available() and dist.is_initialized()
        }

        save_path = os.path.join(self.save_dir, f'epoch_{self.epoch}.pt')
        torch.save(checkpoint, save_path)
        print(f"Model saved to: {save_path}")

        if is_best:
            best_path = os.path.join(self.save_dir, f'{self.args.dataset_type}_ar_vqvae_best_model.pt')
            torch.save(checkpoint, best_path)
            self.best_path = best_path  # Sync best_path for restoration
            print(f"Best model saved to: {best_path}")

            # Also save HF weights (for direct loading)
            try:
                best_hf_path = os.path.join(self.save_dir, f'{self.args.dataset_type}_ar_vqvae_best_model_hf')
                model_to_save.save_pretrained(best_hf_path)
                print(f"Saved HuggingFace model: {best_hf_path}")
            except Exception as e:
                print(f"Failed to save HuggingFace model: {e}")

    def _load_checkpoint(self, checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            checkpoint_multi_gpu = checkpoint.get('multi_gpu', False)

            # Load weights
            if 'model_state_dict' in checkpoint:
                model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
                try:
                    if checkpoint_multi_gpu and not (self.multi_gpu and dist.is_available() and dist.is_initialized()):
                        # Multi-card weights loaded to single card: remove 'module.' prefix
                        new_state = {k[7:] if k.startswith('module.') else k: v
                                     for k, v in checkpoint['model_state_dict'].items()}
                        model_to_load.load_state_dict(new_state, strict=False)
                    else:
                        model_to_load.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    print(f"Model state loaded: {checkpoint_path}")
                except Exception as e:
                    print(f"Model state loading failed, trying compatibility mode: {e}")
                    model_dict = model_to_load.state_dict()
                    pretrained = {k: v for k, v in checkpoint['model_state_dict'].items()
                                  if k in model_dict and v.shape == model_dict[k].shape}
                    model_dict.update(pretrained)
                    model_to_load.load_state_dict(model_dict)
                    print(f"Compatibility mode loading successful, loaded {len(pretrained)}/{len(model_dict)} layers")
            else:
                print("model_state_dict not found, skipping model weight loading")

            # Load optimizer
            if 'optimizer_state_dict' in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(self.device)
                    print("Optimizer state loaded")
                except Exception as e:
                    print(f"Optimizer state loading failed: {e}")
            
            if checkpoint.get('scaler_state_dict') is not None and hasattr(self, 'scaler') and self.scaler is not None:
                try:
                    self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                    print("AMP GradScaler state restored")
                except Exception as e:
                    print(f"AMP GradScaler restoration failed, ignoring: {e}")

            # Training progress
            if 'epoch' in checkpoint:
                self.epoch = checkpoint['epoch']
                print(f"Restored training epoch: {self.epoch}")
            if 'iters' in checkpoint:
                self.iters = checkpoint['iters']
                print(f"Restored training iteration count: {self.iters}")

            # Best metrics/path
            if 'best_loss' in checkpoint:
                self.best_loss = checkpoint['best_loss']
                print(f"Restored best validation loss: {self.best_loss:.6f}")
            if 'best_path' in checkpoint:
                self.best_path = checkpoint['best_path']
                print(f"Restored best model filename: {self.best_path}")

            # Learning rate
            if 'current_learning_rate' in checkpoint:
                current_lr = checkpoint['current_learning_rate']
                for pg in self.optimizer.param_groups:
                    pg['lr'] = current_lr
                print(f"Restored learning rate: {current_lr:.2e}")

            # Other state
            if 'learning_rate' in checkpoint:
                self.learning_rate = checkpoint['learning_rate']
                for pg in self.optimizer.param_groups:
                    pg['lr'] = self.learning_rate
            if 'label_smoothing' in checkpoint:
                self.label_smoothing = checkpoint['label_smoothing']
            if 'se_tokens_per_element' in checkpoint:
                self.se_tokens_per_element = checkpoint['se_tokens_per_element']
            if 'bbox_tokens_per_element' in checkpoint:
                self.bbox_tokens_per_element = checkpoint['bbox_tokens_per_element']

            print(f"Successfully loaded checkpoint, continuing training from epoch {self.epoch}")
            
        except Exception as e:
            print(f"Checkpoint loading failed: {e}")
            print("Training from scratch")

        self.model.to(self.device)
