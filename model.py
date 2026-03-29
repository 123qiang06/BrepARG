import os
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config

class ARModel(nn.Module):
    """
    Autoregressive model for CAD generation based on GPT2
    """
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=8, 
                 dim_feedforward=512, dropout=0.1, max_seq_len=1024, pad_token_id=None):
        super(ARModel, self).__init__()
        
        # Create GPT2 configuration
        config_kwargs = dict(
            vocab_size=vocab_size,
            n_positions=max_seq_len,
            n_embd=d_model,
            n_layer=num_layers,
            n_head=nhead,
            n_inner=dim_feedforward,
            activation_function="gelu_new",
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            bos_token_id=0,
            eos_token_id=0,
            loss_type=None
        )
        if pad_token_id is not None:
            config_kwargs['pad_token_id'] = pad_token_id
        self.config = GPT2Config(**config_kwargs)
        
        # Create GPT2 model with the configuration
        self.model = GPT2LMHeadModel(self.config)
        self.transformer = self.model.transformer
        self.lm_head = self.model.lm_head


        # Save parameters for later use
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        rank = int(os.environ.get('RANK', '0')) if 'RANK' in os.environ else 0
        if rank == 0:
            print(f"Initialized ARModel with vocab_size={vocab_size}, d_model={d_model}, "
                  f"layers={num_layers}, heads={nhead}, max_seq_len={max_seq_len}")
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass through the model
        
        Args:
            input_ids (torch.Tensor): Input token IDs [batch_size, seq_len]
            attention_mask (torch.Tensor): Attention mask [batch_size, seq_len]
            labels (torch.Tensor): Labels for language modeling [batch_size, seq_len]
            
        Returns:
            outputs: Model outputs including loss and logits
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
    
    def generate(self, input_ids, attention_mask=None, **kwargs):
        """
        Generate sequences using the model
        
        Args:
            input_ids (torch.Tensor): Input token IDs [batch_size, seq_len]
            attention_mask (torch.Tensor): Attention mask [batch_size, seq_len]
            **kwargs: Additional arguments for generation
            
        Returns:
            torch.Tensor: Generated token IDs
        """
        
        generation_kwargs = {
            'early_stopping': False,
            'do_sample': True,
            'temperature': 0.6,
            'top_p': 0.6,
            'top_k': 0,
            'repetition_penalty': 1,
            'pad_token_id': self.config.pad_token_id,
            'eos_token_id': self.config.eos_token_id,
            'bos_token_id': self.config.bos_token_id,
        }
        
        generation_kwargs.update(kwargs)
        
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_kwargs
        )
    
    def save_pretrained(self, save_directory):
        """
        Save model to directory
        
        Args:
            save_directory (str): Directory to save the model
        """
        self.model.save_pretrained(save_directory)
        
    def load_state_dict(self, state_dict, strict=True):
        """
        Load state dictionary
        
        Args:
            state_dict (dict): State dictionary
            strict (bool): Whether to strictly enforce that the keys match
        """
        # Check if the state_dict is from a GPT2LMHeadModel directly
        if 'transformer.wte.weight' in state_dict:
            # It's a GPT2LMHeadModel state_dict
            return self.model.load_state_dict(state_dict, strict=strict)
        
        # Check if it's our ARModel state_dict
        if 'model.transformer.wte.weight' in state_dict:
            return super().load_state_dict(state_dict, strict=strict)
        
        # If it's neither, try to adapt the state_dict
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                new_state_dict[key] = value
            else:
                new_state_dict[f'model.{key}'] = value
        
        return super().load_state_dict(new_state_dict, strict=strict)