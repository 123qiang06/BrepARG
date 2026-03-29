import random
import os
import numpy as np
import torch
import pickle
from tqdm import tqdm
from utils import bbox_corners, get_bbox, rotate_axis, rotate_point_cloud

class SurfData(torch.utils.data.Dataset):
    """ Surface VAE Dataloader - supports NCS data """
    def __init__(self, data_list, input_list, validate=False, aug=False, use_type_flag=False): 
        self.validate = validate
        self.aug = aug
        self.use_type_flag = use_type_flag  # Controls whether to use type flag

        # Load validation data
        if self.validate: 
            with open(data_list, "rb") as tf:
                data_paths = pickle.load(tf)['val']
            
            datas = [] 
            for path in data_paths:
                with open(path, "rb") as tf:
                    data = pickle.load(tf)
                if 'surf_ncs' in data:
                    datas.append(data['surf_ncs'])
            if datas:
                self.data = np.vstack(datas)
            else:
                self.data = np.array([]).reshape(0, 32, 32, 3)

        # Load training data (deduplicated)
        else:
            with open(input_list, "rb") as tf:
                self.data = pickle.load(tf)
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        surf_uv = self.data[index]
        if np.random.rand()>0.5 and self.aug:
            for axis in ['x', 'y', 'z']:
                angle = random.choice([90, 180, 270])
                surf_uv = rotate_point_cloud(surf_uv.reshape(-1, 3), angle, axis).reshape(32, 32, 3)
        
        if self.use_type_flag:
            # Add surface type flag (32, 32, 1) filled with 0s
            surface_flag = np.zeros((32, 32, 1), dtype=np.float32)
            surf_uv_with_flag = np.concatenate([surf_uv, surface_flag], axis=-1)  # (32, 32, 4)
            return torch.FloatTensor(surf_uv_with_flag)
        else:
            # Return 3-channel data without flag
            return torch.FloatTensor(surf_uv)  # (32, 32, 3)

class EdgeData(torch.utils.data.Dataset):
    """ Edge VAE Dataloader - supports NCS data """
    def __init__(self, data_list, input_list, validate=False, aug=False, use_type_flag=False): 
        self.validate = validate
        self.aug = aug
        self.use_type_flag = use_type_flag  # Controls whether to use type flag

        # Load validation data
        if self.validate: 
            with open(data_list, "rb") as tf:
                data_paths = pickle.load(tf)['val']

            datas = []
            for path in tqdm(data_paths):
                with open(path, "rb") as tf:
                    data = pickle.load(tf)

                # Modification: use 'edge_ncs' instead of 'graph_edge_grid'
                if 'edge_ncs' in data:
                    datas.append(data['edge_ncs'])
            if datas:
                self.data = np.vstack(datas)
            else:
                self.data = np.array([]).reshape(0, 32, 3)

        # Load training data (deduplicated)
        else:
            with open(input_list, "rb") as tf:
                self.data = pickle.load(tf)         
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        edge_u = self.data[index]  # shape: (32, 3)
        
        # Data augmentation, randomly rotate 50% of the times
        if np.random.rand()>0.5 and self.aug:
            for axis in ['x', 'y', 'z']:
                angle = random.choice([90, 180, 270])
                edge_u = rotate_point_cloud(edge_u, angle, axis)   
        
        # Expand edge data from (32, 3) to (32, 32, 3)
        # Method 1: simple replication (maintain original behavior) - using efficient NumPy operations
        edge_u_expanded = np.tile(edge_u[:, np.newaxis, :], (1, 32, 1))  # more efficient implementation
        
        if self.use_type_flag:
            # Add edge type flag (32, 32, 1) filled with 1s
            edge_flag = np.ones((32, 32, 1), dtype=np.float32)
            edge_u_with_flag = np.concatenate([edge_u_expanded, edge_flag], axis=-1)  # (32, 32, 4)
            return torch.FloatTensor(edge_u_with_flag)
        else:
            # Return 3-channel data without flag
            return torch.FloatTensor(edge_u_expanded)  # (32, 32, 3)

class CombinedData(torch.utils.data.Dataset):
    """ Combined Surface and Edge VAE Dataloader """
    def __init__(self, data_list, surface_list, edge_list, validate=False, aug=False, use_type_flag=True):
        rank = int(os.environ.get('RANK', '0')) if 'RANK' in os.environ else 0
        if rank == 0:
            print('Loading combined surface and edge data...')
        
        # Initialize surface dataset
        self.surf_data = SurfData(data_list, surface_list, validate=validate, aug=aug, use_type_flag=use_type_flag)
        
        # Initialize edge dataset
        self.edge_data = EdgeData(data_list, edge_list, validate=validate, aug=aug, use_type_flag=use_type_flag)
        
        # Combine data
        self.data = []
        
        # Add surface data
        for i in range(len(self.surf_data)):
            self.data.append(('surface', i))
            
        # Add edge data
        for i in range(len(self.edge_data)):
            self.data.append(('edge', i))
            
        if rank == 0:
            print(f'Combined dataset size: {len(self.data)} (surfaces: {len(self.surf_data)}, edges: {len(self.edge_data)})')
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data_type, data_index = self.data[index]
        
        if data_type == 'surface':
            # self.surf_data already returns torch.FloatTensor; avoid redundant wrapping
            return self.surf_data[data_index]
        else:  # edge
            # self.edge_data already returns torch.FloatTensor; avoid redundant wrapping
            return self.edge_data[data_index]

class ARData(torch.utils.data.Dataset):
    """
    Autoregressive CAD generation dataset - uses grouped structure (original + augmented)

    Example data format:
    {
        'train': [
            {
                'original': {
                    'input_ids': [...],
                    'attention_mask': [...]
                },
                'augmented': [
                    {'input_ids': [...], 'attention_mask': [...]},
                    {'input_ids': [...], 'attention_mask': [...]},
                    ...
                ]
            },
            ...
        ],
        'val': [...],
        'test': [...]
    }
    """

    def __init__(self, sequence_file, validate=False, args=None):
        """
        Args:
            sequence_file (str): Path to the pickle sequence file
            validate (bool): False for train, True for val
            args: Configuration object (requires max_seq_len)
        """
        self.args = args
        self.max_seq_len = args.max_seq_len
        self.validate = validate

        rank = int(os.environ.get("RANK", "0"))
        if rank == 0:
            split_name = "val" if validate else "train"
            print(f"Loading grouped AR sequences from '{sequence_file}' (split={split_name})...")

        try:
            with open(sequence_file, "rb") as f:
                data = pickle.load(f)

            # Select split based on validate flag
            split_key = "val" if validate else "train"
            self.groups = data.get(split_key, [])
            original_count = len(self.groups)

            # Filter out groups where all sequences exceed max length
            filtered_groups = []
            for g in self.groups:
                ok = False

                # Check original
                if len(g["original"]["input_ids"]) <= self.max_seq_len:
                    ok = True

                # Check augmented (if exists)
                if "augmented" in g and g["augmented"]:
                    for aug in g["augmented"]:
                        if len(aug["input_ids"]) <= self.max_seq_len:
                            ok = True
                            break

                if ok:
                    filtered_groups.append(g)

            self.groups = filtered_groups
            filtered_count = len(self.groups)

            if rank == 0:
                removed_count = original_count - filtered_count
                print(
                    f"Filtered groups by length (>{self.max_seq_len}): "
                    f"{original_count} -> {filtered_count} ({removed_count} removed)"
                )

            # Save metadata
            self.vocab_size = data["vocab_size"]
            self.special_token_size = data["special_token_size"]
            self.face_index_size = data["face_index_size"]
            
            # Maintain compatibility with old/new field names
            if 'codebook_size' in data:
                self.codebook_size = data['codebook_size']  # old version
            else:
                self.codebook_size = data['se_codebook_size']  # new version
                
            # Add support for new fields
            self.se_codebook_size = data.get('se_codebook_size', self.codebook_size)
            self.bbox_index_size = data.get('bbox_index_size', 2048)

            self.se_tokens_per_element = data["se_tokens_per_element"]
            self.bbox_tokens_per_element = data["bbox_tokens_per_element"]

            self.face_index_offset = data["face_index_offset"]
            self.se_token_offset = data["se_token_offset"]
            self.bbox_token_offset = data["bbox_token_offset"]

            special_tokens = data["special_tokens"]
            self.START_TOKEN = special_tokens["START_TOKEN"]
            self.SEP_TOKEN = special_tokens["SEP_TOKEN"]
            self.END_TOKEN = special_tokens["END_TOKEN"]
            self.PAD_TOKEN = special_tokens["PAD_TOKEN"]

        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise

    def __len__(self):
        return len(self.groups)

    # def _reindex(self, ids: torch.Tensor) -> torch.Tensor:
    #     """
    #     Apply a uniform offset to all "face indices" across the entire sequence:
    #     Face indices are tokens with values in [0, face_index_size).
    #     i' = (i + r) % face_index_size, where r is randomly sampled from [0, face_index_size).
    #     """
    #     N = int(self.face_index_size)
    #     if N <= 1:
    #         return ids

    #     # Identify positions of face indices (entire sequence, including before/after [SEP])
    #     mask = (ids >= 0) & (ids < N)
    #     if not mask.any():
    #         return ids

    #     # Uniform random offset (ensures synchronization)
    #     r = torch.randint(low=0, high=N, size=(1,), device=ids.device).item()

    #     out = ids.clone()
    #     out[mask] = (out[mask] + r) % N
    #     return out

    def __getitem__(self, idx):
        group = self.groups[idx]

        if self.validate:
            sample = group["original"]
        else:
            if random.random() < 0.5 or "augmented" not in group or not group["augmented"]:
                sample = group["original"]
            else:
                sample = random.choice(group["augmented"])

        input_ids = torch.tensor(sample["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(sample["attention_mask"], dtype=torch.long)

        # if (not self.validate) and (random.random() < 0.5):
        #     input_ids = self._reindex(input_ids)

        # if random.random() < 0.5:
        #     input_ids = self._reindex(input_ids)

        return {"input_ids": input_ids, "attention_mask": attention_mask}
    
    def collate_fn(self, batch):
        """
        Batch collation function: pad to the maximum length within the batch.
        """
        input_ids = [item["input_ids"] for item in batch]
        attention_masks = [item["attention_mask"] for item in batch]

        max_length_in_batch = max(len(ids) for ids in input_ids)

        padded_input_ids, padded_attention_masks = [], []
        for ids, mask in zip(input_ids, attention_masks):
            padding_length = max_length_in_batch - len(ids)

            if padding_length > 0:
                padded_ids = torch.cat(
                    [ids, torch.full((padding_length,), self.PAD_TOKEN, dtype=torch.long)]
                )
                padded_mask = torch.cat(
                    [mask, torch.zeros(padding_length, dtype=torch.long)]
                )
            else:
                padded_ids = ids
                padded_mask = mask

            padded_input_ids.append(padded_ids)
            padded_attention_masks.append(padded_mask)

        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_masks),
        }