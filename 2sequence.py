import os
import numpy as np
import torch
import pickle
import argparse
from tqdm import tqdm
import random
from collections import deque

# ===== Utility functions from your project =====
from utils import (
    quantize_bbox,
    bbox_corners,
    get_bbox,
    load_se_vqvae_model,
    rotate_axis,
)

# ==============================
#          Utils
# ==============================

def prepare_surface_edge_batch_for_vqvae(surf_ncs, edge_ncs, edgeFace_adj, use_type_flag=False):
    """
    Combine face/edge data into SE VQ-VAE input (N, C, 32, 32)
    - Face: (num_face, 32, 32, 3)
    - Edge: (num_edge, 32, 3) -> expand to (num_edge, 32, 32, 3)
    """
    surf_data = surf_ncs.astype(np.float32)  # (F, 32, 32, 3)
    num_face = len(surf_data)

    edge_data = edge_ncs.astype(np.float32)  # (E, 32, 3)
    edge_expanded = np.tile(edge_data[:, :, np.newaxis, :], (1, 1, 32, 1))  # (E, 32, 32, 3)
    num_edge = len(edge_data)

    # Edge-face correspondence (for subsequent BFS ordering)
    edge_face_pairs = []
    if len(edgeFace_adj) > 0:
        for edge_adj in edgeFace_adj:
            if len(edge_adj) >= 2:
                face1_idx, face2_idx = edge_adj[0], edge_adj[1]
                edge_face_pairs.append((face1_idx, face2_idx))

    if use_type_flag:
        surf_flags = np.zeros((num_face, 32, 32, 1), dtype=np.float32)
        surf_with_flags = np.concatenate([surf_data, surf_flags], axis=-1)
        edge_flags = np.ones((num_edge, 32, 32, 1), dtype=np.float32)
        edge_with_flags = np.concatenate([edge_expanded, edge_flags], axis=-1)
        combined_data = np.concatenate([surf_with_flags, edge_with_flags], axis=0)
        combined_data = combined_data.transpose(0, 3, 1, 2)
    else:
        combined_data = np.concatenate([surf_data, edge_expanded], axis=0)
        combined_data = combined_data.transpose(0, 3, 1, 2)  # (F+E, 3, 32, 32)

    return combined_data, num_face, num_edge, edge_face_pairs


def calculate_tokens_per_element(se_vqvae_model, device):
    """
    Detect the number of tokens per element for SE-VQ; bbox is fixed at 6 (min/max xyz)
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
        se_tokens = int(token_indices.numel())

    bbox_tokens = 6
    return se_tokens, bbox_tokens

def dfs_face_ordering_from_core(edge_face_pairs, num_faces):
    """
    Face ordering strategy: Depth-First Search (DFS), prioritizing low-degree neighbors
    1. Find the face with highest degree as starting point
    2. Execute DFS from starting point, prioritizing unvisited neighbors with lowest degree
    3. Generate face sequence in visit order
    
    Returns:
        face_order: [face_idx0, face_idx1, ...]  New ordering of original faces
        face_position_map: {original_face_idx: new_position}
    """
    # Build graph & degrees
    nbrs = [set() for _ in range(num_faces)]
    for f1, f2 in edge_face_pairs:
        if 0 <= f1 < num_faces and 0 <= f2 < num_faces and f1 != f2:
            nbrs[f1].add(f2); nbrs[f2].add(f1)
    deg = [len(n) for n in nbrs]

    visited = [False]*num_faces
    face_order = []

    # Use (degree descending, id ascending) as component starting point selection order
    seeds = sorted(range(num_faces), key=lambda x: (-deg[x], x))
    
    def dfs(u):
        """Depth-first search, prioritizing low-degree neighbors"""
        visited[u] = True
        face_order.append(u)
        
        # Get unvisited neighbors, sort by (degree ascending, id ascending)
        unvisited_neighbors = [v for v in nbrs[u] if not visited[v]]
        unvisited_neighbors.sort(key=lambda x: (deg[x], x))
        
        # Recursively visit each neighbor
        for v in unvisited_neighbors:
            if not visited[v]:  # Double check to prevent being visited during ordering
                dfs(v)
    
    # Execute DFS for each connected component
    for s in seeds:
        if not visited[s]:
            dfs(s)

    face_position_map = {f:i for i,f in enumerate(face_order)}
    return face_order, face_position_map


def lexicographic_edge_ordering(edge_face_pairs):
    """
    Edge ordering strategy: (max, min) lexicographic ordering
    1. For each edge's face pair (f1, f2), use 0-based position indices after ordering
    2. Calculate sort key: Key = (max(f1, f2), min(f1, f2))
    3. Sort in ascending order by sort key
    
    Args:
        edge_face_pairs: [(f1, f2), ...] Face position pairs (0-based positions after DFS ordering)
    
    Returns:
        edge_order: [eidx0, eidx1, ...]  New ordering of edges
        ordered_edge_face_pairs: [(f1, f2), ...] Aligned with edge_order
    """
    # Validate input: check if face indices are in valid range starting from 0
    if len(edge_face_pairs) > 0:
        all_face_indices = set()
        for pair in edge_face_pairs:
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                all_face_indices.add(pair[0])
                all_face_indices.add(pair[1])
        
        if len(all_face_indices) > 0:
            min_face_idx = min(all_face_indices)
            max_face_idx = max(all_face_indices)
            # Assert: face indices should start from 0 and be continuous (or at least in reasonable range)
            assert min_face_idx >= 0, f"Face index must >= 0, current minimum: {min_face_idx}"
            assert max_face_idx <= 50, f"Face index exceeds reasonable range, current maximum: {max_face_idx}"
    
    # Build edge ordering information
    edge_sort_info = []
    
    for eidx, pair in enumerate(edge_face_pairs):
        if not (isinstance(pair, (list, tuple)) and len(pair) >= 2):
            continue
        
        f1, f2 = pair[0], pair[1]
        
        # Calculate sort key: (max_idx, min_idx)
        # Use 0-based face position indices
        max_idx = max(f1, f2)
        min_idx = min(f1, f2)
        
        sort_key = (max_idx, min_idx)
        edge_sort_info.append((sort_key, eidx, pair))
    
    # Sort by sort key in ascending order
    edge_sort_info.sort(key=lambda x: x[0])
    
    # Extract sorting results
    edge_order = [item[1] for item in edge_sort_info]
    ordered_edge_face_pairs = [item[2] for item in edge_sort_info]
    
    return edge_order, ordered_edge_face_pairs

# ==============================
#    Preprocessor (group version)
# ==============================

class ARDataPreprocessor:
    def __init__(self, data_list, se_vqvae_model, args):
        self.data_list = data_list
        self.se_vqvae_model = se_vqvae_model
        self.args = args
        self.device = next(se_vqvae_model.parameters()).device

        # Detect token count
        self.se_tokens_per_element, self.bbox_tokens_per_element = calculate_tokens_per_element(
            se_vqvae_model, self.device
        )

        # Read data list (path collection)
        with open(data_list, 'rb') as f:
            ds = pickle.load(f)
        self.train_paths = ds['train']
        self.val_paths = ds.get('val', [])
        self.test_paths = ds.get('test', [])

        # Vocabulary/offsets
        self.face_index_size = 50
        self.se_codebook_size = 8192
        self.bbox_index_size = 2048
        self.special_token_size = 4

        self.face_index_offset = 0
        self.se_token_offset = self.face_index_offset + self.face_index_size
        self.bbox_token_offset = self.se_token_offset + self.se_codebook_size

        self.vocab_size = (
            self.face_index_size + self.se_codebook_size + self.bbox_index_size + self.special_token_size
        )
        special_token_offset = self.bbox_token_offset + self.bbox_index_size
        self.START_TOKEN = special_token_offset
        self.SEP_TOKEN = special_token_offset + 1
        self.END_TOKEN = special_token_offset + 2
        self.PAD_TOKEN = special_token_offset + 3

        self.group_cache = []
        self._process_all_data()

    # ---------- Main processing ----------
    def _process_all_data(self):
        for path in tqdm(self.train_paths, desc="Processing train"):
            g = self._process_single_cad(path, 'train')
            if g: self.group_cache.append(('train', g))
        for path in tqdm(self.val_paths, desc="Processing val"):
            g = self._process_single_cad(path, 'val')
            if g: self.group_cache.append(('val', g))
        for path in tqdm(self.test_paths, desc="Processing test"):
            g = self._process_single_cad(path, 'test')
            if g: self.group_cache.append(('test', g))

    # ---------- Encode a rotation version ----------
        # ---------- Encode a rotation version ----------
    def _encode_single_rotation(
        self,
        surf_ncs, edge_ncs,
        surf_bbox_wcs, edge_bbox_wcs,
        edgeFace_adj,
        rotation_angle
    ):
        """
        Returns: (tokens:list[int], attention_mask:list[int])
        """
        # Deep copy
        current_surf_ncs = surf_ncs.copy()
        current_edge_ncs = edge_ncs.copy()
        current_surf_bbox_wcs = surf_bbox_wcs.copy()
        current_edge_bbox_wcs = edge_bbox_wcs.copy()
        current_edgeFace_adj = [adj[:] for adj in edgeFace_adj] # Deep copy adjacency relations

        # Rotation
        if rotation_angle % 360 != 0:
            surfpos_corners = bbox_corners(current_surf_bbox_wcs)
            edgepos_corners = bbox_corners(current_edge_bbox_wcs)
            surfpos_corners = rotate_axis(surfpos_corners, rotation_angle, 'z', normalized=True)
            edgepos_corners = rotate_axis(edgepos_corners, rotation_angle, 'z', normalized=True)
            current_surf_ncs = rotate_axis(current_surf_ncs, rotation_angle, 'z', normalized=False)
            current_edge_ncs = rotate_axis(current_edge_ncs, rotation_angle, 'z', normalized=False)
            current_surf_bbox_wcs = get_bbox(surfpos_corners).reshape(len(current_surf_bbox_wcs), 6)
            current_edge_bbox_wcs = get_bbox(edgepos_corners).reshape(len(current_edge_bbox_wcs), 6)

        # VQ encoding: prepare face and edge data
        se_data, num_face, num_edge, edge_face_pairs = prepare_surface_edge_batch_for_vqvae(
            current_surf_ncs, current_edge_ncs, current_edgeFace_adj, use_type_flag=False
        )
        
        # ===== New three-stage ordering strategy =====
        # Stage 1: Face ordering (DFS from edge to core)
        face_order, face_position_map = dfs_face_ordering_from_core(edge_face_pairs, num_face)
        
        # Rearrange face-related data according to new face ordering
        current_surf_ncs = current_surf_ncs[face_order]
        current_surf_bbox_wcs = current_surf_bbox_wcs[face_order]
        
        # Update face portion in SE data (first num_face elements)
        se_data[:num_face] = se_data[face_order]
        
        # Update edge-face adjacency relations using new face positions (0-based indices)
        updated_edge_face_pairs = []
        for f1, f2 in edge_face_pairs:
            new_f1 = face_position_map[f1]
            new_f2 = face_position_map[f2]
            updated_edge_face_pairs.append((new_f1, new_f2))
        edge_face_pairs = updated_edge_face_pairs
        
        # Stage 2: Edge ordering (lexicographic: max-min ordering, using 0-based indices)
        edge_order, ordered_edge_face_pairs = lexicographic_edge_ordering(edge_face_pairs)
        
        # Stage 3: Face index cyclic offset (re-indexing at the end)
        max_faces = self.args.max_face
        num_faces = num_face
        N = min(self.face_index_size, max_faces)  # This is 50
        r = random.randint(0, N - 1) if N > 0 else 0
        face_index_map = {i: (i + r) % N for i in range(num_faces)} if N > 0 else {i: i for i in range(num_faces)}

        # bbox quantization (note empty check)
        surf_bbox_indices = []
        edge_bbox_indices = []
        if len(current_surf_bbox_wcs) > 0:
            surf_bbox_indices = quantize_bbox(
                np.array(current_surf_bbox_wcs) * float(self.args.scale),
                num_tokens=self.bbox_index_size
            ).tolist()
        if len(current_edge_bbox_wcs) > 0:
            edge_bbox_indices = quantize_bbox(
                np.array(current_edge_bbox_wcs) * float(self.args.scale),
                num_tokens=self.bbox_index_size
            ).tolist()

        # SE encoding
        se_indices = []
        if len(se_data) > 0:
            se_tensor = torch.tensor(se_data, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                h = self.se_vqvae_model.encoder(se_tensor)
                h = self.se_vqvae_model.quant_conv(h)
                _, _, indices = self.se_vqvae_model.quantize(h)
                token_indices = (
                    indices[2] if isinstance(indices, tuple) and len(indices) > 2
                    else indices[0] if isinstance(indices, tuple)
                    else indices
                )
                se_indices = token_indices.cpu().reshape(len(se_data), self.se_tokens_per_element).tolist()

        surface_indices = se_indices[:num_face] if se_indices else []
        edge_indices = se_indices[num_face:num_face + num_edge] if se_indices else []
        ordered_edge_indices = [edge_indices[i] for i in range(len(edge_indices))] if edge_indices else []
        if edge_indices and len(edge_order) == len(edge_indices):
            ordered_edge_indices = [edge_indices[i] for i in edge_order]
        ordered_edge_bbox_indices = [edge_bbox_indices[i] for i in edge_order] if edge_bbox_indices and len(edge_order) > 0 else []

        # Concatenate tokens
        tokens, attention_mask = [], []
        tokens.append(self.START_TOKEN); attention_mask.append(1)

        # Faces
        for i in range(num_face):
            if i < len(surf_bbox_indices):
                for bbox_idx in surf_bbox_indices[i]:  # 6
                    tokens.append(self.bbox_token_offset + int(bbox_idx)); attention_mask.append(1)
            if i < len(surface_indices):
                for surf_idx in surface_indices[i]:
                    tokens.append(self.se_token_offset + int(surf_idx)); attention_mask.append(1)
            tokens.append(self.face_index_offset + face_index_map[i]); attention_mask.append(1)

        tokens.append(self.SEP_TOKEN); attention_mask.append(1)

        # Edges
        for k, (face_pair) in enumerate(ordered_edge_face_pairs):
            src, dst = face_pair
            tokens.append(self.face_index_offset + face_index_map[src]); attention_mask.append(1)
            tokens.append(self.face_index_offset + face_index_map[dst]); attention_mask.append(1)

            if k < len(ordered_edge_bbox_indices):
                for bbox_idx in ordered_edge_bbox_indices[k]:  # 6
                    tokens.append(self.bbox_token_offset + int(bbox_idx)); attention_mask.append(1)

            if k < len(ordered_edge_indices):
                for eidx in ordered_edge_indices[k]:
                    tokens.append(self.se_token_offset + int(eidx)); attention_mask.append(1)

        tokens.append(self.END_TOKEN); attention_mask.append(1)
        return tokens, attention_mask

    # ---------- Process single CAD ----------
    def _process_single_cad(self, path, split='train'):
        try:
            with open(path, 'rb') as f:
                cad = pickle.load(f)

            # Basic fields
            surf_ncs = np.array(cad.get('surf_ncs', []), dtype=np.float32)       # (F,32,32,3)
            edge_ncs = np.array(cad.get('edge_ncs', []), dtype=np.float32)       # (E,32,3)
            edge_bbox_wcs = np.array(cad.get('edge_bbox_wcs', []), dtype=np.float32)  # (E,6)
            surf_bbox_wcs = np.array(cad.get('surf_bbox_wcs', []), dtype=np.float32)  # (F,6)
            edgeFace_adj = cad.get('edgeFace_adj', [])
            faceEdge_adj = cad.get('faceEdge_adj', None)  # list[list[edge_idx]]

            # 1) Empty check
            if len(surf_ncs) == 0 or len(edge_ncs) == 0:
                return None

            # 2) Upper limit filtering
            if len(surf_ncs) > int(self.args.max_face):
                return None
            if len(edge_ncs) > int(self.args.max_edge):
                return None


            # Filter out faces that are too close to each other
            threshold_value = 0.05
            scaled_value = 3
            
            surf_bbox = surf_bbox_wcs * scaled_value

            _surf_bbox_ = surf_bbox.reshape(len(surf_bbox), 2, 3)
            non_repeat = _surf_bbox_[:1]
            for bbox in _surf_bbox_:
                diff = np.max(np.max(np.abs(non_repeat - bbox), -1), -1)
                same = diff < threshold_value
                if same.sum() >= 1:
                    continue # Duplicate value
                else:
                    non_repeat = np.concatenate([non_repeat, bbox[np.newaxis,:,:]], 0)
            if len(non_repeat) != len(_surf_bbox_):
                return

            # Filter out edges that are too close to each other
            se_bbox = []
            for adj in faceEdge_adj:
                if len(edge_bbox_wcs[adj]) == 0: 
                    return
                se_bbox.append(edge_bbox_wcs[adj] * scaled_value)

            for bbb in se_bbox:
                _edge_bbox_ = bbb.reshape(len(bbb), 2, 3)
                non_repeat = _edge_bbox_[:1]
                for bbox in _edge_bbox_:
                    diff = np.max(np.max(np.abs(non_repeat - bbox), -1), -1)
                    same = diff < threshold_value
                    if same.sum() >= 1:
                        continue # Duplicate value
                    else:
                        non_repeat = np.concatenate([non_repeat, bbox[np.newaxis,:,:]], 0)
                if len(non_repeat) != len(_edge_bbox_):
                    return None

            # Rotation angle set: only train uses data augmentation, val and test only use original data
            if split == 'train' and bool(self.args.aug):
                rotation_angles = [0, 90, 180, 270]
            else:
                rotation_angles = [0]

            # Determine save format based on split
            if split == 'train':
                # train saves group format (original + augmented)
                group = {
                    'original': None,
                    'augmented': []
                }
                
                for rot in rotation_angles:
                    tokens, attn = self._encode_single_rotation(
                        surf_ncs, edge_ncs,
                        surf_bbox_wcs, edge_bbox_wcs,
                        edgeFace_adj,
                        rotation_angle=rot
                    )
                    item = {'input_ids': tokens, 'attention_mask': attn}
                    if rot == 0:
                        group['original'] = item
                    else:
                        group['augmented'].append(item)
                
                # If exception causes original to be empty, don't return
                if group['original'] is None:
                    return None
                
                return group
            else:
                # val and test only save original data
                tokens, attn = self._encode_single_rotation(
                    surf_ncs, edge_ncs,
                    surf_bbox_wcs, edge_bbox_wcs,
                    edgeFace_adj,
                    rotation_angle=0
                )
                item = {'input_ids': tokens, 'attention_mask': attn}
                group = {'original': item}
                return group

        except Exception as e:
            print(f"[WARN] Error processing {path}: {e}")
            return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_list', type=str, default='data/abc_data_split_6bit.pkl', help='Path to pkl with train/val/test paths')
    parser.add_argument('--output_file', type=str, default='data/abc_sequences.pkl', help='Output pickle file (group format)')
    parser.add_argument('--vqvae_se_weight', type=str, default='checkpoint/se/abc_se_vqvae_epoch.pt', help='Pre-trained face/edge VQ-VAE model weight path')
    parser.add_argument('--max_face', type=int, default=50)
    parser.add_argument('--max_edge', type=int, default=150)
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--aug', default=True, type=bool, help='Whether to save rotation augmentation (90/180/270)')

    parser.add_argument("--gpu", type=int, nargs='+', default=[0], help="GPU IDs to use")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.gpu))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    se_vqvae_model = load_se_vqvae_model(args.vqvae_se_weight, False, device)

    processor = ARDataPreprocessor(args.data_list, se_vqvae_model, args)

    # Split cache to each split
    train_groups, val_groups, test_groups = [], [], []
    for split, group in processor.group_cache:
        if split == 'train': train_groups.append(group)
        elif split == 'val': val_groups.append(group)
        elif split == 'test': test_groups.append(group)

    # Package output (including metadata, ARData will use)
    output_data = {
        'train': train_groups,
        'val': val_groups,
        'test': test_groups,
        'vocab_size': processor.vocab_size,
        'special_token_size': processor.special_token_size,
        'face_index_size': processor.face_index_size,
        'se_codebook_size': processor.se_codebook_size,
        'bbox_index_size': processor.bbox_index_size,
        'face_index_offset': processor.face_index_offset,
        'se_token_offset': processor.se_token_offset,
        'bbox_token_offset': processor.bbox_token_offset,
        'se_tokens_per_element': processor.se_tokens_per_element,
        'bbox_tokens_per_element': processor.bbox_tokens_per_element,
        'special_tokens': {
            'START_TOKEN': processor.START_TOKEN,
            'SEP_TOKEN': processor.SEP_TOKEN,
            'END_TOKEN': processor.END_TOKEN,
            'PAD_TOKEN': processor.PAD_TOKEN,
        }
    }

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'wb') as f:
        pickle.dump(output_data, f)

    print(f"[DONE] Saved groups -> {args.output_file}")
    print(f"  train: {len(train_groups)} | val: {len(val_groups)} | test: {len(test_groups)}")
    print(f"  aug enabled: {bool(args.aug)}")


if __name__ == "__main__":
    main()
