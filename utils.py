import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict
from diffusers import VQModel
from chamferdist import ChamferDistance
from OCC.Core.gp import gp_Pnt
from OCC.Core.TColgp import TColgp_Array1OfPnt, TColgp_Array2OfPnt
from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
from OCC.Core.GeomAPI import GeomAPI_PointsToBSplineSurface, GeomAPI_PointsToBSpline
from OCC.Core.GeomAbs import GeomAbs_C2
from OCC.Core.Geom import Geom_BSplineSurface, Geom_BSplineCurve
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeEdge, BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRep import BRep_Tool
from OCC.Core.ShapeFix import ShapeFix_Face, ShapeFix_Wire, ShapeFix_Edge
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Wire, ShapeAnalysis_Shell, ShapeAnalysis_FreeBounds
from OCC.Core.ShapeExtend import ShapeExtend_WireData
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Face, TopoDS_Wire, TopoDS_Shell, TopoDS_Edge, TopoDS_Solid, topods_Shell, topods_Wire, topods_Face, topods_Edge
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_WIRE, TopAbs_SHELL, TopAbs_EDGE
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Extend.TopologyUtils import TopologyExplorer, WireExplorer
import OCC.Core.BRep
from quantise import VectorQuantiser
from occwl.io import Solid, load_step
import shutup; shutup.please()

### Parameter Loading ###

def get_se_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_list', type=str, default='data/deepcad_data_split_6bit.pkl', help='Path to data file path list')
    parser.add_argument('--surface_list', type=str, default='data/deepcad_parsed_unique_surfaces.pkl',
                        help='Path to deduplicated surface source data')
    parser.add_argument('--edge_list', type=str, default='data/deepcad_parsed_unique_edges.pkl',
                        help='Path to deduplicated edge source data')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size per GPU')
    parser.add_argument('--train_epoch', type=int, default=3000, help='number of epochs to train for')
    parser.add_argument('--test_epoch', type=int, default=1, help='number of epochs to test model')
    parser.add_argument('--save_epoch', type=int, default=5, help='number of epochs to save model')
    parser.add_argument('--max_face', type=int, default=50, help='maximum number of faces')
    parser.add_argument('--max_edge', type=int, default=150, help='maximum number of edges per face')
    parser.add_argument('--save_folder', type=str, default='my_data', help='save folder')
    parser.add_argument("--gpu", type=int, nargs='+', default=[0], help="GPU IDs to use for training.")
    parser.add_argument('--weight', type=str, default='', help='Specify checkpoint file path. Leave empty to train from scratch')
    
    # === VQ-VAE Model Configuration Options ===
    parser.add_argument('--use_type_flag', action='store_true', default=False,
                        help='not used in this project')
    
    # Dataset type parameter
    parser.add_argument('--dataset_type', type=str, choices=['furniture', 'deepcad', 'abc'],
                        default='deepcad', help='Dataset type to use')

    # Save dirs and reload
    parser.add_argument('--env', type=str, default="", help='environment')
    parser.add_argument('--dir_name', type=str, default="checkpoint", help='name of the log folder.')
    parser.add_argument('--loss_dir', type=str, default="./", help='name of the loss folder.')
    parser.add_argument('--tb_log_dir', type=str, default="", help='TensorBoard log directory path.')
    args = parser.parse_args()
    # saved folder
    args.save_dir = f'{args.dir_name}/{args.env}'
    return args

def get_ar_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence_file', type=str, default='data/deepcad_sequences_v3_no_vertex_v9.2.pkl', help='source file path of AR sequences')
    
    # Training parameters - Optimized parameters
    # Batch size per GPU
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size - increased for better stability')
    parser.add_argument('--train_epoch', type=int, default=500, help='number of epochs to train for')
    parser.add_argument('--test_epoch', type=int, default=1, help='number of epochs to validate model')
    parser.add_argument('--save_epoch', type=int, default=50, help='number of epochs to save model')
    parser.add_argument('--max_face', type=int, default=50, help='maximum number of faces')
    parser.add_argument('--max_edge', type=int, default=150, help='maximum number of edges per face')
    parser.add_argument('--save_folder', type=str, default='my_data', help='save folder')
    parser.add_argument('--weight', type=str, default='', help='Specify checkpoint file path. Leave empty to train from scratch')

    parser.add_argument('--max_seq_len', type=int, default=2048, help='Maximum sequence length')
    
    # === Anti-overfitting Parameters ===
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate (reduced to reduce overfitting)')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='L2 weight decay coefficient (enhance regularization)')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability (enhance regularization)')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing coefficient (enhance regularization)')
    
    # === Model Architecture Parameters ===
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension (reduced from 768 to 512)')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=8, help='Number of Transformer layers')
    parser.add_argument('--dim_feedforward', type=int, default=1024, help='Feedforward network dimension')
    
    # Dataset type parameter
    parser.add_argument('--dataset_type', type=str, choices=['furniture', 'deepcad', 'abc'],
                        default='abc', help='Dataset type to use')

    # Save dirs and reload
    parser.add_argument('--env', type=str, default="ar/256,1024,8,8", help='environment')
    parser.add_argument('--dir_name', type=str, default="checkpoints", help='name of the log folder.')
    parser.add_argument('--loss_dir', type=str, default="./", help='name of the loss folder.')
    parser.add_argument('--tb_log_dir', type=str, default="logs/ar/256,1024,8,8", help='name of the tensorboard log folder.')
    args = parser.parse_args()
    # saved folder
    args.save_dir = f'{args.dir_name}/{args.env}'
    return args

### eval ###
def check_brep_validity(step_file_path):

    if isinstance(step_file_path, str):
        # Read the STEP file
        step_reader = STEPControl_Reader()
        status = step_reader.ReadFile(step_file_path)

        if status != IFSelect_RetDone:
            print("Error: Unable to read STEP file")
            return False

        step_reader.TransferRoot()
        shape = step_reader.Shape()

    elif isinstance(step_file_path, TopoDS_Solid):
        shape = step_file_path

    else:
        return False

    # Initialize check results
    wire_order_ok = True
    wire_self_intersection_ok = True
    shell_bad_edges_ok = True
    brep_closed_ok = True  # Initialize closed BRep check
    solid_one_ok = True

    # 1. Check if BRep has more than one solid
    if isinstance(step_file_path, str):
        try:
            cad_solid = load_step(step_file_path)
            if len(cad_solid) != 1:
                solid_one_ok = False
        except Exception as e:
            return False

    # 2. Check all wires
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while face_explorer.More():
        face = topods_Face(face_explorer.Current())
        wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
        while wire_explorer.More():
            wire = topods_Wire(wire_explorer.Current())

            # Create a ShapeFix_Wire object
            wire_fixer = ShapeFix_Wire(wire, face, 0.01)
            wire_fixer.Load(wire)
            wire_fixer.SetFace(face)
            wire_fixer.SetPrecision(0.01)
            wire_fixer.SetMaxTolerance(1)
            wire_fixer.SetMinTolerance(0.0001)

            # Fix the wire
            wire_fixer.Perform()
            fixed_wire = wire_fixer.Wire()

            # Analyze the fixed wire
            wire_analysis = ShapeAnalysis_Wire(fixed_wire, face, 0.01)
            wire_analysis.Load(fixed_wire)
            wire_analysis.SetPrecision(0.01)
            wire_analysis.SetSurface(BRep_Tool.Surface(face))

            # 1. Check wire edge order
            order_status = wire_analysis.CheckOrder()
            if order_status != 0:  # 0 means no error
                # print(f"Wire order issue detected: {order_status}")
                wire_order_ok = False

            # 2. Check wire self-intersection
            if wire_analysis.CheckSelfIntersection():
                wire_self_intersection_ok = False

            wire_explorer.Next()
        face_explorer.Next()

    # 3. Check for bad edges in shells
    shell_explorer = TopExp_Explorer(shape, TopAbs_SHELL)
    while shell_explorer.More():
        shell = topods_Shell(shell_explorer.Current())
        shell_analysis = ShapeAnalysis_Shell()
        shell_analysis.LoadShells(shell)

        if shell_analysis.HasBadEdges():
            shell_bad_edges_ok = False

        shell_explorer.Next()

    # 4. Check if BRep is closed (no free edges)
    free_bounds = ShapeAnalysis_FreeBounds(shape)
    free_edges = free_bounds.GetOpenWires()
    edge_explorer = TopExp_Explorer(free_edges, TopAbs_EDGE)
    num_free_edges = 0
    while edge_explorer.More():
        edge = topods_Edge(edge_explorer.Current())
        num_free_edges += 1
        # print(f"Free edge: {edge}")
        edge_explorer.Next()
    if num_free_edges > 0:
        brep_closed_ok = False

    return int(wire_order_ok and wire_self_intersection_ok and shell_bad_edges_ok and brep_closed_ok and solid_one_ok)

### Data Processing ###

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

def get_bbox(pnts):
    """
    Get the tighest fitting 3D (axis-aligned) bounding box giving a set of points
    """
    bbox_corners = []
    for point_cloud in pnts:
        # Find the minimum and maximum coordinates along each axis
        min_x = np.min(point_cloud[:, 0])
        max_x = np.max(point_cloud[:, 0])

        min_y = np.min(point_cloud[:, 1])
        max_y = np.max(point_cloud[:, 1])

        min_z = np.min(point_cloud[:, 2])
        max_z = np.max(point_cloud[:, 2])

        # Create the 3D bounding box using the min and max values
        min_point = np.array([min_x, min_y, min_z])
        max_point = np.array([max_x, max_y, max_z])
        bbox_corners.append([min_point, max_point])
    return np.array(bbox_corners)

def rotate_axis(pnts, angle_degrees, axis, normalized=False):
    """
    Rotate a point cloud around its center by a specified angle in degrees along a specified axis.

    Args:
    - point_cloud: Numpy array of shape (N, ..., 3) representing the point cloud.
    - angle_degrees: Angle of rotation in degrees.
    - axis: Axis of rotation. Can be 'x', 'y', or 'z'.

    Returns:
    - rotated_point_cloud: Numpy array of shape (N, 3) representing the rotated point cloud.
    """

    # Convert angle to radians
    angle_radians = np.radians(angle_degrees)
    
    # Convert points to homogeneous coordinates
    shape = list(np.shape(pnts))
    shape[-1] = 1
    pnts_homogeneous = np.concatenate((pnts, np.ones(shape)), axis=-1)

    # Compute rotation matrix based on the specified axis
    if axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0, 0],
            [0, np.cos(angle_radians), -np.sin(angle_radians), 0],
            [0, np.sin(angle_radians), np.cos(angle_radians), 0],
            [0, 0, 0, 1]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [np.cos(angle_radians), 0, np.sin(angle_radians), 0],
            [0, 1, 0, 0],
            [-np.sin(angle_radians), 0, np.cos(angle_radians), 0],
            [0, 0, 0, 1]
        ])
    elif axis == 'z':
        rotation_matrix = np.array([
            [np.cos(angle_radians), -np.sin(angle_radians), 0, 0],
            [np.sin(angle_radians), np.cos(angle_radians), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    else:
        raise ValueError("Invalid axis. Must be 'x', 'y', or 'z'.")

    # Apply rotation
    rotated_pnts_homogeneous = np.dot(pnts_homogeneous, rotation_matrix.T)
    rotated_pnts = rotated_pnts_homogeneous[...,:3]

    # Scale the point cloud to fit within the -1 to 1 cube
    if normalized:
        max_abs_coord = np.max(np.abs(rotated_pnts))
        rotated_pnts = rotated_pnts / max_abs_coord

    return rotated_pnts

def bbox_corners(bboxes):
    """
    Given the bottom-left and top-right corners of the bbox
    Return all eight corners 
    """
    bboxes_all_corners = []
    for bbox in bboxes:
        bottom_left, top_right = bbox[:3], bbox[3:]
        # Bottom 4 corners
        bottom_front_left = bottom_left
        bottom_front_right = (top_right[0], bottom_left[1], bottom_left[2])
        bottom_back_left = (bottom_left[0], top_right[1], bottom_left[2])
        bottom_back_right = (top_right[0], top_right[1], bottom_left[2])

        # Top 4 corners
        top_front_left = (bottom_left[0], bottom_left[1], top_right[2])
        top_front_right = (top_right[0], bottom_left[1], top_right[2])
        top_back_left = (bottom_left[0], top_right[1], top_right[2])
        top_back_right = top_right

        # Combine all coordinates
        all_corners = [
            bottom_front_left,
            bottom_front_right,
            bottom_back_left,
            bottom_back_right,
            top_front_left,
            top_front_right,
            top_back_left,
            top_back_right,
        ]
        bboxes_all_corners.append(np.vstack(all_corners))
    bboxes_all_corners = np.array(bboxes_all_corners)
    return bboxes_all_corners

def compute_bbox_center_and_size(min_corner, max_corner):
    # Calculate the center
    center_x = (min_corner[0] + max_corner[0]) / 2
    center_y = (min_corner[1] + max_corner[1]) / 2
    center_z = (min_corner[2] + max_corner[2]) / 2
    center = np.array([center_x, center_y, center_z])
    # Calculate the size
    size_x = max_corner[0] - min_corner[0]
    size_y = max_corner[1] - min_corner[1]
    size_z = max_corner[2] - min_corner[2]
    size = max(size_x, size_y, size_z)
    return center, size

def rotate_point_cloud(point_cloud, angle_degrees, axis):
    """
    Rotate a point cloud around its center by a specified angle in degrees along a specified axis.

    Args:
    - point_cloud: Numpy array of shape (N, 3) representing the point cloud.
    - angle_degrees: Angle of rotation in degrees.
    - axis: Axis of rotation. Can be 'x', 'y', or 'z'.

    Returns:
    - rotated_point_cloud: Numpy array of shape (N, 3) representing the rotated point cloud.
    """

    # Convert angle to radians
    angle_radians = np.radians(angle_degrees)

    # Compute rotation matrix based on the specified axis
    if axis == 'x':
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, np.cos(angle_radians), -np.sin(angle_radians)],
                                    [0, np.sin(angle_radians), np.cos(angle_radians)]])
    elif axis == 'y':
        rotation_matrix = np.array([[np.cos(angle_radians), 0, np.sin(angle_radians)],
                                    [0, 1, 0],
                                    [-np.sin(angle_radians), 0, np.cos(angle_radians)]])
    elif axis == 'z':
        rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians), 0],
                                    [np.sin(angle_radians), np.cos(angle_radians), 0],
                                    [0, 0, 1]])
    else:
        raise ValueError("Invalid axis. Must be 'x', 'y', or 'z'.")

    # Center the point cloud
    center = np.mean(point_cloud, axis=0)
    centered_point_cloud = point_cloud - center

    # Apply rotation
    rotated_point_cloud = np.dot(centered_point_cloud, rotation_matrix.T)

    # Translate back to original position
    rotated_point_cloud += center

    # Find the maximum absolute coordinate value
    max_abs_coord = np.max(np.abs(rotated_point_cloud))

    # Scale the point cloud to fit within the -1 to 1 cube
    normalized_point_cloud = rotated_point_cloud / max_abs_coord

    return normalized_point_cloud

def keep_largelist(int_lists):
    # Initialize a list to store the largest integer lists
    largest_int_lists = []

    # Convert each list to a set for efficient comparison
    sets = [set(lst) for lst in int_lists]

    # Iterate through the sets and check if they are subsets of others
    for i, s1 in enumerate(sets):
        is_subset = False
        for j, s2 in enumerate(sets):
            if i != j and s1.issubset(s2) and s1 != s2:
                is_subset = True
                break
        if not is_subset:
            largest_int_lists.append(list(s1))

    # Initialize a set to keep track of seen tuples
    seen_tuples = set()

    # Initialize a list to store unique integer lists
    unique_int_lists = []

    # Iterate through the input list
    for int_list in largest_int_lists:
        # Convert the list to a tuple for hashing
        int_tuple = tuple(sorted(int_list))

        # Check if the tuple is not in the set of seen tuples
        if int_tuple not in seen_tuples:
            # Add the tuple to the set of seen tuples
            seen_tuples.add(int_tuple)

            # Add the original list to the list of unique integer lists
            unique_int_lists.append(int_list)

    return unique_int_lists

def load_se_vqvae_model(vqvae_model_path, use_type_flag, dataset_type, device):
    if dataset_type == 'deepcad':
        num_vq_embeddings = 4096
    elif dataset_type == 'abc':
        num_vq_embeddings = 8192
    try:
        in_channels = 4 if use_type_flag else 3
        
        vqvae_model = VQVAE(
            in_channels=in_channels,  # Set input channels based on flag
            out_channels=3,
            down_block_types=['DownEncoderBlock2D'] * 5,
            up_block_types=['UpDecoderBlock2D'] * 5,
            block_out_channels=[32, 64, 128, 256, 512],  
            layers_per_block=2,
            act_fn='silu',
            latent_channels=128,
            vq_embed_dim=64,
            num_vq_embeddings=num_vq_embeddings,
            norm_num_groups=32,
            sample_size=512,
        )
        
        # Load weights
        checkpoint = torch.load(vqvae_model_path, map_location=device)
        # Only take model parameters
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        vqvae_model.load_state_dict(state_dict)
        vqvae_model.eval().to(device)
        
        # print(f"VQ-VAE loaded successfully (in_channels={in_channels}, use_type_flag={use_type_flag})")
        return vqvae_model
        
    except Exception as e:
        print(f"Failed to load VQ-VAE: {e}")
        print("Using placeholder tokens instead")
        return None

def quantize_bbox(bbox_coords, num_tokens=2048):
    """
    Quantize bounding box coordinates in range [-1, 1] to integer indices.
    Use higher num_tokens (1024) to preserve higher precision for Bbox.
    """
    normalized_coords = (bbox_coords + 1) / 2.0
    if isinstance(bbox_coords, torch.Tensor):
        normalized_coords = torch.clip(normalized_coords, 0, 1)
        scaled_coords = normalized_coords * (num_tokens - 1)
        quantized_indices = torch.round(scaled_coords).long()
    else: # Assume NumPy array
        normalized_coords = np.clip(normalized_coords, 0, 1)
        scaled_coords = normalized_coords * (num_tokens - 1)
        quantized_indices = np.round(scaled_coords).astype(int)
    return quantized_indices

def dequantize_bbox(indices, num_tokens=2048):
    """
    Dequantize integer indices to bounding box coordinates in range [-1, 1].
    """
    if isinstance(indices, torch.Tensor):
        float_indices = indices.float()
    else:
        float_indices = indices.astype(float)
    normalized_coords = float_indices / (num_tokens - 1)
    bbox_coords = normalized_coords * 2.0 - 1.0
    return bbox_coords

def prepare_vqvae_input(data_ncs, data_type='face', use_type_flag=False, device="cpu"):
    """
    Prepare input data for VQVAE.

    This function converts different types (face, edge, bounding box) of NCS/WCS data to
    the standard tensor format (B, C, H, W) required by the VQ-VAE model.

    - The bbox part has been modified according to the replication method in Dataset, only supports (B, 6) input.
    
    Args:
        data_ncs (np.ndarray or list): Input geometric data.
            - For 'face': shape is (B, 32, 32, 3).
            - For 'edge': shape is (B, 32, 3).
            - For 'bbox': shape is strictly (B, 6).
        data_type (str): Data type, options are 'face', 'edge', 'bbox'.
        use_type_flag (bool): Whether to add type flag for face and edge data.
        device (str): Target device, e.g., "cpu" or "cuda".

    Returns:
        torch.Tensor: Formatted PyTorch tensor.
    """
    
    if data_type == 'face':
        # 1. Convert face data
        # Input shape: (B, 32, 32, 3)
        face_data = torch.from_numpy(np.array(data_ncs)).float().to(device)
        
        if use_type_flag:
            # Add face data flag (flag is 0)
            face_flags = torch.zeros(face_data.shape[0], 32, 32, 1, device=device)
            face_data = torch.cat([face_data, face_flags], dim=-1)  # (B, 32, 32, 4)
        
        # Convert to model input format (B, C, H, W)
        # (B, 32, 32, C) -> (B, C, 32, 32)
        return face_data.permute(0, 3, 1, 2)
        
    elif data_type == 'edge':
        # 2. Convert edge data
        # Input shape: (B, 32, 3)
        edge_data = torch.from_numpy(np.array(data_ncs)).float().to(device)
        
        # Expand edge data from line to surface -> (B, 32, 1, 3) -> (B, 32, 32, 3)
        edge_data_expanded = edge_data.unsqueeze(2).repeat(1, 1, 32, 1)
        
        if use_type_flag:
            # Add edge data flag (flag is 1)
            edge_flags = torch.ones(edge_data_expanded.shape[0], 32, 32, 1, device=device)
            edge_data_expanded = torch.cat([edge_data_expanded, edge_flags], dim=-1)  # (B, 32, 32, 4)
        
        # Convert to model input format (B, C, H, W)
        # (B, 32, 32, C) -> (B, C, 32, 32)
        return edge_data_expanded.permute(0, 3, 1, 2)
    
    elif data_type == 'bbox':
        
        bbox_data = np.array(data_ncs)

        batch_size = bbox_data.shape[0]
        # Pre-allocate empty NumPy array to store results
        # bbox_expanded_batch = np.zeros((batch_size, 16, 16, 3), dtype=np.float32)
        bbox_expanded_batch = np.zeros((batch_size, 3, 2), dtype=np.float32)

        # Step B: Loop through each item in the batch
        for i in range(batch_size):
            # Get current bbox to process, shape: (6,)
            bbox_item = bbox_data[i]

            # --- Below is the NumPy code you specified, unmodified ---
            # Convert 6D vector to (8,8,3) tensor - using efficient v1 method
            # bbox_reshaped = edge_bbox.reshape(2, 3)  # (2, 3)
            # bbox_repeated = np.repeat(bbox_reshaped, 8, axis=0)  # (8, 3)
    
            # Efficient implementation: directly tile to second dimension
            # bbox_expanded = np.tile(bbox_repeated[np.newaxis, :, :], (16, 1, 1)) # (8, 8, 3)

            

            # # 1. Extract min and max vectors
            min_vals = bbox_item[:3]  # -> (min_x, min_y, min_z)
            max_vals = bbox_item[3:]  # -> (max_x, max_y, max_z)

            # # 2. Stack them along the new "column" dimension
            # np.stack with axis=-1 will stack (3,) and (3,) into (3, 2)
            bbox_reshaped = np.stack([min_vals, max_vals], axis=-1)

            # Store single processing result into batch array
            bbox_expanded_batch[i] = bbox_reshaped
        
        # Step C: Convert final NumPy batch array to PyTorch tensor
        final_tensor = torch.from_numpy(bbox_expanded_batch).float().to(device)
        
        # Step D: Adjust channel order to match model input (B, H, W, C) -> (B, C, H, W)
        # return final_tensor.permute(0, 3, 1, 2)
        return final_tensor
    
    else:
        raise ValueError(f"Unknown data_type: '{data_type}'. Must be 'face', 'edge', or 'bbox'.")

def decode_tokens_to_ncs(tokens, vqvae_model, data_type='face', tokens_per_element=4, device="cpu"):
    """Decode token indices to NCS data"""
    if len(tokens) == 0:
        return []
    
    with torch.no_grad():
        token_tensor = torch.tensor(tokens, dtype=torch.long).to(device)
        batch_size, seq_len = token_tensor.shape
        
        feat_h = feat_w = int(np.sqrt(tokens_per_element))
        # print(f"Using feature map size: {feat_h}x{feat_w} for {seq_len} tokens")
        
        token_indices_reshaped = token_tensor.reshape(batch_size, feat_h, feat_w)
        
        if hasattr(vqvae_model.quantize, 'embedding'):
            embedding_weight = vqvae_model.quantize.embedding.weight
        elif hasattr(vqvae_model.quantize, 'embed'):
            embedding_weight = vqvae_model.quantize.embed.weight
        else:
            print("Error: Cannot find embedding weight in quantizer")
            return []
        
        quantized_features = torch.nn.functional.embedding(token_indices_reshaped, embedding_weight)
        quantized_features = quantized_features.permute(0, 3, 1, 2)
        
        decoded = vqvae_model.decoder(vqvae_model.post_quant_conv(quantized_features))
        
        if data_type == 'face' or data_type == 'edge':
            return convert_vqvae_output_to_ncs(decoded, data_type)
        else:
            # Non-face/edge cases are not handled in this function
            return []

def parse_sequence_to_cad_data(sequence, vocab_info, se_vqvae_model, device="cpu", scale_factor=1.0):
    """
    Parse autoregressive sequence to CAD data, adapted to new vertex-free sequence format
    New format: [START] bbox_tokens face_tokens face_index ... [SEP] face_index face_index bbox_tokens edge_tokens ... [END]
    
    Args:
        sequence: Input token sequence
        vocab_info: Vocabulary information dictionary
        se_vqvae_model: Face/edge VQ-VAE model
        bbox_vqvae_model: Bounding box VQ-VAE model
        device: Computing device
        scale_factor: Scale factor
        
    Returns:
        CAD data dictionary (does not contain vertex data)
    """
    face_index_offset = vocab_info['face_index_offset']
    se_token_offset = vocab_info['se_token_offset']
    bbox_token_offset = vocab_info['bbox_token_offset']
    se_codebook_size = vocab_info['se_codebook_size']
    bbox_index_size = vocab_info['bbox_index_size']
    
    START_TOKEN = vocab_info['START_TOKEN']
    SEP_TOKEN = vocab_info['SEP_TOKEN']
    END_TOKEN = vocab_info['END_TOKEN']
    
    se_tokens_per_element = vocab_info['se_tokens_per_element']
    bbox_tokens_per_element = vocab_info['bbox_tokens_per_element']
    
    i = 0
    faces, face_bboxes, edges, edge_bboxes, edge_face_pairs = [], [], [], [], []
    
    if i < len(sequence) and sequence[i] == START_TOKEN:
        i += 1
    
    # Part 1: Parse faces - Format: bbox_tokens face_tokens face_index
    while i < len(sequence) and sequence[i] != SEP_TOKEN:
        bbox_tokens = []
        for _ in range(bbox_tokens_per_element):
            if i < len(sequence) and bbox_token_offset <= sequence[i] < bbox_token_offset + bbox_index_size:
                bbox_tokens.append(sequence[i] - bbox_token_offset)
                i += 1
            else: break
        
        face_tokens = []
        for _ in range(se_tokens_per_element):
            if i < len(sequence) and se_token_offset <= sequence[i] < se_token_offset + se_codebook_size:
                face_tokens.append(sequence[i] - se_token_offset)
                i += 1
            else: break
        
        if i < len(sequence) and face_index_offset <= sequence[i] < face_index_offset + vocab_info['face_index_size']:
            face_idx = sequence[i] - face_index_offset
            i += 1
            if len(bbox_tokens) == bbox_tokens_per_element: face_bboxes.append(bbox_tokens)
            else: print(f"Warning: Face {face_idx} bbox token count mismatch, expected {bbox_tokens_per_element}, got {len(bbox_tokens)}")
            if len(face_tokens) == se_tokens_per_element: faces.append(face_tokens)
            else: print(f"Warning: Face {face_idx} feature token count mismatch, expected {se_tokens_per_element}, got {len(face_tokens)}")
        else:
            if i < len(sequence): print(f"Warning: Expected face index token at position {i}, but found {sequence[i]}")
            i += 1
    
    if i < len(sequence) and sequence[i] == SEP_TOKEN:
        i += 1
    
    # Part 2: Parse edges - Format: face_index face_index bbox_tokens edge_tokens
    while i < len(sequence) and sequence[i] != END_TOKEN:
        if i + 1 < len(sequence) and \
           face_index_offset <= sequence[i] < face_index_offset + vocab_info['face_index_size'] and \
           face_index_offset <= sequence[i+1] < face_index_offset + vocab_info['face_index_size']:
            
            src_face = sequence[i] - face_index_offset
            dst_face = sequence[i+1] - face_index_offset
            i += 2
            edge_face_pairs.append((src_face, dst_face))
            
            bbox_tokens = []
            for _ in range(bbox_tokens_per_element):
                if i < len(sequence) and bbox_token_offset <= sequence[i] < bbox_token_offset + bbox_index_size:
                    bbox_tokens.append(sequence[i] - bbox_token_offset)
                    i += 1
                else: break
            if len(bbox_tokens) == bbox_tokens_per_element: edge_bboxes.append(bbox_tokens)
            else: print(f"Warning: Edge {len(edge_bboxes)} bbox token count mismatch, expected {bbox_tokens_per_element}, got {len(bbox_tokens)}")
            
            # REMOVED: Block for parsing vertex tokens is completely removed.
            
            edge_tokens = []
            for _ in range(se_tokens_per_element):
                if i < len(sequence) and se_token_offset <= sequence[i] < se_token_offset + se_codebook_size:
                    edge_tokens.append(sequence[i] - se_token_offset)
                    i += 1
                else: break
            if len(edge_tokens) == se_tokens_per_element: edges.append(edge_tokens)
            else: print(f"Warning: Edge {len(edges)} feature token count mismatch, expected {se_tokens_per_element}, got {len(edge_tokens)}")
        else:
            if i < len(sequence): print(f"Warning: Expected source/target face index, but found token {sequence[i]} at position {i}")
            i += 1
            
    # print(f"Decoded data: {len(faces)} faces, {len(edges)} edges, {len(face_bboxes)} face bboxes, {len(edge_bboxes)} edge bboxes")
    
    surf_ncs = decode_tokens_to_ncs(faces, se_vqvae_model, 'face', se_tokens_per_element, device) if faces else []
    edge_ncs = decode_tokens_to_ncs(edges, se_vqvae_model, 'edge', se_tokens_per_element, device) if edges else []
    surf_bbox_wcs = dequantize_bbox(np.array(face_bboxes), num_tokens=bbox_index_size).tolist() if face_bboxes else []
    edge_bbox_wcs = dequantize_bbox(np.array(edge_bboxes), num_tokens=bbox_index_size).tolist() if edge_bboxes else []
    
    if scale_factor != 1.0:
        surf_bbox_wcs = [bbox / scale_factor for bbox in surf_bbox_wcs]
        edge_bbox_wcs = [bbox / scale_factor for bbox in edge_bbox_wcs]
    
    return {
        'surf_ncs': surf_ncs,
        'edge_ncs': edge_ncs,
        'surf_bbox_wcs': surf_bbox_wcs,
        'edge_bbox_wcs': edge_bbox_wcs,
        'edgeFace_adj': edge_face_pairs,
        'graph_edges': ( [p[0] for p in edge_face_pairs], [p[1] for p in edge_face_pairs] )
    }

def reconstruct_cad_from_sequence(sequence, vocab_info, se_vqvae_model, device="cpu", scale_factor=1.0, verbose=True):
    """Reconstruct CAD model from autoregressive sequence (no explicit vertex data)"""
    # print("=== Starting CAD Reconstruction from Sequence (No Vertex Data) ===")
    
    try:
        cad_data = parse_sequence_to_cad_data(
            sequence, vocab_info, se_vqvae_model, device, scale_factor
        )
        
        surf_ncs_vqvae = np.array(cad_data['surf_ncs'])
        edge_ncs_vqvae = np.array(cad_data['edge_ncs'])
        surf_bbox_vqvae = np.array(cad_data['surf_bbox_wcs'])
        edge_bbox_vqvae = np.array(cad_data['edge_bbox_wcs'])
        graph_edges = cad_data['graph_edges']

        if len(edge_bbox_vqvae) != len(edge_ncs_vqvae):
            print(f"Edge bbox count ({len(edge_bbox_vqvae)}) does not match NCS data count ({len(edge_ncs_vqvae)}). Cannot continue.")
            return None

        # # 1. Build face-edge adjacency relationship
        # src_nodes, dst_nodes = graph_edges
        # FaceEdgeAdj = []
        # face_adj = [[] for _ in range(len(surf_ncs_vqvae))]
        # for edge_idx, (node1, node2) in enumerate(zip(graph_edges[0], graph_edges[1])):
        #     face_adj[node1].append(edge_idx)
        #     face_adj[node2].append(edge_idx)
        # FaceEdgeAdj.extend(face_adj)
        
        # 1. Intelligently reverse derive index mapping
        src_nodes, dst_nodes = graph_edges
        all_face_ids_in_sequence = sorted(list(set(src_nodes) | set(dst_nodes)))
        num_actual_faces = len(surf_ncs_vqvae)

        # 2. Check if derived ID count matches actual decoded face count
        if len(all_face_ids_in_sequence) != num_actual_faces:
            print(f"Warning: Derived unique face ID count ({len(all_face_ids_in_sequence)}) does not match actual decoded face count ({num_actual_faces}).")
            # If counts don't match, create a truncated mapping
            if len(all_face_ids_in_sequence) > num_actual_faces:
                face_id_to_idx_map = {all_face_ids_in_sequence[i]: i for i in range(num_actual_faces)}
            else:
                 face_id_to_idx_map = {face_id: i for i, face_id in enumerate(all_face_ids_in_sequence)}
        else:
            # Counts match, create complete mapping
            face_id_to_idx_map = {face_id: i for i, face_id in enumerate(all_face_ids_in_sequence)}

        # 3. Use mapping to safely build face-edge adjacency relationship
        FaceEdgeAdj = []
        face_adj = [[] for _ in range(num_actual_faces)]
        
        for edge_idx, (node1_id, node2_id) in enumerate(zip(src_nodes, dst_nodes)):
            if node1_id in face_id_to_idx_map and node2_id in face_id_to_idx_map:
                internal_idx1 = face_id_to_idx_map[node1_id]
                internal_idx2 = face_id_to_idx_map[node2_id]
                
                face_adj[internal_idx1].append(edge_idx)
                face_adj[internal_idx2].append(edge_idx)
            else:
                if verbose:
                    print(f"Warning: Edge {edge_idx} references unknown face indices ({node1_id}, {node2_id}). This edge will be ignored.")
        
        # 4. Assign the built adjacency list
        FaceEdgeAdj.extend(face_adj)

        # Convert NCS edge data to WCS coordinates
        edge_wcs_list = []
        edgeV_bbox = []
        
        for edge_idx in range(len(edge_ncs_vqvae)):
            # try:
            # Extract min/max points from [6] format bounding box
            bbox = edge_bbox_vqvae[edge_idx]  # shape: [6]
            min_point = bbox[:3]  # [min_x, min_y, min_z]
            max_point = bbox[3:]  # [max_x, max_y, max_z]
            
            # Calculate bounding box center and size
            bcenter, bsize = compute_bbox_center_and_size(min_point, max_point)
            
            # Convert normalized NCS curve to WCS coordinate system
            ncs_curve = edge_ncs_vqvae[edge_idx]  # shape: [32, 3], normalized to (-1,1)
            wcs_curve = ncs_curve * (bsize / 2) + bcenter
            edge_wcs_list.append(wcs_curve)
            
            # Extract start and end points as bounding box vertices
            bbox_start_end = wcs_curve[[0, -1]]  # shape: [2, 3]
            edgeV_bbox.append(bbox_start_end)
                
            # except Exception as e:
            #     # Use vertex_wcs as fallback
            #     edgeV_bbox.append(vertex_vqvae[edge_idx])
            #     # Generate a simple straight line as fallback
            #     start, end = vertex_vqvae[edge_idx]
            #     edge_wcs_list.append(np.linspace(start, end, 32))
        
        edgeV_bbox = np.array(edgeV_bbox)  # shape: [num_edges, 2, 3]
        # edge_wcs = np.array(edge_wcs_list)  # shape: [num_edges, 32, 3]

        # Step 1: Improved topology-based vertex detection
        try:
            # Assign global ID to each vertex: edge_idx * 2 + vertex_pos_idx (0 or 1)
            total_vertices = len(edge_ncs_vqvae) * 2
            
            # Use union-find to manage vertex merging
            parent = list(range(total_vertices))
            
            def find(x):
                if parent[x] != x:
                    parent[x] = find(parent[x])
                return parent[x]
            
            def union(x, y):
                px, py = find(x), find(y)
                if px != py:
                    parent[px] = py
            
            if verbose:
                print("  Step 1.1: Processing intra-face vertex merging...")
            # Phase 1: Intra-face vertex detection and merging
            face_merged_groups = []  # Store merged vertex groups within each face
            
            for face_idx, edge_indices in enumerate(FaceEdgeAdj):
                if len(edge_indices) == 0:
                    continue
                    
                # Collect global IDs and positions of all vertices within this face
                face_vertices = []  # [(global_vertex_id, position), ...]
                for edge_idx in edge_indices:
                    for vertex_pos_idx in [0, 1]:  # Start and end points
                        global_vertex_id = edge_idx * 2 + vertex_pos_idx
                        position = edgeV_bbox[edge_idx, vertex_pos_idx]
                        face_vertices.append((global_vertex_id, position))
                
                # Calculate distances between all vertex pairs
                n_vertices = len(face_vertices)
                distance_matrix = np.zeros((n_vertices, n_vertices))
                for i in range(n_vertices):
                    for j in range(i+1, n_vertices):
                        vid1, pos1 = face_vertices[i]
                        vid2, pos2 = face_vertices[j]
                        
                        # Calculate geometric distance
                        distance_matrix[i, j] = distance_matrix[j, i] = np.linalg.norm(pos1 - pos2)
                
                # Greedy algorithm: merge vertices with shortest distance each time
                merged = set()  # Already merged vertex IDs
                face_groups = []  # Store vertex merging groups within this face
                
                while len(merged) < n_vertices:
                    # Find the pair of unmerged vertices with minimum distance
                    min_dist = float('inf')
                    min_i, min_j = -1, -1
                    
                    for i in range(n_vertices):
                        if i in merged:
                            continue
                        
                        for j in range(i+1, n_vertices):
                            if j in merged:
                                continue
                            
                            # Check if they come from the same edge
                            vid_i = face_vertices[i][0]
                            vid_j = face_vertices[j][0]
                            edge_i = vid_i // 2
                            edge_j = vid_j // 2
                            
                            # Skip if they are endpoints of the same edge
                            if edge_i == edge_j:
                                continue
                            
                            if distance_matrix[i, j] < min_dist:
                                min_dist = distance_matrix[i, j]
                                min_i, min_j = i, j
                    
                    # Exit loop if no mergeable vertex pair is found
                    if min_i == -1 or min_j == -1:
                        break
                    
                    # Merge these two vertices
                    vid1, _ = face_vertices[min_i]
                    vid2, _ = face_vertices[min_j]
                    union(vid1, vid2)
                    
                    # Record the merging group within this face
                    face_groups.append([vid1, vid2])
                    
                    # Mark as merged
                    merged.add(min_i)
                    merged.add(min_j)
                
                # Add this face's merging groups to the total list
                face_merged_groups.append(face_groups)
            
            # Statistics of intra-face merging results
            face_groups = {}
            for vid in range(total_vertices):
                root = find(vid)
                if root not in face_groups:
                    face_groups[root] = []
                face_groups[root].append(vid)
            
            merged_groups = [group for group in face_groups.values() if len(group) > 1]
            if verbose:
                print(f"  Intra-face merging results: {len(merged_groups)} merged groups")
            
            if verbose:
                print("  Step 1.2: Processing inter-face vertex merging...")
            # Phase 2: Inter-face vertex merging - check if merged vertex groups within faces can be further merged
            
            # For each pair of faces, check if their merging groups have common vertices
            for i in range(len(face_merged_groups)):
                for j in range(i+1, len(face_merged_groups)):
                    face1_groups = face_merged_groups[i]
                    face2_groups = face_merged_groups[j]
                    
                    # Check if merging groups of two faces have intersection
                    for group1 in face1_groups:
                        for group2 in face2_groups:
                            # Check if two groups have common vertices
                            common_vertices = set(group1) & set(group2)
                            if common_vertices:
                                # If there are common vertices, merge all vertices of these two groups
                                for v1 in group1:
                                    for v2 in group2:
                                        union(v1, v2)
            
            # Statistics of inter-face merging results
            final_groups = {}
            for vid in range(total_vertices):
                root = find(vid)
                if root not in final_groups:
                    final_groups[root] = []
                final_groups[root].append(vid)
            
            merged_final_groups = [group for group in final_groups.values() if len(group) > 1]
            if verbose:
                print(f"  Final merging results: {len(merged_final_groups)} merged groups")
            
            # Generate unique vertices and mapping
            unique_vertices = []
            vertex_mapping = [-1] * total_vertices
            
            # Process all vertex groups (including merged and unmerged)
            for root, group in final_groups.items():
                # Calculate average position of vertices in the group
                group_positions = []
                for vertex_id in group:
                    edge_idx = vertex_id // 2
                    vertex_pos_idx = vertex_id % 2
                    if edge_idx < len(edgeV_bbox):
                        group_positions.append(edgeV_bbox[edge_idx, vertex_pos_idx])
                
                if group_positions:
                    avg_position = np.mean(group_positions, axis=0)
                    unique_vertex_idx = len(unique_vertices)
                    unique_vertices.append(avg_position)
                    
                    # Update mapping
                    for vertex_id in group:
                        vertex_mapping[vertex_id] = unique_vertex_idx
            
            unique_vertices = np.array(unique_vertices)
            
            # Build EdgeVertexAdj
            EdgeVertexAdj = np.zeros((len(edge_ncs_vqvae), 2), dtype=int)
            for edge_idx in range(len(edge_ncs_vqvae)):
                start_global_id = edge_idx * 2
                end_global_id = edge_idx * 2 + 1
                
                # Ensure indices are within valid range
                if start_global_id < len(vertex_mapping) and end_global_id < len(vertex_mapping):
                    start_vertex_idx = vertex_mapping[start_global_id]
                    end_vertex_idx = vertex_mapping[end_global_id]
                    
                    # Ensure mapping is valid
                    if start_vertex_idx >= 0 and end_vertex_idx >= 0:
                        EdgeVertexAdj[edge_idx, 0] = start_vertex_idx
                        EdgeVertexAdj[edge_idx, 1] = end_vertex_idx
                    else:
                        if verbose:
                            print(f"Warning: Invalid vertex mapping for edge {edge_idx} ({start_vertex_idx}, {end_vertex_idx})")
                else:
                    if verbose:
                        print(f"Warning: Global vertex ID out of range for edge {edge_idx}")
            
            if verbose:
                print(f"Found {len(unique_vertices)} unique vertices from {total_vertices} original vertices")
            
            # Validate result reasonableness
            for i, adj in enumerate(EdgeVertexAdj):
                if adj[0] == adj[1]:
                    if verbose:
                        print(f"Warning: Edge {i} has same start and end vertex {adj[0]}")
            
        except Exception as e:
            import traceback
            print(f'Vertex detection failed: {e}')
            traceback.print_exc()
            return None

        try:
            # print("Step 2: Joint Optimization...")
            surf_wcs, edge_wcs = joint_optimize(surf_ncs_vqvae, edge_ncs_vqvae, surf_bbox_vqvae, unique_vertices, EdgeVertexAdj, FaceEdgeAdj, len(edge_ncs_vqvae), len(surf_ncs_vqvae))
        except Exception as e:
            import traceback
            print(f'Joint optimization failed: {e}'); traceback.print_exc(); return None
        
        # print("Step 3: Building B-rep...")
        solid = construct_brep(surf_wcs, edge_wcs, FaceEdgeAdj, EdgeVertexAdj)
        # print("B-rep construction completed")
        
        return solid
        
    except Exception as e:
        import traceback
        print(f"Error during reconstruction: {e}"); traceback.print_exc(); return None

def convert_vqvae_output_to_ncs(reconstructed_tensor, data_type='face'):
    """Convert VQVAE output to NCS format"""
    # reconstructed_tensor: (batch, 3, 32, 32)
    
    if data_type == 'face':
        # Convert face data: (batch, 3, 32, 32) -> (batch, 32, 32, 3)
        faces_ncs = []
        for i in range(reconstructed_tensor.shape[0]):
            face_data = reconstructed_tensor[i].permute(1, 2, 0).cpu().numpy()  # (32, 32, 3)
            faces_ncs.append(face_data)
        return faces_ncs
        
    elif data_type == 'edge':
        # Convert edge data: (batch, 3, 32, 32) -> (batch, 32, 3)
        edges_ncs = []
        for i in range(reconstructed_tensor.shape[0]):
            edge_data = reconstructed_tensor[i].permute(1, 2, 0).cpu().numpy()  # (32, 32, 3)
            # Calculate mean of all rows as edge sampling points
            edge_curve = np.mean(edge_data, axis=1)  # (32, 3)
            edges_ncs.append(edge_curve)
        return edges_ncs
    else:
        raise ValueError(f"Unknown data_type: {data_type}")

def create_bspline_curve(ctrs):

    assert ctrs.shape[0] == 4

    poles = TColgp_Array1OfPnt(1, 4)
    for i, ctr in enumerate(ctrs, 1):
        poles.SetValue(i, gp_Pnt(*ctr))

    n_knots = 2
    knots = TColStd_Array1OfReal(1, n_knots)
    knots.SetValue(1, 0.0)
    knots.SetValue(2, 1.0)

    mults = TColStd_Array1OfInteger(1, n_knots)
    mults.SetValue(1, 4)
    mults.SetValue(2, 4)

    bspline_curve = Geom_BSplineCurve(poles, knots, mults, 3)

    return bspline_curve

def create_bspline_surface(ctrs):

    assert ctrs.shape[0] == 16

    poles = TColgp_Array2OfPnt(1, 4, 1, 4)
    for i in range(4):
        for j in range(4):
            idx = i * 4 + j
            poles.SetValue(i + 1, j + 1, gp_Pnt(*ctrs[idx]))

    u_knots = TColStd_Array1OfReal(1, 2)
    v_knots = TColStd_Array1OfReal(1, 2)

    u_knots.SetValue(1, 0.0)
    u_knots.SetValue(2, 1.0)
    v_knots.SetValue(1, 0.0)
    v_knots.SetValue(2, 1.0)

    u_mults = TColStd_Array1OfInteger(1, 2)
    v_mults = TColStd_Array1OfInteger(1, 2)

    u_mults.SetValue(1, 4)
    u_mults.SetValue(2, 4)
    v_mults.SetValue(1, 4)
    v_mults.SetValue(2, 4)

    bspline_surface = Geom_BSplineSurface(poles, u_knots, v_knots, u_mults, v_mults, 3, 3)

    return bspline_surface

def sample_bspline_curve(bspline_curve, num_points=32):
    u_start, u_end = bspline_curve.FirstParameter(), bspline_curve.LastParameter()
    u_range = np.linspace(u_start, u_end, num_points)

    points = np.zeros((num_points, 3))

    for i, u in enumerate(u_range):
        pnt = bspline_curve.Value(u)
        points[i] = [pnt.X(), pnt.Y(), pnt.Z()]

    return points    # 32*3

def sample_bspline_surface(bspline_surface, num_u=32, num_v=32):
    u_start, u_end, v_start, v_end = bspline_surface.Bounds()
    u_range = np.linspace(u_start, u_end, num_u)
    v_range = np.linspace(v_start, v_end, num_v)

    points = np.zeros((num_u, num_v, 3))

    for i, u in enumerate(u_range):
        for j, v in enumerate(v_range):
            pnt = bspline_surface.Value(u, v)
            points[i, j] = [pnt.X(), pnt.Y(), pnt.Z()]

    return points      # 32*32*3

### B-rep Post-processing ###

def compute_bbox_center_and_size(min_corner, max_corner):
    # Calculate the center
    center_x = (min_corner[0] + max_corner[0]) / 2
    center_y = (min_corner[1] + max_corner[1]) / 2
    center_z = (min_corner[2] + max_corner[2]) / 2
    center = np.array([center_x, center_y, center_z])
    # Calculate the size
    size_x = max_corner[0] - min_corner[0]
    size_y = max_corner[1] - min_corner[1]
    size_z = max_corner[2] - min_corner[2]
    size = max(size_x, size_y, size_z)
    return center, size

class STModel(nn.Module):
    def __init__(self, num_edge, num_surf):
        super().__init__()
        self.edge_t = nn.Parameter(torch.zeros((num_edge, 3)))
        self.surf_st = nn.Parameter(torch.FloatTensor([1, 0, 0, 0]).unsqueeze(0).repeat(num_surf, 1))

def get_bbox_minmax(point_cloud):
    # Find the minimum and maximum coordinates along each axis
    min_x = np.min(point_cloud[:, 0])
    max_x = np.max(point_cloud[:, 0])

    min_y = np.min(point_cloud[:, 1])
    max_y = np.max(point_cloud[:, 1])

    min_z = np.min(point_cloud[:, 2])
    max_z = np.max(point_cloud[:, 2])

    # Create the 3D bounding box using the min and max values
    min_point = np.array([min_x, min_y, min_z])
    max_point = np.array([max_x, max_y, max_z])
    return (min_point, max_point)

def edge2loop(face_edges):
    face_edges_flatten = face_edges.reshape(-1,3)     
    # connect end points by closest distance
    merged_vertex_id = []
    
    print(f"  edge2loop: Processing {len(face_edges)} edges")
    
    for edge_idx, startend in enumerate(face_edges):
        self_id = [2*edge_idx, 2*edge_idx+1]
        # left endpoint 
        distance = np.linalg.norm(face_edges_flatten - startend[0], axis=1)
        min_id = list(np.argsort(distance))
        min_id_noself = [x for x in min_id if x not in self_id]
        if len(min_id_noself) > 0:
            merged_vertex_id.append(min_id_noself[0])
            print(f"    Edge {edge_idx} start point {2*edge_idx} merged with vertex {min_id_noself[0]}")
        else:
            print(f"    Edge {edge_idx} start point {2*edge_idx} found no merge target")
            
        # right endpoint
        distance = np.linalg.norm(face_edges_flatten - startend[1], axis=1)
        min_id = list(np.argsort(distance))
        min_id_noself = [x for x in min_id if x not in self_id]
        if len(min_id_noself) > 0:
            merged_vertex_id.append(min_id_noself[0])
            print(f"    Edge {edge_idx} end point {2*edge_idx+1} merged with vertex {min_id_noself[0]}")
        else:
            print(f"    Edge {edge_idx} end point {2*edge_idx+1} found no merge target")

    if len(merged_vertex_id) == 0:
        print(f"  edge2loop: No merge relationships found")
        return np.array([])
    
    merged_vertex_id = np.unique(np.array(merged_vertex_id))
    print(f"  edge2loop: Found {len(merged_vertex_id)} unique merged vertices: {merged_vertex_id}")
    return merged_vertex_id

def joint_optimize(surf_ncs, edge_ncs, surfPos, unique_vertices, EdgeVertexAdj, FaceEdgeAdj, num_edge, num_surf):
    """
    Jointly optimize the face/edge/vertex based on topology
    """
    loss_func = ChamferDistance()

    model = STModel(num_edge, num_surf)
    model = model.cuda().train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.95, 0.999),
        weight_decay=1e-6,
        eps=1e-08,
    )

    # Optimize edges (directly compute)
    edge_ncs_se = edge_ncs[:,[0,-1]]
    edge_vertex_se = unique_vertices[EdgeVertexAdj]

    edge_wcs = []
    # print('Joint Optimization...')
    for wcs, ncs_se, vertex_se in zip(edge_ncs, edge_ncs_se, edge_vertex_se):
        # scale
        scale_target = np.linalg.norm(vertex_se[0] - vertex_se[1])
        scale_ncs = np.linalg.norm(ncs_se[0] - ncs_se[1])
        edge_scale = scale_target / scale_ncs

        edge_updated = wcs*edge_scale
        edge_se = ncs_se*edge_scale  

        # offset
        offset = (vertex_se - edge_se)
        offset_rev = (vertex_se - edge_se[::-1])

        # swap start / end if necessary 
        offset_error = np.abs(offset[0] - offset[1]).mean()
        offset_rev_error =np.abs(offset_rev[0] - offset_rev[1]).mean()
        if offset_rev_error < offset_error:
            edge_updated = edge_updated[::-1]
            offset = offset_rev
    
        edge_updated = edge_updated + offset.mean(0)[np.newaxis,np.newaxis,:]
        edge_wcs.append(edge_updated)

    edge_wcs = np.vstack(edge_wcs)

    # Replace start/end points with corner, and backprop change along curve
    for index in range(len(edge_wcs)):
        start_vec = edge_vertex_se[index,0] - edge_wcs[index, 0]
        end_vec = edge_vertex_se[index,1] - edge_wcs[index, -1]
        weight = np.tile((np.arange(32)/31)[:,np.newaxis], (1,3))
        weighted_vec = np.tile(start_vec[np.newaxis,:],(32,1))*(1-weight) + np.tile(end_vec,(32,1))*weight
        edge_wcs[index] += weighted_vec            

    # Optimize surfaces 
    face_edges = []
    for adj in FaceEdgeAdj:
        all_pnts = edge_wcs[adj]
        face_edges.append(torch.FloatTensor(all_pnts).cuda())

    # Initialize surface in wcs based on surface pos
    surf_wcs_init = [] 
    bbox_threshold_min = []
    bbox_threshold_max = []   
    for edges_perface, ncs, bbox in zip(face_edges, surf_ncs, surfPos):
        surf_center, surf_scale = compute_bbox_center_and_size(bbox[0:3], bbox[3:])
        edges_perface_flat = edges_perface.reshape(-1, 3).detach().cpu().numpy()
        min_point, max_point = get_bbox_minmax(edges_perface_flat)
        edge_center, edge_scale = compute_bbox_center_and_size(min_point, max_point)
        bbox_threshold_min.append(min_point)
        bbox_threshold_max.append(max_point)

        # increase surface size if does not fully cover the wire bbox 
        if surf_scale < edge_scale:
            surf_scale = 1.05*edge_scale
    
        wcs = ncs * (surf_scale/2) + surf_center
        surf_wcs_init.append(wcs)

    surf_wcs_init = np.stack(surf_wcs_init)

    # optimize the surface offset
    surf = torch.FloatTensor(surf_wcs_init).cuda()
    for iters in range(200):
        surf_scale = model.surf_st[:,0].reshape(-1,1,1,1)
        surf_offset = model.surf_st[:,1:].reshape(-1,1,1,3)
        surf_updated = surf + surf_offset 
        
        surf_loss = 0
        for surf_pnt, edge_pnts in zip(surf_updated, face_edges):
            surf_pnt = surf_pnt.reshape(-1,3)
            edge_pnts = edge_pnts.reshape(-1,3).detach()
            surf_loss += loss_func(surf_pnt.unsqueeze(0), edge_pnts.unsqueeze(0), bidirectional=False, reverse=True) 
        surf_loss /= len(surf_updated) 

        optimizer.zero_grad()
        (surf_loss).backward()
        optimizer.step()

        # print(f'Iter {iters} surf:{surf_loss:.5f}') 

    surf_wcs = surf_updated.detach().cpu().numpy()

    return (surf_wcs, edge_wcs)

def add_pcurves_to_edges(face):
    edge_fixer = ShapeFix_Edge()
    top_exp = TopologyExplorer(face)
    for wire in top_exp.wires():
        wire_exp = WireExplorer(wire)
        for edge in wire_exp.ordered_edges():
            edge_fixer.FixAddPCurve(edge, face, False, 0.001)

def fix_wires(face, debug=False):
    top_exp = TopologyExplorer(face)
    for wire in top_exp.wires():
        if debug:
            wire_checker = ShapeAnalysis_Wire(wire, face, 0.01)
            print(f"Check order 3d {wire_checker.CheckOrder()}")
            print(f"Check 3d gaps {wire_checker.CheckGaps3d()}")
            print(f"Check closed {wire_checker.CheckClosed()}")
            print(f"Check connected {wire_checker.CheckConnected()}")
        wire_fixer = ShapeFix_Wire(wire, face, 0.01)

        # wire_fixer.SetClosedWireMode(True)
        # wire_fixer.SetFixConnectedMode(True)
        # wire_fixer.SetFixSeamMode(True)

        assert wire_fixer.IsReady()
        ok = wire_fixer.Perform()
        # assert ok

def fix_face(face):
    fixer = ShapeFix_Face(face)
    fixer.SetPrecision(0.01)
    fixer.SetMaxTolerance(0.1)
    ok = fixer.Perform()
    # assert ok
    fixer.FixOrientation()
    face = fixer.Face()
    return face

def get_bbox_norm(point_cloud):
    # Find the minimum and maximum coordinates along each axis
    min_x = np.min(point_cloud[:, 0])
    max_x = np.max(point_cloud[:, 0])

    min_y = np.min(point_cloud[:, 1])
    max_y = np.max(point_cloud[:, 1])

    min_z = np.min(point_cloud[:, 2])
    max_z = np.max(point_cloud[:, 2])

    # Create the 3D bounding box using the min and max values
    min_point = np.array([min_x, min_y, min_z])
    max_point = np.array([max_x, max_y, max_z])
    return np.linalg.norm(max_point - min_point)

def construct_brep(surf_wcs, edge_wcs, FaceEdgeAdj, EdgeVertexAdj):
    """
    Fit parametric surfaces / curves and trim into B-rep
    """
    # print('Building the B-rep...')
    # Fit surface bspline
    recon_faces = []  
    for points in surf_wcs:
        num_u_points, num_v_points = 32, 32
        uv_points_array = TColgp_Array2OfPnt(1, num_u_points, 1, num_v_points)
        for u_index in range(1,num_u_points+1):
            for v_index in range(1,num_v_points+1):
                pt = points[u_index-1, v_index-1]
                point_3d = gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2]))
                uv_points_array.SetValue(u_index, v_index, point_3d)
        approx_face =  GeomAPI_PointsToBSplineSurface(uv_points_array, 3, 8, GeomAbs_C2, 5e-2).Surface() 
        recon_faces.append(approx_face)

    recon_edges = []
    for points in edge_wcs:
        num_u_points = 32
        u_points_array = TColgp_Array1OfPnt(1, num_u_points)
        for u_index in range(1,num_u_points+1):
            pt = points[u_index-1]
            point_2d = gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2]))
            u_points_array.SetValue(u_index, point_2d)
        try:
            approx_edge = GeomAPI_PointsToBSpline(u_points_array, 0, 8, GeomAbs_C2, 5e-3).Curve()  
        except Exception as e:
            print('high precision failed, trying mid precision...')
            try:
                approx_edge = GeomAPI_PointsToBSpline(u_points_array, 0, 8, GeomAbs_C2, 8e-3).Curve()  
            except Exception as e:
                print('mid precision failed, trying low precision...')
                approx_edge = GeomAPI_PointsToBSpline(u_points_array, 0, 8, GeomAbs_C2, 5e-2).Curve()
        recon_edges.append(approx_edge)

    # Create edges from the curve list
    edge_list = []
    for curve in recon_edges:
        edge = BRepBuilderAPI_MakeEdge(curve).Edge()
        edge_list.append(edge)

    # Cut surface by wire 
    post_faces = []
    post_edges = []
    for idx,(surface, edge_incides) in enumerate(zip(recon_faces, FaceEdgeAdj)):
        corner_indices = EdgeVertexAdj[edge_incides]
        
        # ordered loop
        loops = []
        ordered = [0]
        seen_corners = [corner_indices[0,0], corner_indices[0,1]]
        next_index = corner_indices[0,1]

        while len(ordered)<len(corner_indices):
            while True:
                next_row = [idx for idx, edge in enumerate(corner_indices) if next_index in edge and idx not in ordered]
                if len(next_row) == 0:
                    break
                ordered += next_row
                next_index = list(set(corner_indices[next_row][0]) - set(seen_corners))
                if len(next_index)==0:break
                else: next_index = next_index[0]
                seen_corners += [corner_indices[next_row][0][0], corner_indices[next_row][0][1]]
            
            cur_len = int(np.array([len(x) for x in loops]).sum()) # add to inner / outer loops
            loops.append(ordered[cur_len:])
            
            # Swith to next loop
            next_corner =  list(set(np.arange(len(corner_indices))) - set(ordered))
            if len(next_corner)==0:break
            else: next_corner = next_corner[0]
            next_index = corner_indices[next_corner][0]
            ordered += [next_corner]
            seen_corners += [corner_indices[next_corner][0], corner_indices[next_corner][1]]
            next_index = corner_indices[next_corner][1]

        # Determine the outer loop by bounding box length (?)
        bbox_spans = [get_bbox_norm(edge_wcs[x].reshape(-1,3)) for x in loops]
        
        # Create wire from ordered edges
        _edge_incides_ = [edge_incides[x] for x in ordered]
        edge_post = [edge_list[x] for x in _edge_incides_]
        post_edges += edge_post

        out_idx = np.argmax(np.array(bbox_spans))
        inner_idx = list(set(np.arange(len(loops))) - set([out_idx]))

        # Outer wire
        wire_builder = BRepBuilderAPI_MakeWire()
        for edge_idx in loops[out_idx]:
            wire_builder.Add(edge_list[edge_incides[edge_idx]])
        outer_wire = wire_builder.Wire()

        # Inner wires
        inner_wires = []
        for idx in inner_idx:
            wire_builder = BRepBuilderAPI_MakeWire()
            for edge_idx in loops[idx]:
                wire_builder.Add(edge_list[edge_incides[edge_idx]])
            inner_wires.append(wire_builder.Wire())
    
        # Cut by wires
        face_builder = BRepBuilderAPI_MakeFace(surface, outer_wire)
        for wire in inner_wires:
            face_builder.Add(wire)
        face_occ = face_builder.Shape()
        fix_wires(face_occ)
        add_pcurves_to_edges(face_occ)
        fix_wires(face_occ)
        face_occ = fix_face(face_occ)
        post_faces.append(face_occ)

    # Sew faces into solid 
    sewing = BRepBuilderAPI_Sewing()
    sewing.SetTolerance(1e-3)  # Set tolerance to 1e-3
    for face in post_faces:
        sewing.Add(face)
        
    # Perform the sewing operation
    sewing.Perform()
    sewn_shell = sewing.SewedShape()

    # Make a solid from the shell
    maker = BRepBuilderAPI_MakeSolid()
    maker.Add(sewn_shell)
    maker.Build()
    solid = maker.Solid()

    return solid
