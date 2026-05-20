import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # Allow shell scripts to override this value.
root = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../")
sys.path.append(root)
import pickle
import logging
import copy
import numpy as np
import traceback
from collections import OrderedDict
import torch
import glob
import matplotlib.pyplot as plt
import torch.nn.functional as F
import re
import time
import random

from mvp.config import data_root
from mvp.data.util import bbox_sensor_to_map, bbox_map_to_sensor, pcd_sensor_to_map, pcd_map_to_sensor, get_distance
from mvp.tools.iou import iou2d
from mvp.tools.polygon_space import bbox_to_polygon
from mvp.tools.squeezeseg.interface import SqueezeSegInterface
from mvp.defense.detection_util import filter_segmentation
from mvp.tools.lidar_seg import lidar_segmentation
from mvp.tools.ground_detection import get_ground_plane
from mvp.tools.polygon_space import get_occupied_space, get_free_space, bbox_to_polygon
from mvp.visualize.attack import draw_attack
from mvp.visualize.defense import visualize_defense, draw_roc
from mvp.data.opv2v_dataset import OPV2VDataset
from mvp.perception.opencood_perception import OpencoodPerception
from mvp.attack.lidar_spoof_early_attacker import LidarSpoofEarlyAttacker
from mvp.attack.lidar_spoof_intermediate_attacker import LidarSpoofIntermediateAttacker
from mvp.attack.lidar_spoof_late_attacker import LidarSpoofLateAttacker
from mvp.attack.lidar_remove_early_attacker import LidarRemoveEarlyAttacker
from mvp.attack.lidar_remove_intermediate_attacker import LidarRemoveIntermediateAttacker
from mvp.attack.lidar_remove_late_attacker import LidarRemoveLateAttacker
from mvp.defense.perception_defender import CADDefender, ROBOSACDefender, CPGuardDefender, PerceptionDefender, GCPDefender
from scripts.train_ta_mvig_attack import OccupancyDataProcessor, MVIGNet, pickle_cache_load, pickle_cache_dump

result_dir = os.path.normpath(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../result"))
os.makedirs(result_dir, exist_ok=True)


attack_mode = os.getenv("MVIG_EVAL_ATTACK_MODE", "BAC")  # 'RC' or 'BAC' or 'BASIC' or 'RC+'
persistence = int(os.getenv("MVIG_EVAL_PERSISTENCE", "0"))
attack_persist = True if persistence > 0 else False
is_visualize = os.getenv("MVIG_EVAL_VISUALIZE", "false").lower() in {"1", "true", "yes", "y"}
eval_cache_size = int(os.getenv("MVIG_EVAL_CACHE_SIZE", "10"))
model_path = os.getenv("MVIG_EVAL_MODEL_PATH", "checkpoints/best_mvig_model_spoof_20.pth")
requested_defenders = [x.strip().lower() for x in os.getenv("MVIG_EVAL_DEFENSES", "CAD").split(",") if x.strip()]
eval_attack_type = os.getenv("MVIG_EVAL_ATTACK_TYPE", "spoof").strip().lower()

attack_frame_ids = list(range(9-persistence, 10))
total_frames = 10

logging.basicConfig(filename=os.path.join(result_dir, "evaluate.log"), filemode="a", level=logging.INFO)

dataset = OPV2VDataset(root_path=os.path.join(data_root, "OPV2V"), mode="test")
dataset.cache_size = eval_cache_size  # Use cache_size to control the dataset size; use cache size 21 for the frame-20 test setup.
fixed_attack_ids = None

perception_list = [
    OpencoodPerception(fusion_method="early", model_name="pointpillar"),
    OpencoodPerception(fusion_method="intermediate", model_name="pointpillar"),
    OpencoodPerception(fusion_method="late", model_name="pointpillar"),
]
perception_dict = OrderedDict([(x.name, x) for x in perception_list])

if eval_attack_type == "remove":
    attacker_list = [
        LidarRemoveIntermediateAttacker(perception_dict["pointpillar_intermediate"], dataset, step=100, sync=0, init=False, online=False),
    ]
else:
    attacker_list = [
        LidarSpoofIntermediateAttacker(perception_dict["pointpillar_intermediate"], dataset, step=100, sync=0, init=False, online=False),
    ]
attacker_dict = OrderedDict([(x.name, x) for x in attacker_list])

available_defenders = OrderedDict([
    ("cad", CADDefender()),  # Original occupancy-grid conflict detection method
    ("robosac", ROBOSACDefender(difference_threshold=0.35)),
    ("cpguard", CPGuardDefender(difference_threshold=0.33)),
    ("gcp", GCPDefender(difference_threshold=0.33)),
])
defender_list = [defender for name, defender in available_defenders.items() if name in requested_defenders]
if not defender_list:
    raise ValueError(f"No valid defenders selected from MVIG_EVAL_DEFENSES={requested_defenders}")
defender_dict = OrderedDict([(x.name, x) for x in defender_list])

pickle_cache = OrderedDict()
pickle_cache_size = 600

def pickle_cache_load(file_path):
    file_path = os.path.normpath(file_path)
    if file_path in pickle_cache:
        return pickle_cache[file_path]
    else:
        data = pickle.load(open(file_path, 'rb'))
        if len(pickle_cache) >= pickle_cache_size:
            pickle_cache.popitem(last=False)
        pickle_cache[file_path] = data
        return data
    

def pickle_cache_dump(data, file_path):
    file_path = os.path.normpath(file_path)
    if file_path in pickle_cache:
        pickle_cache[file_path] = data
    pickle.dump(data, open(file_path, 'wb'))


def normal_case_iterator(f):
    def wrapper(*args, **kwargs):
        max_cases = dataset.cache_size  # Use the same limit as dataset cache
        case_count = 0
        
        for case_id, case in dataset.case_generator(tag="multi_frame", index=True, use_lidar=True, use_camera=False):
            if case_count >= max_cases:
                break
                
            data_dir = os.path.join(result_dir, "normal/{:06d}".format(case_id))
            os.makedirs(data_dir, exist_ok=True)

            kwargs.update({
                "case_id": case_id,
                "case": case,
                "data_dir": data_dir,
            })
            f(*args, **kwargs)
            case_count += 1
    return wrapper


def attack_case_iterator(f):
    def wrapper(*args, **kwargs):
        attacker = args[0]
        max_attacks = dataset.cache_size  # Use the same limit as dataset cache
        
        # # If max_attacks is smaller than 100, randomly sample the same number of attack IDs
        # if max_attacks < 100:
        #     # Generate random indices in [0, 99]
        #     sampled_attack_ids = random.sample(range(100), max_attacks)
        #     logging.info(f"Randomly sampled {max_attacks} attacks from 100 possible attacks: {sorted(sampled_attack_ids)}")
        # else:
        #     # If max_attacks >= 100, process all attacks
        #     sampled_attack_ids = None
        
        for attack_id, attack in enumerate(attacker.attack_list):
           
            ###----- Used for fixed-frame testing----------
            if fixed_attack_ids is not None and attack_id not in fixed_attack_ids:
                continue
            ###---------------------------

            # # Skip the current attack if random sampling is enabled and the ID is not in the sampled list
            # if sampled_attack_ids is not None and attack_id not in sampled_attack_ids:
            #     continue
            
            if attack_id >= max_attacks:
                break
                
            data_dir = os.path.join(result_dir, "attack/{}/{:06d}".format(attacker.name, attack_id))
            os.makedirs(data_dir, exist_ok=True)
            case_id = attack["attack_meta"]["case_id"]
            case = dataset.get_case(case_id, tag="multi_frame", use_lidar=True, use_camera=False)

            kwargs.update({
                "case_id": case_id,
                "case": case,
                "data_dir": data_dir,
                "attack_id": attack_id,
                "attack": attack,
            })
            f(*args, **kwargs)
    return wrapper


@normal_case_iterator
def normal_perception(case_id=None, case=None, data_dir=None):
    for perception_name, perception in perception_dict.items():
        save_file = os.path.join(data_dir, "{}.pkl".format(perception_name))
        if os.path.isfile(save_file):
            logging.info("perception {} on normal case {} already exists".format(perception.name, case_id))
            continue
        else:
            logging.info("Processing perception {} on normal case {}".format(perception.name, case_id))

        perception_feature = [{} for _ in range(total_frames)]
        for frame_id in list(range(total_frames)):
            for vehicle_id in list(case[frame_id].keys()):
                pred_bboxes, pred_scores = perception.run(case[frame_id], ego_id=vehicle_id)
                perception_feature[frame_id][vehicle_id] = {"pred_bboxes": pred_bboxes, "pred_scores": pred_scores}

        pickle_cache_dump(perception_feature, save_file)
        

@attack_case_iterator
def attack_perception(attacker, case_id=None, case=None, data_dir=None, attack_id=None, attack=None):
    # Check whether the MVIG model is used
    is_mvig = hasattr(attacker, 'mvig_model') and attacker.mvig_model is not None
    
    # Check whether random-baseline mode is active
    is_random_baseline = is_mvig and hasattr(attacker.mvig_model, 'random_baseline') and attacker.mvig_model.random_baseline
    
    # Set the save suffix according to the model type
    if is_random_baseline:
        save_suffix = "_random"
    elif is_mvig:
        save_suffix = "_mvig"
    else:
        save_suffix = ""
    
    # Determine whether persistent-attack mode is enabled
    use_persistent_attack = attack_persist and is_mvig
    
    # Set the attack-frame range
    if use_persistent_attack:
        # Persistent-attack mode: from initial_frame to frame 9
        initial_frame = 9-persistence  # First attacked frame, used for MVIG prediction
        all_target_frames = list(range(initial_frame, 10))
        logging.info(f"Using persistent attack mode from frame {all_target_frames[0]} to {all_target_frames[-1]}")
    else:
        # Single-frame attack mode: attack only frame 9
        all_target_frames = [9]
        initial_frame = 9
    
    # Record attack information for all frames
    all_frames_attack_info = [{} for _ in range(total_frames)]
    
    logging.info(f"Processing {'persistent ' if use_persistent_attack else ''}{'Random-baseline' if is_random_baseline else 'MVIG-optimized' if is_mvig else 'baseline'} attack {attacker.name} and attack case {attack_id}")

    attack_opts = attack["attack_opts"]
    attack_opts["victim_vehicle_id"] = attack["attack_meta"]["victim_vehicle_id"]
    attack_opts["attack_mode"] = attack_mode
    
    # Update attack-frame information in attack_meta
    attack["attack_meta"]["attack_frame_ids"] = all_target_frames
    attack["attack_meta"]["is_persistent_attack"] = use_persistent_attack
    
    # Pre-compute attack positions for all frames when MVIG is used
    all_frames_positions = None
    
    if attack_mode in ["RC", "RC+"] and is_mvig:
        with torch.no_grad():  # Make sure gradients are not computed here
            try:
                # Prepare sequence data
                history_length = 5
                # Use only historical frames to predict the target frame
                if use_persistent_attack:
                    # For persistent attacks, use the previous 5 frames to predict frame 5
                    history_frames = list(range(max(0, initial_frame - history_length + 1), initial_frame))
                else:
                    # For single-frame attacks, use the previous 5 frames to predict frame 9
                    history_frames = list(range(max(0, initial_frame - history_length + 1), initial_frame))
                    
                # Get the case ID
                case_id = attack["attack_meta"]["case_id"]
                
                # Create sequence data following the format used in train_ta_mvig_attack.py
                sequence = {
                    'case': case,
                    'attack': attack,
                    'meta': attack["attack_meta"],
                    'case_id': case_id,
                    'sequence_id': attack_id,
                    'target_idx': len(history_frames) - 1
                }
                
                # Use the same path layout as train_ta_mvig_attack.py
                occupancy_path = os.path.join(result_dir, f"normal/{case_id:06d}/occupancy_map.pkl")
                
                if os.path.exists(occupancy_path):
                    # Load occupancy features
                    occupancy_feature = pickle_cache_load(occupancy_path)
                    sequence['occupancy_feature'] = occupancy_feature
                    
                    # Initialize the processor
                    processor = OccupancyDataProcessor()
                    
                    # Get the map name
                    map_name = case[0][list(case[0].keys())[0]].get("map", "unknown")
                
                  

                    # Prepare temporal data
                    temporal_graphs = processor.prepare_temporal_data(
                        occupancy_feature,
                        frame_ids=history_frames,
                        map_name=map_name
                    )
                    
                    if temporal_graphs:
                        # Start timing before executing the code
                        start_time = time.time()
                        # Use the MVIG model to predict the target-frame position
                        pred_box, grid_map = attacker.mvig_model(temporal_graphs)

                        # Compute and log the execution time
                        execution_time = time.time() - start_time
                        logging.info(f"TIMING: attacker.mvig_model execution took {execution_time:.4f} seconds")
                        
                        # Visualize the grid map only for non-random-baseline mode
                        if not is_random_baseline and is_visualize:
                            try:
                                # Move grid_map from GPU to CPU and convert it to a NumPy array
                                grid_map_np = grid_map.detach().cpu().numpy()
                                
                                # Handle either single-frame or batched data
                                if len(grid_map_np.shape) >= 3:  # [B, H, W]
                                    batch_size = grid_map_np.shape[0]
                                    for b in range(batch_size):
                                        # Create a semantic visualization
                                        create_semantic_visualization(
                                            grid_map_np[b], 
                                            pred_box[b] if pred_box is not None else None,
                                            case, 
                                            map_name=temporal_graphs[0].get('map_name'),
                                            data_dir=data_dir,
                                            suffix=f"_{b}{save_suffix}"
                                        )
                                else:
                                    # Handle a single image
                                    create_semantic_visualization(
                                        grid_map_np,
                                        pred_box if pred_box is not None else None,
                                        case,
                                        map_name=temporal_graphs[0].get('map_name'),
                                        data_dir=data_dir,
                                        suffix=save_suffix
                                    )
                                
                            except Exception as e:
                                logging.error(f"Error visualizing grid map: {str(e)}\nTraceback:\n{traceback.format_exc()}")
                        
                        # Process the position using the same logic as train_ta_mvig_attack.py
                        # Distinguish between spoofing and removal attacks
                        if "remove" in attacker.name:
                            try:
                                # Get the device from the model parameters
                                device = next(attacker.mvig_model.parameters()).device
                                
                                # Initialize position_tensor as a clone of pred_box
                                position_tensor = pred_box.clone()
                                
                                # Get the original predictions before the attack
                                original_pred_bboxes = []
                                original_pred_scores = []
                                
                                # Query the perception model directly for the original predictions
                                if hasattr(attacker, 'perception') and attacker.perception:
                                    orig_bboxes, orig_scores = attacker.perception.run(
                                    case[initial_frame], 
                                        ego_id=attack_opts["victim_vehicle_id"]
                                    )
                                    original_pred_bboxes = orig_bboxes
                                    original_pred_scores = orig_scores

                                if len(original_pred_bboxes) > 0:
                                    # Convert position_tensor to map coordinates first, then to the victim frame
                                    position_map = bbox_sensor_to_map(
                                        position_tensor.detach().cpu().numpy()[0],
                                    case[initial_frame][attack_opts["attacker_vehicle_id"]]["lidar_pose"]
                                    )
                                    position_victim = bbox_map_to_sensor(
                                        position_map,
                                    case[initial_frame][attack_opts["victim_vehicle_id"]]["lidar_pose"]
                                    )

                                    # Find the closest box in original_pred_bboxes
                                    min_dist = float('inf')
                                    bbox_to_remove = None
                                    
                                    for bbox in original_pred_bboxes:
                                        # Compute Euclidean distance using only x and y
                                        dist = np.sqrt(
                                            (position_victim[0] - bbox[0])**2 + 
                                            (position_victim[1] - bbox[1])**2
                                        )
                                        if dist < min_dist:
                                            min_dist = dist
                                            bbox_to_remove = bbox

                                    if bbox_to_remove is not None:
                                        # Convert the selected box to map coordinates first, then to the attacker frame
                                        bbox_map = bbox_sensor_to_map(
                                            bbox_to_remove,
                                        case[initial_frame][attack_opts["victim_vehicle_id"]]["lidar_pose"]
                                        )
                                        bbox_attacker = bbox_map_to_sensor(
                                            bbox_map,
                                        case[initial_frame][attack_opts["attacker_vehicle_id"]]["lidar_pose"]
                                        )
                                        
                                        # Update position_tensor
                                        position_tensor = torch.tensor(
                                            bbox_attacker,
                                            device=device,
                                            dtype=torch.float32
                                        ).unsqueeze(0)
                                        
                            except Exception as e:
                                logging.error(f"Error in removal attack processing: {str(e)}")
                                # Ensure a valid position_tensor still exists when an error occurs
                                if 'position_tensor' not in locals():
                                    position_tensor = pred_box.clone()
                        else:
                            # Spoofing attacks use the original pred_box
                            position_tensor = pred_box.detach()

                        # Convert to a NumPy array for downstream processing
                        position_np = position_tensor.cpu().numpy()
                        
                        # If persistent-attack mode is enabled, compute attack positions for all frames
                        if use_persistent_attack:
                            if attack_mode == "RC" or attack_mode == "RC+":
                                all_frames_positions = []
                                # Get the initial position in map coordinates
                                initial_position_map = bbox_sensor_to_map(
                                    position_np[0],
                                    case[initial_frame][attack_opts["attacker_vehicle_id"]]["lidar_pose"]
                                )
                                current_position_map = initial_position_map.copy()

                                # frame id 20 adjustment
                                # current_position_map[0] += 3
                                # current_position_map[1] += 25
                                
                                # Compute the attack position for each frame
                                for frame_idx in all_target_frames:

                                    # frame id 20 adjustment
                                    # if frame_idx == all_target_frames[-2]:
                                    #    attack_interrupted = True

                                    if frame_idx != all_target_frames[0]:
                                        if is_random_baseline:
                                            # Random-baseline mode: generate a coherent random trajectory
                                            victim_id = attack_opts["victim_vehicle_id"]
                                            victim_pose = case[frame_idx][victim_id]["lidar_pose"]
                                            current_position_victim = bbox_map_to_sensor(current_position_map, victim_pose)
                                            
                                            # Compute the current velocity if this is not the first move
                                            current_velocity = np.zeros(2)
                                            if frame_idx > all_target_frames[1]:  # At least two historical frames are available
                                                prev_frame_idx = all_target_frames[all_target_frames.index(frame_idx) - 1]
                                                prev_position_map = bbox_sensor_to_map(
                                                    all_frames_positions[-1],  # Position from the previous frame
                                                    case[prev_frame_idx][attack_opts["attacker_vehicle_id"]]["lidar_pose"]
                                                )
                                                # Compute the velocity vector (x, y)
                                                current_velocity[0] = current_position_map[0] - prev_position_map[0]
                                                current_velocity[1] = current_position_map[1] - prev_position_map[1]
                                            
                                            # If no clear velocity exists, assign a random direction
                                            if np.linalg.norm(current_velocity) < 0.1:
                                                # Generate a random direction
                                                random_angle = np.random.uniform(0, 2 * np.pi)
                                                current_velocity = np.array([np.cos(random_angle), np.sin(random_angle)])
                                            else:
                                                # Normalize the direction vector
                                                current_velocity = current_velocity / np.linalg.norm(current_velocity)
                                            
                                            # Use a random displacement in a reasonable range, 1.0-1.4 m per frame
                                            random_distance = np.random.uniform(1.0, 1.4)
                                            
                                            # Add small random jitter to the direction (+/-15 degrees)
                                            angle_jitter = np.random.uniform(-0.26, 0.26)  # +/-15 degrees in radians
                                            angle = np.arctan2(current_velocity[1], current_velocity[0]) + angle_jitter
                                            direction = np.array([np.cos(angle), np.sin(angle)])
                                            
                                            # Compute the new position
                                            dx = direction[0] * random_distance
                                            dy = direction[1] * random_distance
                                            
                                            new_x = current_position_victim[0] + dx
                                            new_y = current_position_victim[1] + dy
                                            
                                            # Convert the new position back to map coordinates
                                            new_pos_victim = np.concatenate(([new_x, new_y], current_position_victim[2:]))
                                            new_pos_map = bbox_sensor_to_map(new_pos_victim, victim_pose)
                                            
                                            # Smooth the motion with 80% new position and 20% current position
                                            current_position_map = 0.8 * new_pos_map + 0.2 * current_position_map
                                            
                                        elif hasattr(attacker, 'mvig_model') and not is_random_baseline:
                                            # Convert the current position from map coordinates to the victim frame
                                            victim_id = attack_opts["victim_vehicle_id"]
                                            victim_pose = case[frame_idx][victim_id]["lidar_pose"]
                                            current_position_victim = bbox_map_to_sensor(current_position_map, victim_pose)
                                            
                                            # Compute the current velocity if this is not the first move
                                            current_velocity = np.zeros(2)
                                            if frame_idx > all_target_frames[1]:  # At least two historical frames are available
                                                prev_frame_idx = all_target_frames[all_target_frames.index(frame_idx) - 1]
                                                prev_position_map = bbox_sensor_to_map(
                                                    all_frames_positions[-1],  # Position from the previous frame
                                                    case[prev_frame_idx][attack_opts["attacker_vehicle_id"]]["lidar_pose"]
                                                )
                                                # Compute the velocity vector (x, y)
                                                current_velocity[0] = current_position_map[0] - prev_position_map[0]
                                                current_velocity[1] = current_position_map[1] - prev_position_map[1]
                                            
                                            grid_map_2d = grid_map[0]
                                            
                                            # Normalize grid_map values to the [0, 1] range
                                            grid_min = grid_map_2d.min()
                                            grid_max = grid_map_2d.max()
                                            if grid_max > grid_min:  # Avoid division by zero
                                                normalized_grid = (grid_map_2d - grid_min) / (grid_max - grid_min)
                                            else:
                                                normalized_grid = grid_map_2d
                                            
                                            # Perform the grid conversion in the victim frame
                                            grid_x, grid_y = world_to_grid_coords(
                                                current_position_victim[0], 
                                                current_position_victim[1],
                                                normalized_grid.shape,
                                                range_limit
                                            )
                                            
                                            # Compute the average probability around the current position
                                            current_grid_x, current_grid_y = world_to_grid_coords(
                                                current_position_victim[0], 
                                                current_position_victim[1],
                                                normalized_grid.shape,
                                                range_limit
                                            )
                                            
                                            neighborhood_probs = []
                                            for nx in range(max(0, current_grid_x-2), min(normalized_grid.shape[0], current_grid_x+3)):
                                                for ny in range(max(0, current_grid_y-2), min(normalized_grid.shape[1], current_grid_y+3)):
                                                    # Move CUDA tensors to CPU before reading scalar values
                                                    if hasattr(normalized_grid, 'cpu'):
                                                        prob_value = normalized_grid[nx, ny].cpu().item()
                                                    else:
                                                        prob_value = normalized_grid[nx, ny]
                                                    neighborhood_probs.append(prob_value)
                                            
                                            avg_neighborhood_prob = np.mean(neighborhood_probs) if neighborhood_probs else 0
                                            
                                            # Keep the current position if the surrounding region has low overall probability
                                            if avg_neighborhood_prob < 0.55:  # Use the mean neighborhood probability as the decision criterion
                                                best_pos = current_position_map.copy()
                                                logging.info(f"Current position neighborhood has low probability ({avg_neighborhood_prob:.3f}), maintaining position")
                                                
                                                # Mark the attack as interrupted so later frames keep the current world position
                                                attack_interrupted = True
                                                interrupted_frame = frame_idx
                                            else:
                                                # Search when nearby regions still have high probability
                                                logging.info(f"Good neighborhood probability ({avg_neighborhood_prob:.3f}), searching for best position")
                                                
                                                # Expand the search range to 2.0 m to allow larger motion
                                                search_radius = 2.0
                                                # Search for the best point within a 2.0 m radius
                                                best_prob = 0
                                                best_pos = current_position_map.copy()

                                                # frame id 20 adjustment
                                                # Extract orientation information from the current position

                                                # if len(current_position_victim) >= 7:


                                                #     vehicle_direction = np.array([np.cos(yaw), np.sin(yaw)])
                                                #     logging.info(f"Using attack box heading direction: [{vehicle_direction[0]:.2f}, {vehicle_direction[1]:.2f}]")
                                                # else:

                                                # If orientation is unavailable, compute the direction from the attack box to the victim
                                                victim_id = attack_opts["victim_vehicle_id"]
                                                victim_pose = case[frame_idx][victim_id]["lidar_pose"]
                                                
                                                # Get the victim position, which is the origin in the victim frame
                                                victim_position = np.array([0.0, 0.0])
                                                
                                                # Compute the direction vector from the current position to the victim
                                                direction_to_victim = victim_position - current_position_victim[:2]
                                            
                                                # Normalize the direction vector
                                                norm = np.linalg.norm(direction_to_victim)
                                                if norm > 1e-6:
                                                    vehicle_direction = direction_to_victim / norm
                                                    logging.info(f"Using direction toward victim: [{vehicle_direction[0]:.2f}, {vehicle_direction[1]:.2f}]")
                                                else:
                                                    # Use the default forward direction if the point is too close
                                                    vehicle_direction = np.array([1.0, 0.0])
                                                    logging.info("Using default forward direction (too close to victim)")
                                                

                                                current_velocity = np.zeros(2)
                                                has_velocity = False
                                                
                                                if frame_idx > all_target_frames[1] and len(all_frames_positions) > 0:  # At least two historical frames are available
                                                    prev_frame_idx = all_target_frames[all_target_frames.index(frame_idx) - 1]
                                                    prev_position_map = bbox_sensor_to_map(
                                                        all_frames_positions[-1],  # Position from the previous frame
                                                        case[prev_frame_idx][attack_opts["attacker_vehicle_id"]]["lidar_pose"]
                                                    )
                                                    # Compute the velocity vector (x, y)
                                                    current_velocity[0] = current_position_map[0] - prev_position_map[0]
                                                    current_velocity[1] = current_position_map[1] - prev_position_map[1]
                                                    
                                                    # Check whether a clear velocity is available
                                                    if np.linalg.norm(current_velocity) >= 0.1:
                                                        has_velocity = True

                                                        current_velocity = current_velocity / np.linalg.norm(current_velocity)
                                                
                                                # Determine the base movement direction
                                                # Prefer the velocity direction when it is clear; otherwise use the vehicle heading
                                                if has_velocity:
                                                    # Compute the angle between the velocity direction and the vehicle heading
                                                    dot_product = np.dot(current_velocity, vehicle_direction)
                                                    angle_diff = np.arccos(np.clip(dot_product, -1.0, 1.0))
                                                    
                                                    # If the velocity direction differs too much from the forward heading (>90 deg), the vehicle may be reversing
                                                    # In that case, consider using the rearward direction opposite to the heading
                                                    if angle_diff > np.pi/2:
                                                        # Check whether it aligns with the rearward direction
                                                        reverse_direction = -vehicle_direction
                                                        dot_product_reverse = np.dot(current_velocity, reverse_direction)
                                                        
                                                        if dot_product_reverse > 0.7:  # If it is roughly aligned with the rearward direction
                                                            base_direction = current_velocity  # Keep the current velocity direction
                                                            logging.info("Using current velocity (reverse direction)")
                                                        else:
                                                            # If the velocity matches neither forward nor backward well, prefer the forward heading
                                                            base_direction = vehicle_direction
                                                            logging.info("Velocity inconsistent with vehicle orientation, using vehicle heading")
                                                    else:
                                                        # If the velocity is roughly aligned with the forward heading, use the current velocity
                                                        base_direction = current_velocity
                                                        logging.info("Using current velocity (forward direction)")
                                                else:
                                                    # Use the vehicle heading when no clear velocity exists
                                                    base_direction = vehicle_direction
                                                    logging.info("No significant velocity, using vehicle heading direction")
                                                
                                                # Compute a gradient-based direction for refinement
                                                gradient_x = 0
                                                gradient_y = 0
                                                
                                                # Compute the probability gradient in the forward sector, considering only +/-60 degrees
                                                current_angle = np.arctan2(base_direction[1], base_direction[0])
                                                for r in range(1, 4):  # Check radii from 1 to 3 grid cells
                                                    for angle_offset in np.linspace(-np.pi/3, np.pi/3, 5):  # Forward +/-60 degrees sampled at 5 directions
                                                        angle = current_angle + angle_offset
                                                        dx = r * np.cos(angle)
                                                        dy = r * np.sin(angle)
                                                        

                                                        nx = int(current_grid_x + dx)
                                                        ny = int(current_grid_y + dy)
                                                        
                                                        if 0 <= nx < normalized_grid.shape[0] and 0 <= ny < normalized_grid.shape[1]:
                                                            # Get the probability at that point
                                                            if hasattr(normalized_grid, 'cpu'):
                                                                prob = normalized_grid[nx, ny].cpu().item()
                                                            else:
                                                                prob = normalized_grid[nx, ny]
                                                            
                                                            # Compute the gradient contribution so higher probability contributes more
                                                            # Assign a larger weight when the angle is closer to the current direction
                                                            angle_weight = 1.0 - abs(angle_offset) / (np.pi/3)
                                                            weight = prob * angle_weight / (r * r)  # Use a smaller weight for farther points
                                                            gradient_x += weight * np.cos(angle)
                                                            gradient_y += weight * np.sin(angle)
                                                
                                                # Determine the final preferred direction
                                                if abs(gradient_x) > 1e-6 or abs(gradient_y) > 1e-6:
                                                    # If the gradient is valid, compute the normalized gradient direction
                                                    gradient_norm = np.sqrt(gradient_x**2 + gradient_y**2)
                                                    gradient_direction = np.array([gradient_x/gradient_norm, gradient_y/gradient_norm])
                                                    
                                                    # Compute the angle between the gradient direction and the base direction
                                                    dot_product = np.dot(gradient_direction, base_direction)
                                                    angle_diff = np.arccos(np.clip(dot_product, -1.0, 1.0))
                                                    
                                                    # Set the blending ratio according to the angle difference
                                                    # Larger angle differences favor keeping the base direction
                                                    if angle_diff < np.pi/6:  # If the difference is under 30 degrees, rely more on the gradient direction
                                                        blend_factor = 0.7  # 70% base direction and 30% gradient direction
                                                    elif angle_diff < np.pi/4:  # If the difference is under 45 degrees, use a moderate amount of gradient guidance
                                                        blend_factor = 0.8  # 80% base direction and 20% gradient direction
                                                    elif angle_diff < np.pi/3:  # If the difference is under 60 degrees, use only a small amount of gradient guidance
                                                        blend_factor = 0.9  # 90% base direction and 10% gradient direction
                                                    else:  # If the difference is above 60 degrees, almost ignore the gradient direction
                                                        blend_factor = 0.95  # 95% base direction and 5% gradient direction
                                                    
                                                    # Blend the base direction with the gradient direction
                                                    blended_direction = blend_factor * base_direction + (1-blend_factor) * gradient_direction
                                                    blended_norm = np.linalg.norm(blended_direction)
                                                    
                                                    if blended_norm > 1e-6:
                                                        ideal_direction = blended_direction / blended_norm
                                                        logging.info(f"Blending directions: base={base_direction}, gradient={gradient_direction}, result={ideal_direction}")
                                                    else:
                                                        ideal_direction = base_direction
                                                        logging.info(f"Using base direction due to blending issue: {base_direction}")
                                                else:
                                                    # Use the base direction directly when no valid gradient exists
                                                    ideal_direction = base_direction
                                                    logging.info(f"No significant gradient, using base direction: {base_direction}")
                                                
                                                # Target step length reduced to 0.8 m per frame
                                                ideal_distance = 0.8
                                                
                                                # Increase the search density
                                                for dist_factor in np.linspace(0.7, 1.3, 5):  # Distance scaling factor
                                                    target_dist = ideal_distance * dist_factor
                                                    
                                                    for angle_offset in np.linspace(-0.3, 0.3, 7):  # Angle offset in radians
                                                        # Compute the rotated direction
                                                        angle = np.arctan2(ideal_direction[1], ideal_direction[0]) + angle_offset
                                                        direction = np.array([np.cos(angle), np.sin(angle)])
                                                        
                                                        # Compute the new position
                                                        dx = direction[0] * target_dist
                                                        dy = direction[1] * target_dist
                                                        
                                                        new_x = current_position_victim[0] + dx
                                                        new_y = current_position_victim[1] + dy
                                                        grid_x, grid_y = world_to_grid_coords(new_x, new_y, normalized_grid.shape, range_limit)
                                                        
                                                        if 0 <= grid_x < normalized_grid.shape[0] and 0 <= grid_y < normalized_grid.shape[1]:
                                                            # Base probability
                                                            if hasattr(normalized_grid, 'cpu'):
                                                                prob = normalized_grid[grid_x, grid_y].cpu().item()
                                                            else:
                                                                prob = normalized_grid[grid_x, grid_y]
                                                            
                                                            # Direction-consistency bonus; higher when closer to the preferred direction
                                                            direction_bonus = 0.2 * (1.0 - abs(angle_offset) / 0.3)
                                                            
                                                            # Velocity-consistency bonus; higher when closer to the preferred velocity
                                                            speed_bonus = 0.15 * (1.0 - abs(dist_factor - 1.0) / 0.3)
                                                            
                                                            # Add an extra consistency bonus when historical velocity is available
                                                            history_bonus = 0
                                                            if has_velocity:
                                                                # Compute the alignment between the new direction and the historical direction
                                                                history_dir = current_velocity  # Already normalized historical direction
                                                                new_dir = direction  # Currently evaluated direction
                                                                dir_similarity = np.dot(history_dir, new_dir)  # Closer to 1 is better
                                                                history_bonus = 0.3 * dir_similarity  # Up to 0.3 additional bonus
                                                            
                                                            # Total score
                                                            total_score = prob + direction_bonus + speed_bonus + history_bonus
                                                            
                                                            if total_score > best_prob:
                                                                best_prob = total_score
                                                                # Convert the best position back to map coordinates
                                                                best_pos_victim = np.concatenate(([new_x, new_y], current_position_victim[2:]))
                                                                best_pos = bbox_sensor_to_map(best_pos_victim, victim_pose)
                                                
                                                # Reduce smoothing to allow larger motion
                                                current_position_map = 0.8 * best_pos + 0.2 * current_position_map
                                        
                                    # Convert map coordinates to the attacker frame of the current frame
                                    frame_position = bbox_map_to_sensor(
                                        current_position_map,
                                        case[frame_idx][attack_opts["attacker_vehicle_id"]]["lidar_pose"]
                                    )
                                    all_frames_positions.append(frame_position)

                                    
                                    # If the attack is interrupted, record the current world position so later frames keep it
                                    if 'attack_interrupted' in locals() and attack_interrupted:
                                        # Record the interrupted position and reuse it for all later frames
                                        fixed_world_position = current_position_map.copy()
                                        logging.info(f"Attack interrupted at frame {frame_idx}, fixing position for all subsequent frames")
                                        
                                        # Pre-compute the position for all later frames
                                        remaining_frames = all_target_frames[all_target_frames.index(frame_idx)+1:]
                                        for next_frame in remaining_frames:
                                            # Convert the fixed world position to the attacker frame of each corresponding frame
                                            next_frame_position = bbox_map_to_sensor(
                                                fixed_world_position,
                                                case[next_frame][attack_opts["attacker_vehicle_id"]]["lidar_pose"]
                                            )
                                            all_frames_positions.append(next_frame_position)
                                        
                                        # Break the loop and stop processing later frames
                                        break
                                    
                                    mode = "MVIG-guided" if hasattr(attacker, 'mvig_model') and not is_random_baseline else "random"
                                    logging.info(f"Generated {mode} trajectory with {len(all_frames_positions)} positions")
                        else:
                            # In single-frame attack mode, copy the result to all frames
                            positions = np.tile(position_np, (total_frames, 1))
                            attack_opts["positions"] = positions
                            attack_opts["frame_ids"] = [initial_frame]
                            attack["attack_meta"]["attack_frame_ids"] = [initial_frame]
                            bboxes_list = [positions[i] for i in range(len(positions))]
                            attack["attack_meta"]["bboxes"] = bboxes_list
                            attack["attack_meta"]["bbox"] = [bboxes_list[initial_frame]]
                            
                            # Save the predicted position into the attacker object
                            attacker.mvig_predicted_position = positions
                            
                            logging.info(f"Using {'Random-baseline' if is_random_baseline else 'MVIG-optimized'} position for target frame {initial_frame}")
                    else:
                        logging.warning(f"No temporal graphs generated for case {case_id}")
                        attack_opts["frame_ids"] = [9]
                        attack["attack_meta"]["attack_frame_ids"] = [9]
                        use_persistent_attack = False  # Fall back to single-frame attack
                else:
                    logging.warning(f"Occupancy map not found for case {case_id}: {occupancy_path}")
                    attack_opts["frame_ids"] = [9]
                    attack["attack_meta"]["attack_frame_ids"] = [9]
                    use_persistent_attack = False  # Fall back to single-frame attack
            except Exception as e:
                logging.error(f"Error using MVIG model: {str(e)}\nTraceback:\n{traceback.format_exc()}")
                logging.error("Falling back to default attack position")
                attack_opts["frame_ids"] = [9]
                attack["attack_meta"]["attack_frame_ids"] = [9]
                use_persistent_attack = False  # Fall back to single-frame attack
    else: # baseline attack
        attack_opts["frame_ids"] = attack_frame_ids
        attack["attack_meta"]["attack_frame_ids"] = attack_frame_ids


    # # Create the metadata dictionary
    # attack_metadata = {
    #     'used_mvig': is_mvig,
    #     'is_random_baseline': is_random_baseline,
    #     'is_persistent_attack': use_persistent_attack,
    #     'history_frames': attack["attack_meta"]["attack_frame_ids"]
    # }
    

    if use_persistent_attack:

        new_case = copy.deepcopy(case)
        

        all_frames_bboxes = []
        
        for frame_idx, target_frame in enumerate(all_target_frames):
            logging.info(f"Executing persistent attack on frame {target_frame} ({frame_idx+1}/{len(all_target_frames)})")
            

            current_attack_opts = copy.deepcopy(attack_opts)
            current_attack_opts["frame_ids"] = [target_frame]
            

            if all_frames_positions:

                current_positions = np.zeros((total_frames, len(all_frames_positions[0])))
                current_positions[target_frame] = all_frames_positions[frame_idx]
                current_attack_opts["positions"] = current_positions
                

                if frame_idx < len(all_frames_positions):
                    all_frames_bboxes.append(all_frames_positions[frame_idx])
            

            current_attack = copy.deepcopy(attack)
            current_attack["attack_meta"]["current_frame_index"] = frame_idx
            current_attack["attack_meta"]["current_frame_id"] = target_frame
            current_attack["attack_meta"]["attack_frame_ids"] = all_target_frames
            

            frame_new_case, frame_attack_info = attacker.run(new_case, current_attack_opts)
            

            new_case[target_frame] = frame_new_case[target_frame]
            

            if isinstance(frame_attack_info, list):
                all_frames_attack_info[target_frame] = frame_attack_info[target_frame]
            else:
                all_frames_attack_info[target_frame] = frame_attack_info
            

            frame_save_file = os.path.join(data_dir, f"attack_info_frame{target_frame}{save_suffix}.pkl")
            if isinstance(frame_attack_info, list):
                pickle_cache_dump(frame_attack_info, frame_save_file)
            else:
                frame_info = copy.deepcopy(frame_attack_info)
                if is_mvig and 'positions' in current_attack_opts:
                    frame_info['mvig_position'] = current_attack_opts['positions']
                frame_info.update({
                    'used_mvig': is_mvig,
                    'is_random_baseline': is_random_baseline,
                    'is_persistent_attack': use_persistent_attack,
                    'current_frame': target_frame,
                    'attack_frame_ids': all_target_frames
                })
                pickle_cache_dump(frame_info, frame_save_file)
        

        attack_info = all_frames_attack_info
        

        if all_frames_bboxes:
            attack["attack_meta"]["bboxes"] = all_frames_bboxes

            attack["attack_meta"]["bbox"] = all_frames_bboxes
    else:

        new_case, attack_info = attacker.run(case, attack_opts)
        

        attack["attack_meta"]["attack_frame_ids"] = all_target_frames
        attack["attack_meta"]["is_persistent_attack"] = False

        if 'positions' in attack_opts and attack_opts['positions'] is not None:
            position = attack_opts['positions'][all_target_frames[0]]
            if not isinstance(attack["attack_meta"].get("bbox", None), list):
                attack["attack_meta"]["bbox"] = [position]
            attack["attack_meta"]["bboxes"] = [position]
    

    attack_metadata = {
        'used_mvig': is_mvig,
        'is_random_baseline': is_random_baseline,
        'is_persistent_attack': use_persistent_attack,
        'attack_frame_ids': all_target_frames
    }
    
    # Add interruption information if the attack was interrupted
    if 'attack_interrupted' in locals() and attack_interrupted:
        attack_metadata['attack_interrupted'] = True
        # Record the frame where the interruption occurred
        if 'fixed_world_position' in locals():
            interrupted_frame_idx = all_target_frames.index(interrupted_frame)
            attack_metadata['interrupted_at_frame'] = interrupted_frame
            logging.info(f"Recording attack interruption at frame {interrupted_frame} (position {interrupted_frame_idx+1}/{len(all_target_frames)})")
    else:
        attack_metadata['attack_interrupted'] = False


    save_file = os.path.join(data_dir, f"attack_info{save_suffix}.pkl")
    if isinstance(attack_info, list):
        pickle_cache_dump(attack_info, save_file)
        metadata_file = os.path.join(data_dir, f"attack_metadata{save_suffix}.pkl")
        pickle_cache_dump(attack_metadata, metadata_file)
    else:
        if is_mvig and 'positions' in attack_opts:
            attack_info['mvig_position'] = attack_opts['positions']
        attack_info.update(attack_metadata)
        pickle_cache_dump(attack_info, save_file)

    if isinstance(attacker, LidarSpoofEarlyAttacker) or isinstance(attacker, LidarRemoveEarlyAttacker):
        # Early-fusion attacks evaluation
        for perception_name in ["pointpillar_early", "pointpillar_intermediate"]:
            perception_save_file = os.path.join(data_dir, f"{perception_name}{save_suffix}.pkl")
            perception = perception_dict[perception_name]
            perception_feature = [{} for _ in range(total_frames)]
            
            # Evaluate the effect on all attacked frames
            for target_frame in attack["attack_meta"]["attack_frame_ids"]:
                pred_bboxes, pred_scores = perception.run(new_case[target_frame], 
                                                        ego_id=attack_opts["victim_vehicle_id"])
                
                # Store results and metadata in perception_feature
                perception_feature[target_frame][attack_opts["victim_vehicle_id"]] = {
                    "pred_bboxes": pred_bboxes, 
                    "pred_scores": pred_scores
                }
                
            # Attach the metadata dictionary
            metadata_dict = os.path.join(data_dir, f"perception_metadata{save_suffix}.pkl")
            pickle_cache_dump(attack_metadata, metadata_dict)
            
            pickle_cache_dump(perception_feature, perception_save_file)
            
            # Visualization with appropriate suffix
            if is_visualize:
                dataset.load_feature(new_case, perception_feature)
                visualization_file = os.path.join(data_dir, f"visualization{save_suffix}.png")
                draw_attack(attack, case, new_case, mode="multi_frame", show=False, 
                            save=visualization_file)
            else:
                # Visualization with appropriate suffix
                dataset.load_feature(new_case, attack_info)
                visualization_file = os.path.join(data_dir, f"visualization{save_suffix}.png")

        # Pass the track_frames argument for persistent attacks
        if use_persistent_attack:
            if is_visualize:
                draw_attack(attack, case, new_case, mode="multi_frame", show=False, 
                        save=visualization_file, track_frames=all_target_frames)
            
            # # Create a separate visualization for each frame
            # for target_frame in all_target_frames:
            #     frame_visualization_file = os.path.join(data_dir, f"visualization_frame{target_frame}{save_suffix}.png")
                
            #     # Create metadata for a single-frame attack
            #     frame_attack = copy.deepcopy(attack)
            #     frame_attack["attack_meta"]["attack_frame_ids"] = [target_frame]
                
            #     # Draw a single-frame visualization
            #     draw_attack(frame_attack, case, new_case, mode="multi_frame", show=False, 
            #                save=frame_visualization_file)
        else:
            # Single-frame attacks do not need track_frames
            if is_visualize:
                draw_attack(attack, case, new_case, mode="multi_frame", show=False, 
                        save=visualization_file)


def attack_evaluation(attacker, perception_name, is_mvig=False):

     # Check whether random-baseline mode is active
    is_random_baseline = is_mvig and hasattr(attacker.mvig_model, 'random_baseline') and attacker.mvig_model.random_baseline
    logging.info("Evaluating attack {} at perception {} {}".format(
        attacker.name, 
        perception_name,
        "(MVIG)" if is_mvig and not is_random_baseline else "(Baseline)"
    ))
    max_cases = dataset.cache_size
    actual_case_number = min(len(attacker.attack_list), max_cases)
    
    success_log = np.zeros(actual_case_number).astype(bool)
    max_iou = np.zeros((actual_case_number, 2)).astype(np.float32)
    best_score = np.zeros((actual_case_number, 2)).astype(np.float32)

    save_dir = os.path.join(result_dir, "evaluation")
    os.makedirs(save_dir, exist_ok=True)

    delta_ap_values = []  # Store the list of AP deltas

    @attack_case_iterator
    def attack_evaluation_processor(attacker, perception_name, is_mvig=False, case_id=None, case=None, data_dir=None, attack_id=None, attack=None):
        if attack_id >= actual_case_number:
            return
            
        ego_id = attack["attack_meta"]["victim_vehicle_id"]
        attacker_id = attack["attack_meta"]["attacker_vehicle_id"]
        case_id = attack["attack_meta"]["case_id"]
        
        # Check whether persistent-attack mode is active
        is_persistent_attack = attack_persist
        
        # Get the list of attacked frames
        if is_persistent_attack:
            # If the attack is persistent and multiple attack-frame IDs exist, use all of them
            target_frame_ids = attack_frame_ids
            logging.info(f"Using attack_frame_ids from attack_meta: {target_frame_ids}")
        else:
            # Use single-frame attack by default
            target_frame_ids = [9]  # Default attack frame
            
        logging.info(f"Evaluating {'persistent' if is_persistent_attack else 'single-frame'} attack "
                   f"with {len(target_frame_ids)} target frames: {target_frame_ids}")

        # Set the save suffix according to the model type
        if is_random_baseline:
            suffix = "_random"
        elif is_mvig:
            suffix = "_mvig"
        else:
            suffix = ""
        
        # Load the full feature data for the attacked state
        if "early" in attacker.name:
            attack_feature_file = os.path.join(data_dir, f"{perception_name}{suffix}.pkl")
        else:
            attack_feature_file = os.path.join(data_dir, f"attack_info{suffix}.pkl")
        
        if not os.path.exists(attack_feature_file):
            logging.warning(f"Attack feature file not found: {attack_feature_file}")
            return
            
        attack_feature_data = pickle_cache_load(attack_feature_file)
        
        # Load the feature data for the non-attacked state
        normal_feature_path = os.path.join(result_dir, f"normal/{case_id:06d}/{perception_name}.pkl")
        if not os.path.exists(normal_feature_path):
            logging.warning(f"Normal feature file not found: {normal_feature_path}")
            return
            
        normal_feature_data = pickle_cache_load(normal_feature_path)
        
        # Get the initial IoU and scores for the non-attacked state
        for frame_idx in target_frame_ids:
            # Skip invalid frames
            if len(normal_feature_data) <= frame_idx or ego_id not in normal_feature_data[frame_idx]:
                continue
                
            # Get the attack bbox for the current frame
            if "bboxes" in attack["attack_meta"] and len(attack["attack_meta"]["bboxes"]) > 0:
                # For multi-frame attacks, locate the corresponding bbox
                if is_persistent_attack:
                    # Make sure the index stays within range
                    bbox_idx = min(target_frame_ids.index(frame_idx), len(attack["attack_meta"]["bboxes"]) - 1)
                    attack_bbox = bbox_sensor_to_map(attack["attack_meta"]["bboxes"][bbox_idx], case[frame_idx][attacker_id]["lidar_pose"])
                else:
                    attack_bbox = bbox_sensor_to_map(attack["attack_meta"]["bboxes"][0], case[frame_idx][attacker_id]["lidar_pose"])
            elif "bbox" in attack["attack_meta"]:
                # Single-frame attacks usually have only one bbox
                if isinstance(attack["attack_meta"]["bbox"], list):
                    attack_bbox = bbox_sensor_to_map(attack["attack_meta"]["bbox"][0], case[frame_idx][attacker_id]["lidar_pose"])
                else:
                    attack_bbox = bbox_sensor_to_map(attack["attack_meta"]["bbox"], case[frame_idx][attacker_id]["lidar_pose"])
            else:
                logging.warning(f"No bbox information found in attack_meta for frame {frame_idx}")
                continue
                
            # Convert to the ego frame
            attack_bbox = bbox_map_to_sensor(attack_bbox, case[frame_idx][ego_id]["lidar_pose"])
            
            # Get predicted boxes in the non-attacked state
            pred_bboxes = normal_feature_data[frame_idx][ego_id]["pred_bboxes"]
            pred_scores = normal_feature_data[frame_idx][ego_id]["pred_scores"]
            
            # Compute IoU and the best score
            for j, pred_bbox in enumerate(pred_bboxes):
                iou = iou2d(pred_bbox, attack_bbox)
                if iou > max_iou[attack_id, 0]:
                    max_iou[attack_id, 0] = iou
                    best_score[attack_id, 0] = pred_scores[j]
            
            # Get predicted boxes in the attacked state
            # Make sure the data is valid
            if isinstance(attack_feature_data, list) and len(attack_feature_data) > frame_idx and ego_id in attack_feature_data[frame_idx]:
                pred_bboxes = attack_feature_data[frame_idx][ego_id]["pred_bboxes"]
                pred_scores = attack_feature_data[frame_idx][ego_id]["pred_scores"]
                
                # Compute IoU and the best score
            for j, pred_bbox in enumerate(pred_bboxes):
                iou = iou2d(pred_bbox, attack_bbox)
                if iou > max_iou[attack_id, 1]:
                    max_iou[attack_id, 1] = iou
                    best_score[attack_id, 1] = pred_scores[j]

                    # Get the attack mode
            current_attack_mode = attack_mode
        
            if "attack_mode" in attack_feature_data[frame_idx][ego_id]:
                current_attack_mode = attack_feature_data[frame_idx][ego_id]["attack_mode"]
            
            # Determine whether the attack succeeds on the current frame
            frame_success = False
            if current_attack_mode in ["BASIC", "BAC"]:
                # Untargeted attack: success means a prediction box is generated
                if len(pred_bboxes) > 0:
                    frame_success = True
                    logging.info(f"Case {attack_id}, Frame {frame_idx}: BASIC/BAC attack successful - generated {len(pred_bboxes)} detection boxes")
            else:
                # Targeted attack: use the IoU-based criterion
                if attacker.name.startswith("lidar_spoof") and max_iou[attack_id, 1] > 0:
                    frame_success = True
                if attacker.name.startswith("lidar_remove") and max_iou[attack_id, 1] == 0:
                    frame_success = True
                
            # Record the result for the current frame
            if is_persistent_attack:
                # For multi-frame attacks, all frames must succeed for the whole attack to count as successful
                if not frame_success:
                    # If any frame fails, the whole attack fails
                    success_log[attack_id] = False
                    logging.info(f"Case {attack_id}, Frame {frame_idx}: Attack failed, marking entire attack as failed")
                elif frame_idx == target_frame_ids[-1]:
                    # The whole attack succeeds only if the last frame succeeds and no earlier frame failed
                    success_log[attack_id] = True
                    logging.info(f"Case {attack_id}: All frames successful, marking attack as successful")
            else:
                # For single-frame attacks, set the result directly
                success_log[attack_id] = frame_success

            # Record the AP delta when available
            if "delta_ap_0.5" in attack_feature_data[frame_idx][ego_id]:
                delta_ap_values.append(attack_feature_data[frame_idx][ego_id]["delta_ap_0.5"])

    attack_evaluation_processor(attacker, perception_name, is_mvig)
    
    # Save attack results without changing the original format
    if is_random_baseline:
        suffix = "_random"
    elif is_mvig:
        suffix = "_mvig"
    else:
        suffix = ""
        
    pickle_cache_dump(
        {"success": success_log, "iou": max_iou, "score": best_score},
        os.path.join(save_dir, f"attack_result_{attacker.name}_{perception_name}{suffix}.pkl")
    )
    
    # Compute evaluation metrics
    success_rate = np.mean(success_log)
    avg_iou = np.mean(max_iou[:, 1])
    avg_score = np.mean(best_score[:, 1])
    avg_delta_ap = np.mean([v for v in delta_ap_values if v is not None and v != 0])
    if np.isnan(avg_delta_ap):
        avg_delta_ap = 0.0
    
    logging.info(f"Evaluation of attack {attacker.name} at perception {perception_name} "
                f"{('(MVIG)' if is_mvig else '(Baseline)')}, "
                f"total case number {actual_case_number:.2f}, "
                f"success number {np.sum(success_log):.2f}, "
                f"success rate {success_rate:.2f}, "
                f"average IoU {avg_iou:.2f}, "
                f"average score {avg_score:.2f}, "
                f"average AP@0.5 delta {avg_delta_ap:.4f}")
    
    # Return results while keeping the original format unchanged
    return {
        "attack_success": success_rate,
        "detection_rate": avg_score,
        "avg_iou": avg_iou,
        "avg_delta_ap": avg_delta_ap,
        "total_cases": actual_case_number,
        "success_cases": np.sum(success_log),
        "max_iou": max_iou,
        "best_score": best_score,
        "success_log": success_log
    }


@normal_case_iterator
def occupancy_map(lidar_seg_api, case_id=None, case=None, data_dir=None):
    save_file = os.path.join(data_dir, "occupancy_map.pkl")
    if os.path.isfile(save_file):
        logging.info("Occupancy map of case {} already exists".format(case_id))
        return
    else:
        logging.info("Processing occupancy map of case {}".format(case_id))

    occupancy_feature = [{} for _ in range(total_frames)]
    for frame_id in list(range(total_frames)):
        for vehicle_id, vehicle_data in case[frame_id].items():
            lidar, lidar_pose = vehicle_data["lidar"], vehicle_data["lidar_pose"]
            pcd = pcd_sensor_to_map(lidar, lidar_pose)

            lane_info = pickle_cache_load(os.path.join(data_root, "carla/{}_lane_info.pkl".format(vehicle_data["map"])))
            lane_areas = pickle_cache_load(os.path.join(data_root, "carla/{}_lane_areas.pkl".format(vehicle_data["map"])))
            lane_planes = pickle_cache_load(os.path.join(data_root, "carla/{}_ground_planes.pkl".format(vehicle_data["map"])))

            ground_indices, in_lane_mask, point_height = get_ground_plane(pcd, lane_info=lane_info, lane_areas=lane_areas, lane_planes=lane_planes, method="map")
            lidar_seg = lidar_segmentation(lidar, method="squeezeseq", interface=lidar_seg_api)
            
            object_segments = filter_segmentation(lidar, lidar_seg, lidar_pose, in_lane_mask=in_lane_mask, point_height=point_height, max_range=50)
            object_mask = np.zeros(pcd.shape[0]).astype(bool)
            if len(object_segments) > 0:
                object_indices = np.hstack(object_segments)
                object_mask[object_indices] = True

            ego_bbox = vehicle_data["ego_bbox"]
            ego_area = bbox_to_polygon(ego_bbox)
            ego_area_height = ego_bbox[5]

            ret = {
                "ego_area": ego_area,
                "ego_area_height": ego_area_height,
                "plane": None,
                "ground_indices": ground_indices,
                "point_height": point_height,
                "object_segments": object_segments,
                "lidar_pose": lidar_pose,
            }

            height_thres = 0
            occupied_areas, occupied_areas_height = get_occupied_space(pcd, object_segments, point_height=point_height, height_thres=height_thres)
            free_areas = get_free_space(lidar, lidar_pose, object_mask, in_lane_mask=in_lane_mask, point_height=point_height, max_range=50, height_thres=height_thres, height_tolerance=0.2)
            ret["occupied_areas"] = occupied_areas
            ret["occupied_areas_height"] = occupied_areas_height
            ret["free_areas"] = free_areas
            
            occupancy_feature[frame_id][vehicle_id] = ret

    pickle_cache_dump(occupancy_feature, save_file)


@attack_case_iterator
def defense(attacker, defender, perception_name, is_mvig=False, case_id=None, case=None, data_dir=None, attack_id=None, attack=None):
    # Check whether random-baseline mode is active
    is_random_baseline = is_mvig and hasattr(attacker, 'mvig_model') and attacker.mvig_model is not None and hasattr(attacker.mvig_model, 'random_baseline') and attacker.mvig_model.random_baseline
    
    # Set the save suffix according to the model type
    if is_random_baseline:
        suffix = "_random"
    elif is_mvig:
        suffix = "_mvig"
    else:
        suffix = ""
    
    # Check whether persistent-attack mode is active
    use_persistent_attack = attack_persist
    

    if use_persistent_attack:
        # Get all attacked frames from attack_meta
        if attack_frame_ids is not None:
            all_target_frames = attack_frame_ids
            logging.info(f"Processing persistent defense against attack on frames {all_target_frames}")
        else:
            # Use the default frame if no attack frame is specified
            all_target_frames = [9]
            use_persistent_attack = False
            logging.warning("No attack_frame_ids found in attack_meta, falling back to single-frame defense")
    else:
        all_target_frames = [9]
    
    # Set file paths
    if "early" in attacker.name:
        defense_file = os.path.join(data_dir, f"{defender.name}_{perception_name}{suffix}.pkl")
        vis_file = os.path.join(data_dir, f"{defender.name}_{perception_name}{suffix}.png")
    else:
        defense_file = os.path.join(data_dir, f"{defender.name}{suffix}.pkl")
        vis_file = os.path.join(data_dir, f"{defender.name}{suffix}.png")
    
    logging.info(f"Processing defense {defender.name} against attack {attacker.name} on attack case {attack_id} {'(MVIG)' if is_mvig and not is_random_baseline else '(Baseline)'}")

    # Load perception features after the attack
    if "early" in attacker.name:
        perception_file = os.path.join(data_dir, f"{perception_name}{suffix}.pkl")
        if os.path.exists(perception_file):
            perception_feature = pickle_cache_load(perception_file)
        else:
            logging.warning(f"Perception feature file not found: {perception_file}")
            return
    else:
        attack_info_file = os.path.join(data_dir, f"attack_info{suffix}.pkl")
        if os.path.exists(attack_info_file):
            perception_feature = pickle_cache_load(attack_info_file)
        else:
            logging.warning(f"Attack info file not found: {attack_info_file}")
            return
    
    case = dataset.load_feature(case, perception_feature)

    # Load occupancy-map features
    occupancy_feature = pickle_cache_load(os.path.join(result_dir, "normal/{:06d}/occupancy_map.pkl".format(case_id)))
    case = dataset.load_feature(case, occupancy_feature)

    # Initialize the dictionary that stores defense results for all frames
    all_frames_metrics = [{} for _ in range(total_frames)]
    
    if use_persistent_attack:
        # Check whether the defender is based on temporal anomalies, such as ROBOSAC+ or GCP
        is_temporal_defender = defender.name.lower() in ["robosac", "gcp"]
        
        if is_temporal_defender:
            # Temporal-anomaly-based defenders process all attacked frames at once
            logging.info(f"Executing threshold-based defense on all frames {all_target_frames} at once")
            
            # Set multi-frame defense parameters
            defend_opts = {"frame_ids": all_target_frames}
            if hasattr(attacker, 'perception'):
                defend_opts["perception"] = attacker.perception
            
            # Call the defender on all frames
            _, _, metrics = defender.run(case, defend_opts)
            
            # Save results
            pickle_cache_dump(metrics, defense_file)
        else:
            # For non-temporal defenders, run defense frame by frame
            for frame_idx, target_frame in enumerate(all_target_frames):
                logging.info(f"Executing persistent defense on frame {target_frame} ({frame_idx+1}/{len(all_target_frames)})")
                
                # Set defense parameters for the current frame
                defend_opts = {"frame_ids": [target_frame]}
                if hasattr(attacker, 'perception'):
                    defend_opts["perception"] = attacker.perception
                
                # Call the defender
                _, _, frame_metrics = defender.run(case, defend_opts)
                
                # Update the current frame entry in all_frames_metrics
                all_frames_metrics[target_frame] = frame_metrics[target_frame]
                
                # Save defense results for each frame
                frame_defense_file = os.path.join(data_dir, f"{defender.name}_frame{target_frame}{suffix}.pkl")
                pickle_cache_dump(frame_metrics, frame_defense_file)
                
                # Generate visualizations for the CAD defender
                if is_visualize and defender.name == "cad":
                    frame_vis_file = os.path.join(data_dir, f"{defender.name}_frame{target_frame}{suffix}.png")
                    visualize_defense(case, frame_metrics, show=False, save=frame_vis_file)
            
            # Merge defense results from all frames
            metrics = all_frames_metrics
            
            # Save the merged results
            pickle_cache_dump(metrics, defense_file)
    else:
        # Single-frame defense mode: run defense once
        defend_opts = {"frame_ids": [9]}
        if hasattr(attacker, 'perception'):
            defend_opts["perception"] = attacker.perception
    
        # Call the defender
        new_case, score, metrics = defender.run(case, defend_opts)
        
        # Save results
        pickle_cache_dump(metrics, defense_file)
    
    # Generate visualizations for the CAD defender
    if is_visualize and defender.name == "cad":

        if use_persistent_attack:
            visualize_defense(case, metrics, show=False, save=vis_file, track_frames=all_target_frames)
        else:
            visualize_defense(case, metrics, show=False, save=vis_file)


def calculate_ap_at_iou(gt_bboxes, pred_bboxes, iou_threshold=0.5):
    """Compute AP at IoU=0.5
    Args:
        gt_bboxes: [N, 7] Ground-truth boxes
        pred_bboxes: [M, 7] Predicted boxes
        iou_threshold: IoU threshold, default 0.5
    Returns:
        ap: Average Precision value
    """
    if len(gt_bboxes) == 0 or len(pred_bboxes) == 0:
        return 0.0
    

    ious = np.zeros((len(pred_bboxes), len(gt_bboxes)))
    for i, pred in enumerate(pred_bboxes):
        for j, gt in enumerate(gt_bboxes):
            ious[i, j] = iou2d(pred, gt)
    

    confidence = np.ones(len(pred_bboxes))
    
    # Sort by confidence; all confidences are currently equal
    sort_indices = np.argsort(-confidence)
    
    # Initialize the TP and FP arrays
    tp = np.zeros(len(pred_bboxes))
    fp = np.zeros(len(pred_bboxes))
    

    gt_matched = np.zeros(len(gt_bboxes), dtype=bool)
    

    for i, pred_idx in enumerate(sort_indices):

        max_iou = np.max(ious[pred_idx])
        max_gt_idx = np.argmax(ious[pred_idx])
        
        if max_iou >= iou_threshold and not gt_matched[max_gt_idx]:
            tp[i] = 1
            gt_matched[max_gt_idx] = True
        else:
            fp[i] = 1
    
    # Compute cumulative values
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    
    # Compute precision and recall
    precision = cum_tp / (cum_tp + cum_fp + 1e-10)
    recall = cum_tp / len(gt_bboxes)
    
    # Compute AP using all points instead of 11-point interpolation
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))
    
    # Compute the area under the PR curve
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    return ap


def defense_evaluation(attacker, defender, perception_name, is_mvig=False):
    save_dir = os.path.join(result_dir, "evaluation")
    os.makedirs(save_dir, exist_ok=True)
    
    # Check whether random-baseline mode is active
    is_random_baseline = is_mvig and hasattr(attacker, 'mvig_model') and attacker.mvig_model is not None and hasattr(attacker.mvig_model, 'random_baseline') and attacker.mvig_model.random_baseline

    logging.info("Evaluating defense {} against attack {} on perception {} {}".format(
        defender.name, 
        attacker.name, 
        perception_name,
        "(MVIG)" if is_mvig and not is_random_baseline else "(Baseline)"
    ))

    # Add summary statistics
    total_attack_frames = 0
    effective_attack_frames = 0
    total_attack_cases = 0
    interrupted_cases = 0
    
    # Set the save suffix according to the model type
    if is_random_baseline:
        suffix = "_random"
    elif is_mvig:
        suffix = "_mvig"
    else:
        suffix = ""
    
    persistence_frame = "persistence_" + str(persistence)
    range_value = "range_" + str(range_limit)
    save_file = os.path.join(save_dir, f"defense_result_{attacker.name}_{defender.name}_{attack_mode}_{persistence_frame}_{range_value}{suffix}.pkl")
    

    if defender.name == "robosac" or defender.name == "cpguard" or defender.name == "gcp":

        return evaluate_defender_with_threshold(attacker, defender, perception_name, is_mvig, save_dir, suffix)
    

    defense_results = {
        "spoof_error": [],
        "spoof_label": [],
        "spoof_location": [],
        "remove_error": [],
        "remove_label": [],
        "remove_location": [],
        "success": [],
        "avg_iou": [],
        "avg_error": [],
        "ap_50": [],
        "case_ids": []
    }
    
    @attack_case_iterator
    def defense_evaluation_processor(attacker, defender, perception_name, is_mvig=False, case_id=None, case=None, data_dir=None, attack_id=None, attack=None, iou_thres=0.7, dist_thres=40):
        nonlocal total_attack_frames, effective_attack_frames, total_attack_cases, interrupted_cases
        

        total_attack_cases += 1
        

        is_persistent_attack = False
        if "is_persistent_attack" in attack["attack_meta"]:
            is_persistent_attack = attack["attack_meta"]["is_persistent_attack"]
        else:
            is_persistent_attack = attack_persist
        

        is_rc_multi_frame = attack_mode in ["RC", "RC+"] and is_persistent_attack and is_mvig
        

        attack_interrupted = False
        interrupted_at_frame = None
        
        if is_rc_multi_frame:

            if "attack_frame_ids" in attack["attack_meta"]:
                target_frame_ids = attack["attack_meta"]["attack_frame_ids"]
                total_frames_this_case = len(target_frame_ids)
                total_attack_frames += total_frames_this_case
                

                metadata_file = os.path.join(data_dir, f"attack_metadata{suffix}.pkl")
                attack_info_file = os.path.join(data_dir, f"attack_info{suffix}.pkl")
                

                if os.path.exists(metadata_file):
                    attack_metadata = pickle_cache_load(metadata_file)
                    if "attack_interrupted" in attack_metadata and attack_metadata["attack_interrupted"]:
                        attack_interrupted = True
                        if "interrupted_at_frame" in attack_metadata:
                            interrupted_at_frame = attack_metadata["interrupted_at_frame"]
                            interrupted_cases += 1
                
                elif os.path.exists(attack_info_file):
                    attack_info = pickle_cache_load(attack_info_file)
                    if isinstance(attack_info, dict) and "attack_interrupted" in attack_info and attack_info["attack_interrupted"]:
                        attack_interrupted = True
                        if "interrupted_at_frame" in attack_info:
                            interrupted_at_frame = attack_info["interrupted_at_frame"]
                            interrupted_cases += 1
                

                if attack_interrupted and interrupted_at_frame is not None:

                    effective_frames_this_case = sum(1 for frame in target_frame_ids if frame <= interrupted_at_frame)
                    effective_attack_frames += effective_frames_this_case
                    logging.info(f"Case {attack_id}: Attack interrupted at frame {interrupted_at_frame}, "
                                f"effective frames: {effective_frames_this_case}/{total_frames_this_case}")
                else:

                    effective_attack_frames += total_frames_this_case
        

        if "early" in attacker.name:
            defense_file = os.path.join(data_dir, f"{defender.name}_{perception_name}{suffix}.pkl")
        else:
            defense_file = os.path.join(data_dir, f"{defender.name}{suffix}.pkl")
        
        if not os.path.exists(defense_file):
            logging.warning(f"Defense file not found: {defense_file}")
            return
            
        metrics = pickle_cache_load(defense_file)

        attacker_vehicle_id = attack["attack_meta"]["attacker_vehicle_id"]
        victim_vehicle_id = attack["attack_meta"]["victim_vehicle_id"]
        
        # Get the attack mode

        base_attack_mode = "spoof" if "spoof" in attacker.name else "remove"
        

        current_attack_mode = attack_mode
        

        # if os.path.exists(attack_info_file):
        #     attack_info = pickle_cache_load(attack_info_file)
        #     if isinstance(attack_info, list):

        #         if len(attack_info) > 0 and victim_vehicle_id in attack_info[-1]:
        #             if "attack_mode" in attack_info[-1][victim_vehicle_id]:
        #                 current_attack_mode = attack_info[-1][victim_vehicle_id]["attack_mode"]

        #             for frame_data in attack_info:
        #                 if victim_vehicle_id in frame_data and "attack_mode" in frame_data[victim_vehicle_id]:
        #                     current_attack_mode = frame_data[victim_vehicle_id]["attack_mode"]
        #                     break
        #     elif isinstance(attack_info, dict) and "attack_mode" in attack_info:
        #         current_attack_mode = attack_info["attack_mode"]

        # Check whether persistent-attack mode is active
        is_persistent_attack = False
        

        if "is_persistent_attack" in attack["attack_meta"]:
            is_persistent_attack = attack["attack_meta"]["is_persistent_attack"]
        else:
            is_persistent_attack = attack_persist
        
        # Get the list of attacked frames
        if is_persistent_attack and "attack_frame_ids" in attack["attack_meta"]:
            # If the attack is persistent and multiple attack-frame IDs exist, use all of them
            target_frame_ids = attack["attack_meta"]["attack_frame_ids"]
        else:

            target_frame_ids = [9]
        
        logging.info(f"Evaluating defense for {'persistent' if is_persistent_attack else 'single-frame'} attack "
                    f"with {len(target_frame_ids)} target frames: {target_frame_ids}")
        

        rc_attack_bbox = None
        if "bboxes" in attack["attack_meta"] and len(attack["attack_meta"]["bboxes"]) > 0:

            rc_attack_bbox = bbox_sensor_to_map(
                attack["attack_meta"]["bboxes"][-1], 
                case[-1][attacker_vehicle_id]["lidar_pose"]
            )
        elif "bbox" in attack["attack_meta"] and len(attack["attack_meta"]["bbox"]) > 0:

            rc_attack_bbox = bbox_sensor_to_map(
                attack["attack_meta"]["bbox"][-1], 
                case[-1][attacker_vehicle_id]["lidar_pose"]
            )


        frame_results = {
            "spoof_error": [],
            "spoof_label": [],
            "spoof_location": [],
            "remove_error": [],
            "remove_label": [],
            "remove_location": [],
            "success": [],
            "avg_iou": [],
            "avg_error": [],
            "ap_50": []
        }
        

        current_case_id = f"{case_id}_{attack_id}"


        for frame_idx, frame_id in enumerate(target_frame_ids):

            if attack_interrupted and interrupted_at_frame is not None and frame_id > interrupted_at_frame:
                logging.info(f"Attack was interrupted at frame {interrupted_at_frame}, marking defense for frame {frame_id} as failed without checking")

                frame_results["ap_50"].append(np.array([0.0]).astype(np.float32))
                frame_results["avg_iou"].append(np.array([0.0]).astype(np.float32))
                frame_results["avg_error"].append(np.array([0.0]).astype(np.float32))
                frame_results["spoof_error"].append(np.array([]).astype(np.float32))
                frame_results["spoof_label"].append(np.array([]).astype(np.float32))
                frame_results["spoof_location"].append(np.array([]).reshape(0, 2))
                frame_results["remove_error"].append(np.array([]).astype(np.float32))
                frame_results["remove_label"].append(np.array([]).astype(np.float32))
                frame_results["remove_location"].append(np.array([]).reshape(0, 2))
                frame_results["success"].append(np.array([False]).astype(np.int8))
                continue
                

            if frame_id >= len(metrics) or victim_vehicle_id not in metrics[frame_id]:
                logging.warning(f"Vehicle {victim_vehicle_id} not found in metrics for frame {frame_id}")
                continue
                
            vehicle_metrics = metrics[frame_id][victim_vehicle_id]

            gt_bboxes = vehicle_metrics["gt_bboxes"]
            pred_bboxes = vehicle_metrics["pred_bboxes"]
            lidar_pose = vehicle_metrics["lidar_pose"]


            if len(gt_bboxes) == 0 or len(pred_bboxes) == 0:

                frame_results["ap_50"].append(np.array([0.0]).astype(np.float32))
                frame_results["avg_iou"].append(np.array([0.0]).astype(np.float32))
                frame_results["avg_error"].append(np.array([0.0]).astype(np.float32))
                frame_results["spoof_error"].append(np.array([]).astype(np.float32))
                frame_results["spoof_label"].append(np.array([]).astype(np.float32))
                frame_results["spoof_location"].append(np.array([]).reshape(0, 2))
                frame_results["remove_error"].append(np.array([]).astype(np.float32))
                frame_results["remove_label"].append(np.array([]).astype(np.float32))
                frame_results["remove_location"].append(np.array([]).reshape(0, 2))
                frame_results["success"].append(np.array([False]).astype(np.int8))
                continue

            # iou 2d
            gt_bboxes[:, 2] = 0
            gt_bboxes[:, 5] = 1
            pred_bboxes[:, 2] = 0
            pred_bboxes[:, 5] = 1

            iou = np.zeros((gt_bboxes.shape[0], pred_bboxes.shape[0]))
            for i, gt_bbox in enumerate(gt_bboxes):
                for j, pred_bbox in enumerate(pred_bboxes):
                    iou[i, j] = iou2d(gt_bbox, pred_bbox)

            spoof_label = np.max(iou, axis=0) <= iou_thres
            spoof_mask = np.logical_and(get_distance(pred_bboxes[:, :2], lidar_pose[:2]) > 1, get_distance(pred_bboxes[:, :2], lidar_pose[:2]) <= dist_thres)
            remove_label = np.max(iou, axis=1) <= iou_thres
            remove_mask = get_distance(gt_bboxes[:, :2], lidar_pose[:2]) <= dist_thres

            spoof_error = np.zeros(pred_bboxes.shape[0])
            spoof_location = np.zeros((pred_bboxes.shape[0], 2))
            for error_area, error, gt_error, bbox_index in vehicle_metrics["spoof"]:
                if bbox_index < 0 or bbox_index >= len(spoof_error):
                    continue
                if error > spoof_error[bbox_index]:
                    spoof_location[bbox_index] = np.array(list(list(error_area.centroid.coords)[0]))
                    spoof_error[bbox_index] = error

            remove_error = np.zeros(gt_bboxes.shape[0])
            remove_location = np.zeros((gt_bboxes.shape[0], 2))
            for error_area, error, gt_error, bbox_index in vehicle_metrics["remove"]:
                if bbox_index < 0 or bbox_index >= len(remove_error):
                    continue
                if error > remove_error[bbox_index]:
                    remove_location[bbox_index] = np.array(list(list(error_area.centroid.coords)[0]))
                    remove_error[bbox_index] = error


            if current_attack_mode in ["BASIC", "BAC"]:

                attack_bboxes = []
                for j, pred_bbox in enumerate(pred_bboxes):

                    max_iou = 0
                    for i, gt_bbox in enumerate(gt_bboxes):
                        max_iou = max(max_iou, iou[i, j])
                    

                    if max_iou <= iou_thres:
                        attack_bboxes.append(pred_bbox)
                

                attack_bboxes = np.array(attack_bboxes) if attack_bboxes else np.array([])
                logging.info(f"Case {attack_id}, Frame {frame_id}: Found {len(attack_bboxes)} potential attack boxes (boxes with low IoU to GT)")
                

                detected_location = spoof_location if base_attack_mode == "spoof" else remove_location
                detected_location = detected_location[spoof_mask] if base_attack_mode == "spoof" else detected_location[remove_mask]
                

                if len(detected_location) > 0 and len(attack_bboxes) > 0:

                    all_distances = np.zeros((len(detected_location), len(attack_bboxes)))
                    

                    for i, loc in enumerate(detected_location):
                        for j, box in enumerate(attack_bboxes):
                            all_distances[i, j] = np.sqrt(np.sum((loc - box[:2])**2))
                    

                    min_distance = np.min(all_distances)
                    

                    is_success = min_distance < 2
                    if is_success:
                        logging.info(f"Case {attack_id}, Frame {frame_id}: Successfully detected {current_attack_mode} attack! Distance to nearest attack box: {min_distance:.2f}m")
                    else:
                        logging.info(f"Case {attack_id}, Frame {frame_id}: Failed to accurately locate attack boxes. Min distance: {min_distance:.2f}m")
                elif len(attack_bboxes) == 0:

                    logging.info(f"Case {attack_id}, Frame {frame_id}: No attack boxes found in {current_attack_mode} attack mode")
                    is_success = False
                else:

                    logging.info(f"Case {attack_id}, Frame {frame_id}: No detection locations found in {current_attack_mode} attack mode")
                    is_success = False
                

                if not is_success:
                    if "detected_attackers" in vehicle_metrics and str(attacker_vehicle_id) in vehicle_metrics["detected_attackers"]:
                        is_success = True
                        logging.info(f"Case {attack_id}, Frame {frame_id}: Successfully detected {current_attack_mode} attack! Attacker ID {attacker_vehicle_id} correctly identified.")
                    elif "classification_results" in vehicle_metrics:
                        for collab_id, is_attacker in vehicle_metrics["classification_results"].items():
                            if int(collab_id) == attacker_vehicle_id and is_attacker == 1:
                                is_success = True
                                logging.info(f"Case {attack_id}, Frame {frame_id}: Successfully detected {current_attack_mode} attack! Attacker ID {attacker_vehicle_id} correctly classified.")
                                break
            else:

                detected_location = spoof_location if base_attack_mode == "spoof" else remove_location
                detected_location = detected_location[spoof_mask] if base_attack_mode == "spoof" else detected_location[remove_mask]
                    

                current_frame_attack_bbox = None
                

                if "bboxes" in attack["attack_meta"] and len(attack["attack_meta"]["bboxes"]) > 0:

                    if frame_idx < len(attack["attack_meta"]["bboxes"]):
                        current_frame_attack_bbox = bbox_sensor_to_map(
                            attack["attack_meta"]["bboxes"][frame_idx], 
                            case[frame_id][attacker_vehicle_id]["lidar_pose"]
                        )

                    elif rc_attack_bbox is not None:
                        current_frame_attack_bbox = rc_attack_bbox

                elif rc_attack_bbox is not None:
                    current_frame_attack_bbox = rc_attack_bbox
                
                if len(detected_location) > 0 and current_frame_attack_bbox is not None:
                    min_distance = np.min(get_distance(detected_location, current_frame_attack_bbox[:2]))
                    is_success = min_distance < 2
                    if is_success:
                        logging.info(f"Case {attack_id}, Frame {frame_id}: Successfully detected {base_attack_mode} attack! Distance to attack: {min_distance:.2f}m")
                    else:
                        logging.info(f"Case {attack_id}, Frame {frame_id}: Failed to detect {base_attack_mode} attack. Min distance: {min_distance:.2f}m")
                else:
                    is_success = False
                    if len(detected_location) == 0:
                        logging.info(f"Case {attack_id}, Frame {frame_id}: No detection locations found")
                    if current_frame_attack_bbox is None:
                        logging.info(f"Case {attack_id}, Frame {frame_id}: No attack bbox available")


            if base_attack_mode == "spoof":

                valid_ious = np.max(iou, axis=0)[spoof_mask]
                avg_iou = np.mean(valid_ious) if len(valid_ious) > 0 else 0
            else:

                valid_ious = np.max(iou, axis=1)[remove_mask]
                avg_iou = np.mean(valid_ious) if len(valid_ious) > 0 else 0


            if base_attack_mode == "spoof":
                valid_errors = spoof_error[spoof_mask]

                if len(valid_errors) > 0:
                    normalized_errors = np.exp(valid_errors) / np.sum(np.exp(valid_errors))
                    avg_error = np.mean(normalized_errors)
                else:
                    avg_error = 0
            else:
                valid_errors = remove_error[remove_mask]
                if len(valid_errors) > 0:
                    normalized_errors = np.exp(valid_errors) / np.sum(np.exp(valid_errors))
                    avg_error = np.mean(normalized_errors)
                else:
                    avg_error = 0


            ap_50 = calculate_ap_at_iou(gt_bboxes, pred_bboxes, iou_threshold=0.5)


            frame_results["ap_50"].append(np.array([ap_50]).astype(np.float32))
            frame_results["avg_iou"].append(np.array([avg_iou]).astype(np.float32))
            frame_results["avg_error"].append(np.array([avg_error]).astype(np.float32))
            frame_results["spoof_error"].append(spoof_error[spoof_mask])
            frame_results["spoof_label"].append(spoof_label[spoof_mask])
            frame_results["spoof_location"].append(spoof_location[spoof_mask])
            frame_results["remove_error"].append(remove_error[remove_mask])
            frame_results["remove_label"].append(remove_label[remove_mask])
            frame_results["remove_location"].append(remove_location[remove_mask])
            frame_results["success"].append(np.array([is_success]).astype(np.int8))
            

            if is_persistent_attack:

                if not any(key in frame_results and frame_results[key] for key in frame_results):
                    logging.warning(f"No valid frame results for attack {attack_id}")
                    return
                    

                any_frame_success = np.any([np.any(s) for s in frame_results["success"] if len(s) > 0])
                defense_results["success"].append(np.array([any_frame_success]).astype(np.int8))
                

                for key in ["ap_50", "avg_iou", "avg_error"]:
                    if frame_results[key] and any(len(v) > 0 for v in frame_results[key]):
                        avg_value = np.mean([np.mean(v) for v in frame_results[key] if len(v) > 0])
                        defense_results[key].append(np.array([avg_value]).astype(np.float32))
                    else:
                        defense_results[key].append(np.array([0.0]).astype(np.float32))
                

                for key in ["spoof_error", "spoof_label", "remove_error", "remove_label"]:
                    if frame_results[key] and any(len(v) > 0 for v in frame_results[key]):
                        combined = np.concatenate([v for v in frame_results[key] if len(v) > 0])
                        defense_results[key].append(combined)
                        

                        if key in ["spoof_error", "remove_error"]:
                            defense_results["case_ids"].extend([current_case_id] * len(combined))
                    else:
                        defense_results[key].append(np.array([]))
                

                for key in ["spoof_location", "remove_location"]:
                    if frame_results[key] and any(len(v) > 0 for v in frame_results[key]):
                        combined = np.concatenate([v for v in frame_results[key] if len(v) > 0])
                        defense_results[key].append(combined)
                    else:
                        defense_results[key].append(np.array([]).reshape(0, 2))
                
                logging.info(f"Case {attack_id}: Persistent attack defense evaluation complete. "
                            f"Defense success: {any_frame_success}, "
                            f"Evaluated {len(target_frame_ids)} frames.")
            else:

                for key in defense_results.keys():
                    if key == "case_ids":
                        continue
                    if frame_results[key]:
                        defense_results[key].extend(frame_results[key])
                        

                        if key in ["spoof_error", "remove_error"]:
                            defense_results["case_ids"].extend([current_case_id] * len(frame_results[key][0]))


    defense_evaluation_processor(attacker, defender, perception_name, is_mvig)


    for key, data in defense_results.items():
        if key == "case_ids":
            continue
        if data:

            shapes = [d.shape for d in data if hasattr(d, 'shape')]
            if all(len(shape) == 1 for shape in shapes):
                defense_results[key] = np.concatenate(data)
            else:
                try:
                    defense_results[key] = np.concatenate(data).reshape(-1)
                except ValueError:

                    flat_list = []
                    for item in data:
                        if isinstance(item, np.ndarray):
                            flat_list.extend(item.flatten())
                        else:
                            flat_list.append(item)
                    defense_results[key] = np.array(flat_list)
        else:
            defense_results[key] = np.array([])

    pickle_cache_dump(defense_results, save_file)

    persistence_frame = "persistence_" + str(persistence)
    range_value = "range_" + str(range_limit)
    

    if not attack_persist:

        if "spoof" in attacker.name and len(defense_results["spoof_error"]) > 0:
            spoof_best_TPR, spoof_best_FPR, spoof_roc_auc, spoof_best_thres = draw_roc(
                defense_results["spoof_error"], 
                defense_results["spoof_label"],
                save=os.path.join(save_dir, f"roc_{attacker.name}_{defender.name}_{attack_mode}_{persistence_frame}_{range_value}{suffix}.png"))
        else:
            spoof_best_TPR, spoof_best_FPR, spoof_roc_auc, spoof_best_thres = 0, 0, 0, 0
            
        if "remove" in attacker.name and len(defense_results["remove_error"]) > 0:
            remove_best_TPR, remove_best_FPR, remove_roc_auc, remove_best_thres = draw_roc(
                defense_results["remove_error"], 
                defense_results["remove_label"],
                save=os.path.join(save_dir, f"roc_{attacker.name}_{defender.name}_{attack_mode}_{persistence_frame}_{range_value}{suffix}.png"))
        else:
            remove_best_TPR, remove_best_FPR, remove_roc_auc, remove_best_thres = 0, 0, 0, 0
    else:

        try:

            if "spoof" in attacker.name and len(defense_results["spoof_error"]) > 0:
                spoof_best_TPR, spoof_best_FPR, spoof_roc_auc, spoof_best_thres = draw_roc(
                    defense_results["spoof_error"], 
                    defense_results["spoof_label"],
                    save=os.path.join(save_dir, f"roc_{attacker.name}_{defender.name}_{attack_mode}_{persistence_frame}_{range_value}{suffix}.png"),
                    multi_frame=True, 
                    case_ids=defense_results["case_ids"]
                )
                logging.info(f"Multi-frame Spoof ROC analysis - AUC: {spoof_roc_auc:.4f}, Best threshold: {spoof_best_thres:.4f}")
            else:
                spoof_best_TPR, spoof_best_FPR, spoof_roc_auc, spoof_best_thres = 0, 0, 0, 0
                

            if "remove" in attacker.name and len(defense_results["remove_error"]) > 0:
                remove_best_TPR, remove_best_FPR, remove_roc_auc, remove_best_thres = draw_roc(
                    defense_results["remove_error"], 
                    defense_results["remove_label"],
                    save=os.path.join(save_dir, f"roc_{attacker.name}_{defender.name}_{attack_mode}_{persistence_frame}_{range_value}{suffix}.png"),
                    multi_frame=True,
                    case_ids=defense_results["case_ids"]
                )
                logging.info(f"Multi-frame Remove ROC analysis - AUC: {remove_roc_auc:.4f}, Best threshold: {remove_best_thres:.4f}")
            else:
                remove_best_TPR, remove_best_FPR, remove_roc_auc, remove_best_thres = 0, 0, 0, 0
        except Exception as e:
            logging.warning(f"Error in multi-frame ROC analysis: {str(e)}")

            if "spoof" in attacker.name and len(defense_results["spoof_error"]) > 0:
                spoof_best_TPR, spoof_best_FPR, spoof_roc_auc, spoof_best_thres = draw_roc(
                    defense_results["spoof_error"], 
                    defense_results["spoof_label"],
                    save=os.path.join(save_dir, f"roc_{attacker.name}_{defender.name}_{attack_mode}_{persistence_frame}_{range_value}{suffix}.png"))
            else:
                spoof_best_TPR, spoof_best_FPR, spoof_roc_auc, spoof_best_thres = 0, 0, 0, 0
                
            if "remove" in attacker.name and len(defense_results["remove_error"]) > 0:
                remove_best_TPR, remove_best_FPR, remove_roc_auc, remove_best_thres = draw_roc(
                    defense_results["remove_error"], 
                    defense_results["remove_label"],
                    save=os.path.join(save_dir, f"roc_{attacker.name}_{defender.name}_{attack_mode}_{persistence_frame}_{range_value}{suffix}.png"))
            else:
                remove_best_TPR, remove_best_FPR, remove_roc_auc, remove_best_thres = 0, 0, 0, 0
    

    attack_result_file = os.path.join(save_dir, f"attack_result_{attacker.name}_{perception_name}{suffix}.pkl")
    if os.path.exists(attack_result_file):
        attack_result = pickle_cache_load(attack_result_file)
    

        if len(defense_results["success"]) > 0 and "success" in attack_result and len(attack_result["success"]) > 0:

            min_len = min(len(defense_results["success"]), len(attack_result["success"]))
            success_rate = np.mean(attack_result["success"][:min_len] * defense_results["success"][:min_len])
        else:
            success_rate = 0
    else:
        logging.warning(f"Attack result file not found: {attack_result_file}")
        success_rate = 0
    

    attack_type = "persistent" if attack_persist else "single-frame"
    logging.info(f"Evaluation of defense {defender.name} against {attack_type} attack {attacker.name} on perception {perception_name} "
                f"{('(MVIG)' if is_mvig else '(Baseline)')}, success rate {success_rate:.2f}: "
                f"For spoofing attack, best TPR {spoof_best_TPR:.2f}, best FPR {spoof_best_FPR:.2f}, "
                f"ROC AUC {spoof_roc_auc:.2f}, best threshold {spoof_best_thres:.2f}; "
                f"For removal attack, best TPR {remove_best_TPR:.2f}, best FPR {remove_best_FPR:.2f}, "
                f"ROC AUC {remove_roc_auc:.2f}, best threshold {remove_best_thres:.2f}.")
    
    logging.info(f"Average IoU: {np.mean(defense_results['avg_iou']):.4f}, "
                f"Average Error: {np.mean(defense_results['avg_error']):.4f}, "
                f"AP@0.5: {np.mean(defense_results['ap_50']):.4f}")
    

    if attack_mode in ["RC", "RC+"] and attack_persist and is_mvig:
        logging.info(f"MVIG RC multi-frame attack statistics - total cases: {total_attack_cases}, "
                    f"interrupted cases: {interrupted_cases}, "
                    f"total attacked frames: {total_attack_frames}, "
                    f"effective attacked frames: {effective_attack_frames}, "
                    f"effective-frame ratio: {effective_attack_frames/max(total_attack_frames, 1):.2f}")
    

    result = {
        "success_rate": success_rate,
        "spoof_best_TPR": spoof_best_TPR,
        "spoof_best_FPR": spoof_best_FPR,
        "spoof_roc_auc": spoof_roc_auc,
        "spoof_best_thres": spoof_best_thres,
        "remove_best_TPR": remove_best_TPR,
        "remove_best_FPR": remove_best_FPR,
        "remove_roc_auc": remove_roc_auc,
        "remove_best_thres": remove_best_thres,
        "avg_iou": np.mean(defense_results["avg_iou"]),
        "avg_error": np.mean(defense_results["avg_error"]),
        "ap_50": np.mean(defense_results["ap_50"]),
        # Add summary statistics
        "total_attack_cases": total_attack_cases,
        "interrupted_cases": interrupted_cases,
        "total_attack_frames": total_attack_frames,
        "effective_attack_frames": effective_attack_frames,
        "effective_frame_ratio": effective_attack_frames/max(total_attack_frames, 1)
    }
    
    return result


def evaluate_defender_with_threshold(attacker, defender, perception_name, is_mvig, save_dir, suffix):
    """
    Generic evaluation helper for threshold-based defenders such as ROBOSAC, CPGuard, and GCP.
    It computes TPR, FPR, and success rate under a fixed threshold.
    """
    defender_name = defender.name.upper()
    logging.info(f"Using threshold-based evaluation for {defender_name} defender against {attacker.name}")
    

    tp_count = 0
    fp_count = 0
    tn_count = 0
    fn_count = 0
    success_count = 0
    total_cases = 0
    valid_cases = 0
    

    difference_scores = []
    true_labels = []
    

    case_difference_scores = {}
    case_attack_labels = {}
    
    @attack_case_iterator
    def threshold_evaluation_processor(attacker, defender, perception_name, is_mvig=False, case_id=None, case=None, data_dir=None, attack_id=None, attack=None):
        nonlocal tp_count, fp_count, tn_count, fn_count, success_count, total_cases, valid_cases
        nonlocal difference_scores, true_labels, case_difference_scores, case_attack_labels
        

        total_cases += 1
        logging.info(f"Processing case {attack_id} (total cases so far: {total_cases})")
        

        if "early" in attacker.name:
            defense_file = os.path.join(data_dir, f"{defender.name}_{perception_name}{suffix}.pkl")
        else:
            defense_file = os.path.join(data_dir, f"{defender.name}{suffix}.pkl")
        
        logging.info(f"Looking for defense file: {defense_file}")
        if not os.path.exists(defense_file):
            logging.warning(f"Defense file not found: {defense_file}")

            fn_count += 1
            return
        

        valid_cases += 1
        logging.info(f"Found valid defense file for case {attack_id} (valid cases so far: {valid_cases})")
            
        metrics = pickle_cache_load(defense_file)
        
        attacker_vehicle_id = attack["attack_meta"]["attacker_vehicle_id"]
        victim_vehicle_id = attack["attack_meta"]["victim_vehicle_id"]
        
        logging.info(f"Attacker vehicle ID: {attacker_vehicle_id}, Victim vehicle ID: {victim_vehicle_id}")
        

        if "attack_frame_ids" in attack["attack_meta"]:
            attack_frame_ids = attack["attack_meta"]["attack_frame_ids"]
            is_persistent_attack = len(attack_frame_ids) > 1
        else:
            attack_frame_ids = [9]
            is_persistent_attack = False
        
        logging.info(f"Attack frame IDs: {attack_frame_ids}, Persistent attack: {is_persistent_attack}")
        

        if case_id not in case_difference_scores:
            case_difference_scores[case_id] = {}
            case_attack_labels[case_id] = {}
        

        frame_detection_results = []
        

        success_detected = False
        

        if "all_detected_attackers" in metrics[0] and str(attacker_vehicle_id) in metrics[0]["all_detected_attackers"]:
            success_detected = True
            tp_count += 1
            logging.info(f"Case {attack_id}: Successfully detected attacker from all_detected_attackers")
        

        for frame_id in attack_frame_ids:
            logging.info(f"Checking frame {frame_id} for case {attack_id}")
            frame_success = False
            

            if "detected_attackers" in metrics[frame_id] and str(attacker_vehicle_id) in metrics[frame_id]["detected_attackers"]:
                frame_success = True
                if not success_detected:
                    success_detected = True
                    tp_count += 1
                    logging.info(f"Case {attack_id}: Successfully detected attacker in frame {frame_id}")
            

            for ego_id, ego_metrics in metrics[frame_id].items():
                if not isinstance(ego_metrics, dict):
                    continue
                

                if "classification_results" in ego_metrics and "classification_labels" in ego_metrics:
                    for collab_id, is_classified_as_attacker in ego_metrics["classification_results"].items():

                        is_actual_attacker = int(collab_id) == attacker_vehicle_id
                        

                        if collab_id not in case_difference_scores[case_id]:
                            case_difference_scores[case_id][collab_id] = []
                            case_attack_labels[case_id][collab_id] = []
                        

                        if "difference_scores" in ego_metrics and collab_id in ego_metrics["difference_scores"]:

                            diff_score = ego_metrics["difference_scores"][collab_id]
                            difference_scores.append(diff_score)
                            true_labels.append(1 if is_actual_attacker else 0)
                            

                            case_difference_scores[case_id][collab_id].append(diff_score)
                            case_attack_labels[case_id][collab_id].append(1 if is_actual_attacker else 0)


                            if is_classified_as_attacker and is_actual_attacker:

                                frame_success = True

                                if not success_detected and int(ego_id) == victim_vehicle_id:
                                    success_detected = True
                                    tp_count += 1
                                    logging.info(f"Case {attack_id}: Successfully detected attacker with classification result (diff score: {diff_score:.4f})")
                            elif is_classified_as_attacker and not is_actual_attacker:
                                fp_count += 1
                            elif not is_classified_as_attacker and not is_actual_attacker:
                                tn_count += 1
                            elif not is_classified_as_attacker and is_actual_attacker:

                                if not success_detected and int(ego_id) == victim_vehicle_id:
                                    fn_count += 1
                

                elif "difference_scores" in ego_metrics:
                    for collab_id, diff_score in ego_metrics["difference_scores"].items():

                        is_actual_attacker = int(collab_id) == attacker_vehicle_id

                        is_classified_as_attacker = diff_score > defender.difference_threshold
                        

                        if collab_id not in case_difference_scores[case_id]:
                            case_difference_scores[case_id][collab_id] = []
                            case_attack_labels[case_id][collab_id] = []
                        

                        case_difference_scores[case_id][collab_id].append(diff_score)
                        case_attack_labels[case_id][collab_id].append(1 if is_actual_attacker else 0)


                        difference_scores.append(diff_score)
                        true_labels.append(1 if is_actual_attacker else 0)
                        

                        if is_classified_as_attacker and is_actual_attacker:

                            frame_success = True

                            if not success_detected and int(ego_id) == victim_vehicle_id:
                                success_detected = True
                                tp_count += 1
                                logging.info(f"Case {attack_id}: Successfully detected attacker with difference score {diff_score:.4f}")
                        elif is_classified_as_attacker and not is_actual_attacker:
                            fp_count += 1
                        elif not is_classified_as_attacker and not is_actual_attacker:
                            tn_count += 1
                        elif not is_classified_as_attacker and is_actual_attacker:

                            if not success_detected and int(ego_id) == victim_vehicle_id:
                                fn_count += 1
            

            frame_detection_results.append(frame_success)
        

        if not success_detected:
            fn_count += 1
            logging.warning(f"Case {attack_id}: Failed to detect attacker!")
        else:

            success_count += 1
    

    threshold_evaluation_processor(attacker, defender, perception_name, is_mvig)
    

    logging.info(f"{defender_name} Defense debug info - threshold: {defender.difference_threshold}")
    logging.info(f"Detailed detection results - Total cases: {total_cases}, Valid cases: {valid_cases}")
    logging.info(f"Total attackers: {tp_count + fn_count}, Detected: {tp_count}, "
                f"Missed: {fn_count}, False alarms: {fp_count}")
    

    tpr = tp_count / max(tp_count + fn_count, 1)
    fpr = fp_count / max(fp_count + tn_count, 1)
    

    accuracy = (tp_count + tn_count) / max(tp_count + tn_count + fp_count + fn_count, 1)
    precision = tp_count / max(tp_count + fp_count, 1)
    recall = tpr
    f1 = 2 * (precision * recall) / max(precision + recall, 1e-10)
    

    success_rate = success_count / max(valid_cases, 1)
    

    best_TPR = 0
    best_FPR = 0 
    roc_auc = 0
    best_thres = defender.difference_threshold
    

    if not attack_persist:

        if len(difference_scores) > 0 and len(set(true_labels)) > 1:
            try:

                persistence_frame = "persistence_" + str(persistence)
                range_value = "range_" + str(range_limit)
                best_TPR, best_FPR, roc_auc, best_thres = draw_roc(
                    np.array(difference_scores), 
                    np.array(true_labels),
                    save=os.path.join(save_dir, f"roc_lidar_spoof_{attacker.name}_{defender.name}_{attack_mode}_{persistence_frame}_{range_value}{suffix}.png")
                )
                logging.info(f"Single-frame ROC analysis - AUC: {roc_auc:.4f}, Best threshold: {best_thres:.4f}")
            except Exception as e:
                logging.warning(f"Could not draw ROC curve: {str(e)}")
    else:

        try:

            all_frame_scores = []
            all_frame_labels = []
            all_frame_case_ids = []
            

            for case_id, collab_data in case_difference_scores.items():
                for collab_id, frame_scores in collab_data.items():
                    if frame_scores:

                        for frame_id, score in enumerate(frame_scores):
                            all_frame_scores.append(score)
                            all_frame_labels.append(case_attack_labels[case_id][collab_id][frame_id])
                            all_frame_case_ids.append(f"{case_id}_{collab_id}")
            

            if all_frame_scores and len(set(all_frame_labels)) > 1:
                persistence_frame = "persistence_" + str(persistence)
                range_value = "range_" + str(range_limit)
                best_TPR, best_FPR, roc_auc, best_thres = draw_roc(
                    np.array(all_frame_scores),
                    np.array(all_frame_labels),
                    save=os.path.join(save_dir, f"roc_lidar_spoof_{attacker.name}_{defender.name}_{attack_mode}_{persistence_frame}_{range_value}{suffix}.png"),
                    multi_frame=True,
                    case_ids=all_frame_case_ids
                )
                logging.info(f"Multi-frame ROC analysis - AUC: {roc_auc:.4f}, Best threshold: {best_thres:.4f}")
            else:
                logging.warning("Not enough multi-frame data to draw ROC curve")
        except Exception as e:
            logging.warning(f"Could not draw multi-frame ROC curve: {str(e)}")
    

    defense_results = {
        "success_rate": success_rate,
        "tpr": best_TPR,
        "fpr": best_FPR,
        "roc_auc": roc_auc,
        "tp_count": tp_count,
        "fp_count": fp_count,
        "tn_count": tn_count, 
        "fn_count": fn_count,
        "total_cases": total_cases,
        "valid_cases": valid_cases,
        "difference_scores": np.array(difference_scores if not attack_persist else all_frame_scores),
        "true_labels": np.array(true_labels if not attack_persist else all_frame_labels)
    }
    

    persistence_frame = "persistence_" + str(persistence)
    range_value = "range_" + str(range_limit)
    save_file = os.path.join(save_dir, f"defense_result_{attacker.name}_{defender.name}_{attack_mode}_{persistence_frame}_{range_value}{suffix}.pkl")
    
    pickle_cache_dump(defense_results, save_file)
    

    logging.info(f"{defender_name} Defense Evaluation against {attacker.name} on {attack_mode} "
                f"{('(MVIG)' if is_mvig else '(Baseline)')}: "
                f"Success Rate: {success_rate:.4f}, TPR: {tpr:.4f}, FPR: {fpr:.4f}")
    
    logging.info(f"Detection counts - TP: {tp_count}, FP: {fp_count}, TN: {tn_count}, FN: {fn_count}, "
                f"Total cases: {total_cases}, Valid cases: {valid_cases}")
    
    

    result = {
        "success_rate": success_rate,
        "spoof_best_TPR": tpr,
        "spoof_best_FPR": fpr,
        "spoof_roc_auc": roc_auc,
        "spoof_best_thres": defender.difference_threshold,
        "remove_best_TPR": tpr,
        "remove_best_FPR": fpr,
        "remove_roc_auc": roc_auc,
        "remove_best_thres": defender.difference_threshold,
        "avg_iou": np.array([0]),
        "avg_error": np.array([0]),
        "ap_50": np.array([0])
    }
    
    return result


def create_semantic_visualization(grid_map, pred_box=None, case=None, map_name=None, data_dir="./", suffix=""):
    """
    Create a semantic grid-map visualization.
    This simplified version keeps only the heatmap view.

    Args:
    - grid_map: Grid probability map
    - pred_box: Unused
    - case: Unused
    - map_name: Optional map name used in the title
    - data_dir: Output directory
    - suffix: Output filename suffix
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np
    import os
    from scipy.ndimage import gaussian_filter
    import warnings
    

    def enhance_grid_map(heatmap, sigma=4.0, contrast=2.5, threshold=0.3):

        smoothed = gaussian_filter(heatmap, sigma=sigma)
        

        if smoothed.max() > 0:
            smoothed = smoothed / smoothed.max()
        

        enhanced = np.power(smoothed, 1.0/contrast)
        

        enhanced[enhanced < threshold] = 0
        

        if enhanced.max() > 0:
            enhanced = enhanced / enhanced.max()
            

        if enhanced.max() - enhanced.min() < 0.1:

            enhanced = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min() + 1e-8)
            
        return enhanced
    

    def normalize_grid_map(heatmap):

        normalized = heatmap.copy()
        if normalized.max() > 0:
            normalized = normalized / normalized.max()
        return normalized
    

    grid_size = grid_map.shape
    enhanced_map = enhance_grid_map(grid_map, sigma=4.0, contrast=2.5, threshold=0.3)
    

    normalized_map = normalize_grid_map(grid_map)
    


    risk_colors = [
        (0.0, (1.0, 1.0, 1.0, 0.0)),
        (0.2, (0.18, 0.21, 0.38, 0.6)),
        (0.3, (0.24, 0.43, 0.65, 0.7)),
        (0.4, (0.25, 0.63, 0.82, 0.8)),
        (0.5, (0.22, 0.73, 0.70, 0.85)),
        (0.6, (0.39, 0.82, 0.51, 0.9)),
        (0.7, (0.75, 0.87, 0.31, 0.95)),
        (0.8, (0.94, 0.75, 0.19, 0.95)),
        (0.9, (0.96, 0.43, 0.17, 0.98)),
        (1.0, (0.75, 0.15, 0.18, 1.0))
    ]
    
    risk_cmap = mcolors.LinearSegmentedColormap.from_list('risk_cmap', risk_colors)
    

    def plot_and_save_heatmap(heatmap_data, output_suffix, title_suffix=""):

        plt.figure(figsize=(12, 10), facecolor='white')
        ax = plt.gca()
        ax.set_facecolor('white')
        

        ax.grid(False)
        

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('lightgray')
            spine.set_linewidth(0.5)
        

        heatmap = ax.imshow(heatmap_data, 
                            cmap=risk_cmap, 
                            interpolation='bilinear',
                            extent=[0, grid_size[1], grid_size[0], 0],
                            zorder=2)
        

        data_min, data_max = heatmap_data.min(), heatmap_data.max()
        data_range = data_max - data_min
        

        if data_range > 0.1:


            actual_levels = []
            standard_levels = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            standard_colors = ['#303264', '#3F6EA5', '#40A0D0', '#38BAB3', '#63D183', '#BFDD4F', '#F0BF30', '#F56E2B']
            

            for level in standard_levels:
                if data_min <= level <= data_max:
                    actual_levels.append(level)
            

            if actual_levels:
                risk_levels = actual_levels

                contour_colors = [standard_colors[standard_levels.index(level)] for level in risk_levels]
                

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    plt.contour(heatmap_data, 
                               levels=risk_levels, 
                               colors=contour_colors,
                               linewidths=1.5, 
                               zorder=3)
            else:

                logging.info(f"No standard contour levels found in data range ({data_min:.3f}-{data_max:.3f}), using percentiles instead.")

                percentiles = [20, 30, 40, 60, 75, 90]
                risk_levels = [np.percentile(heatmap_data[heatmap_data > 0], p) for p in percentiles]
                
                if len(risk_levels) > 0 and max(risk_levels) > min(risk_levels):
                    contour_colors = standard_colors[:len(percentiles)]
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        plt.contour(heatmap_data, 
                                  levels=risk_levels, 
                                  colors=contour_colors,
                                  linewidths=1.5, 
                                  zorder=3)
        else:
            logging.info(f"Data range too small ({data_min:.3f}-{data_max:.3f}), skipping contour lines.")
        


        if data_max > 0:

            tick_count = 6
            tick_locs = np.linspace(0, data_max, tick_count)
            tick_labels = []
            

            risk_level_names = ['None', 'Very Low', 'Low', 'Medium', 'High', 'Critical']
            
            for i, loc in enumerate(tick_locs):

                if i < len(risk_level_names):
                    level_name = risk_level_names[i]
                    tick_labels.append(f"{level_name}")
                else:
                    tick_labels.append("")
        else:

            tick_locs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            tick_labels = ['None', 'Very Low', 'Low', 'Medium', 'High', 'Critical']
        
        cbar = plt.colorbar(heatmap, ax=ax, orientation='vertical', 
                           pad=0.01, ticks=tick_locs)
        cbar.set_label('Attack Risk Level', fontsize=16, weight='bold')
        cbar.ax.set_yticklabels(tick_labels, fontsize=8)
        cbar.ax.tick_params(axis='y', which='major', pad=8)
        

        map_str = f" - {map_name}" if map_name else ""
        plt.title(f"Attack Risk Map{map_str}{title_suffix}", 
                  fontsize=16, 
                  pad=10,
                  weight='bold')
        

        ax.set_xticks([])
        ax.set_yticks([])
        

        ax.set_frame_on(True)
        

        output_file = os.path.join(data_dir, f"vulnerability_heatmap{suffix}{output_suffix}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Vulnerability heatmap saved to {output_file}")
    

    plot_and_save_heatmap(enhanced_map, "_enhanced")
    

    plot_and_save_heatmap(normalized_map, "_normalized", " (Normalized)")


def world_to_grid_coords(world_x, world_y, grid_shape=(200, 200), range_limit=50):
    """Convert world coordinates to grid coordinates in the victim frame."""
    x_range = [-range_limit, range_limit]
    y_range = [-range_limit, range_limit]

    norm_x = (world_x - x_range[0]) / (x_range[1] - x_range[0]) * 2 - 1
    norm_y = (world_y - y_range[0]) / (y_range[1] - y_range[0]) * 2 - 1
    

    grid_x = int((norm_x + 1) / 2 * (grid_shape[1] - 1))
    grid_y = int((norm_y + 1) / 2 * (grid_shape[0] - 1))
    

    grid_x = max(1, min(grid_x, grid_shape[1]-2))
    grid_y = max(1, min(grid_y, grid_shape[0]-2))
    
    return grid_x, grid_y


def main():

    device = "cuda:0"
    global range_limit
    defense_only = False
    logging.info(
        "Runtime overrides - attack_mode=%s, attack_type=%s, persistence=%d, visualize=%s, cache_size=%d, defenders=%s, model_path=%s, cuda_visible_devices=%s",
        attack_mode,
        eval_attack_type,
        persistence,
        is_visualize,
        eval_cache_size,
        ",".join(requested_defenders),
        model_path,
        os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    )


    # logging.info("######################## Perception on normal cases ########################")
    # normal_perception()
    
    # logging.info("######################## Calculate occupancy map ########################")
    # lidar_seg_api = SqueezeSegInterface()
    # occupancy_map(lidar_seg_api)
    model_filename = os.path.basename(model_path)

    match = re.search(r'best_mvig_model_(\w+)_(\d+)\.pth', model_filename)
    if match:
        attack_type = match.group(1)
        range_limit = int(match.group(2))
        logging.info(f"Parsed from model path: attack_type={attack_type}, range_limit={range_limit}")
    else:

        attack_type = "spoof"
        range_limit = 15
        logging.warning(f"Could not parse attack_type and range_limit from {model_path}, using defaults: attack_type={attack_type}, range_limit={range_limit}")
    

    mvig_model = MVIGNet(attack_type=attack_type, node_dim=100, hidden_dim=64, num_layers=3, grid_size=(200, 200)).to(device)
    checkpoint = torch.load(os.path.join(result_dir, model_path))
    mvig_model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    mvig_model.eval()
    logging.info(f"Loaded MVIG model with attack_type={attack_type}, range_limit={range_limit}")
    

    random_mvig_model = MVIGNet(attack_type=attack_type, node_dim=100, hidden_dim=64, num_layers=3, grid_size=(200, 200), random_baseline=True).to(device)
    random_mvig_model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    random_mvig_model.eval()
    logging.info("Loaded random baseline MVIG model")


    results = {
        'baseline': {},
        'mvig': {},
        'defense': {}
    }


    for attacker_name, attacker_class in attacker_dict.items():
        logging.info(f"\n######################## Evaluating {attacker_name} ########################")
        

        logging.info("Evaluating random baseline attack...")
        

        baseline_attacker = attacker_class
        baseline_attacker.mvig_model = random_mvig_model

        

        if not defense_only:
            attack_perception(baseline_attacker)
        

        if "early" in attacker_name:
            for perception_name in ["pointpillar_early", "pointpillar_intermediate"]:
                key = f"{attacker_name}_{perception_name}"
                results['baseline'][key] = attack_evaluation(baseline_attacker, perception_name, is_mvig=True)
                logging.info(f"Random Baseline {key} - Attack Success: {results['baseline'][key]['attack_success']:.4f}, "
                           f"Detection Rate: {results['baseline'][key]['detection_rate']:.4f}, "
                           f"AP@0.5 Decrease: {results['baseline'][key]['avg_delta_ap']:.4f}")
                

                logging.info(f"Evaluating defenses against random baseline {key}...")
                for defender_name, defender in defender_dict.items():
                    defense_key = f"{attacker_name}_{defender_name}_{perception_name}"
                    defense(baseline_attacker, defender, perception_name, is_mvig=True)
                    results['defense'][f"baseline_{defense_key}"] = defense_evaluation(baseline_attacker, defender, perception_name, is_mvig=True)
        else:
            if not defense_only:
                results['baseline'][attacker_name] = attack_evaluation(baseline_attacker, baseline_attacker.perception.name, is_mvig=True)
                logging.info(f"Random Baseline {attacker_name} - Attack Success: {results['baseline'][attacker_name]['attack_success']:.4f}, "
                        f"Detection Rate: {results['baseline'][attacker_name]['detection_rate']:.4f}, "
                        f"AP@0.5 Decrease: {results['baseline'][attacker_name]['avg_delta_ap']:.4f}")
            

            logging.info(f"Evaluating defenses against random baseline {attacker_name}...")

            selected_defender_names = ["cad", "robosac", "cpguard", "gcp"]
            selected_defenders = {name: defender for name, defender in defender_dict.items() 
                                if defender.name in selected_defender_names}


            for defender_name, defender in selected_defenders.items():
                defense_key = f"{attacker_name}_{defender_name}"
                defense(baseline_attacker, defender, baseline_attacker.perception.name, is_mvig=True)
                results['defense'][f"baseline_{defense_key}"] = defense_evaluation(baseline_attacker, defender, baseline_attacker.perception.name, is_mvig=True)

       

        if attack_mode == "RC":
            if "spoof" in attacker_name or "remove" in attacker_name:
                logging.info("Evaluating MVIG-optimized attack...")
                

                mvig_attacker = attacker_class
                mvig_attacker.mvig_model = mvig_model
                

                if not defense_only:
                    attack_perception(mvig_attacker)
                

                if "early" in attacker_name:
                    for perception_name in ["pointpillar_early", "pointpillar_intermediate"]:
                        key = f"{attacker_name}_{perception_name}"
                        results['mvig'][key] = attack_evaluation(mvig_attacker, perception_name, is_mvig=True)
                        logging.info(f"MVIG {key} - Attack Success: {results['mvig'][key]['attack_success']:.4f}, "
                                f"Detection Rate: {results['mvig'][key]['detection_rate']:.4f}, "
                                f"AP@0.5 Decrease: {results['mvig'][key]['avg_delta_ap']:.4f}")
                        

                        logging.info(f"Evaluating defenses against MVIG {key}...")
                        for defender_name, defender in defender_dict.items():
                            defense_key = f"{attacker_name}_{defender_name}_{perception_name}"
                            defense(mvig_attacker, defender, perception_name, is_mvig=True)
                            results['defense'][f"mvig_{defense_key}"] = defense_evaluation(mvig_attacker, defender, perception_name, is_mvig=True)
                else:
                    results['mvig'][attacker_name] = attack_evaluation(mvig_attacker, mvig_attacker.perception.name, is_mvig=True)
                    logging.info(f"MVIG {attacker_name} - Attack Success: {results['mvig'][attacker_name]['attack_success']:.4f}, "
                            f"Detection Rate: {results['mvig'][attacker_name]['detection_rate']:.4f}, "
                            f"AP@0.5 Decrease: {results['mvig'][attacker_name]['avg_delta_ap']:.4f}")
                    

                    logging.info(f"Evaluating defenses against MVIG {attacker_name}...")

                    selected_defender_names = ["cad", "robosac", "cpguard", "gcp"]
                    selected_defenders = {name: defender for name, defender in defender_dict.items() 
                                        if defender.name in selected_defender_names}


                    for defender_name, defender in selected_defenders.items():
                        defense_key = f"{attacker_name}_{defender_name}"
                        defense(mvig_attacker, defender, mvig_attacker.perception.name, is_mvig=True)
                        results['defense'][f"mvig_{defense_key}"] = defense_evaluation(
                            mvig_attacker, defender, mvig_attacker.perception.name, is_mvig=True
                            )


    logging.info("\n######################## Summary Report ########################")

    for attacker_name in results['baseline'].keys():
        if attacker_name in results['mvig']:
            logging.info(f"\nComparison for {attacker_name}:")
            logging.info("Random Baseline vs MVIG Attack Performance:")
            for metric in ['attack_success', 'detection_rate', 'avg_delta_ap']:
                baseline_value = results['baseline'][attacker_name][metric]
                mvig_value = results['mvig'][attacker_name][metric]
                

                if abs(baseline_value) < 0.001:
                    if abs(mvig_value) < 0.001:
                        improvement = "0.0"
                    else:
                        improvement = "N/A"
                else:
                    improvement = f"{((mvig_value - baseline_value) / baseline_value) * 100:+.1f}%"
                    

                if metric == 'avg_delta_ap':
                    logging.info(f"Average AP@0.5 Decrease: {baseline_value:.4f} -> {mvig_value:.4f} ({improvement})")
                else:
                    logging.info(f"{metric}: {baseline_value:.4f} -> {mvig_value:.4f} ({improvement})")

            for defender_name, defender in defender_dict.items():
                logging.info(f"\nComparison for {defender_name}:")
                logging.info("Defense Performance:")
                defense_key = f"{attacker_name}_{defender_name}"
                for metric in ['success_rate', 'spoof_best_TPR', 'spoof_best_FPR', 'remove_best_TPR', 'remove_best_FPR']:
                    baseline_value = results['defense'][f"baseline_{defense_key}"][metric]
                    mvig_value = results['defense'][f"mvig_{defense_key}"][metric]
                    

                    if abs(baseline_value) < 0.001:
                        if abs(mvig_value) < 0.001:
                            improvement = "0.0"
                        else:
                            improvement = "N/A"
                    else:
                        improvement = f"{((mvig_value - baseline_value) / baseline_value) * 100:+.1f}%"
                        
                    logging.info(f"defense_{metric}: {baseline_value:.4f} -> {mvig_value:.4f} ({improvement})")
    
  
    # Save results
    with open(os.path.join(result_dir, 'evaluation_results.pkl'), 'wb') as f:
        pickle.dump(results, f)


    if attack_mode in ["RC", "RC+"] and attack_persist:
        total_cases = 0
        interrupted_cases = 0
        total_frames = 0
        effective_frames = 0
        
        for key, value in results['defense'].items():
            if key.startswith("mvig_") and "total_attack_cases" in value:
                total_cases += value["total_attack_cases"]
                interrupted_cases += value["interrupted_cases"]
                total_frames += value["total_attack_frames"]
                effective_frames += value["effective_attack_frames"]
        
        if total_cases > 0:
            logging.info("\nOverall MVIG RC multi-frame attack statistics:")
            logging.info(f"Total cases: {total_cases}")
            logging.info(f"Interrupted cases: {interrupted_cases} ({interrupted_cases/total_cases*100:.1f}%)")
            logging.info(f"Total attacked frames: {total_frames}")
            logging.info(f"Effective attacked frames: {effective_frames}")
            logging.info(f"Effective-frame ratio: {effective_frames/max(total_frames, 1):.2f}")


if __name__ == "__main__":
    main()
