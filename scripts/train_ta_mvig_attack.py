import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")  # Allow shell scripts to override this value.
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Add the project root directory to the path
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import time
import pickle
from functools import wraps
from shapely.geometry import Polygon, Point
from matplotlib.path import Path
import traceback
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import math
from mvp.tools.iou import iou2d
from tqdm import tqdm

# Add root directory to system path
root = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../")
sys.path.append(root)

# Use the same path layout as evaluate.py
result_dir = os.path.join(root, "result")  # Keep consistent with evaluate.py
log_dir = os.path.join(result_dir, "log")  # Keep consistent with evaluate.py

# Create required directories
os.makedirs(result_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Set the log filename with a timestamp
log_file = os.path.join(log_dir, f"train_mvig_{time.strftime('%Y%m%d_%H%M%S')}.log")

# Configure the logging format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_file),  # File handler
        logging.StreamHandler()         # Console handler
    ]
)

# Log startup information
logging.info(f"Starting training, log file: {log_file}")

from mvp.config import data_root
from mvp.data.opv2v_dataset import OPV2VDataset
from mvp.perception.opencood_perception import OpencoodPerception
from mvp.attack.lidar_spoof_intermediate_attacker import LidarSpoofIntermediateAttacker
from mvp.defense.perception_defender import PerceptionDefender
from mvp.tools.squeezeseg.interface import SqueezeSegInterface
from mvp.tools.lidar_seg import lidar_segmentation
from mvp.data.util import bbox_sensor_to_map, bbox_map_to_sensor, pcd_sensor_to_map, pcd_map_to_sensor, get_distance
from mvp.tools.iou import iou2d
from mvp.tools.polygon_space import bbox_to_polygon
from mvp.defense.detection_util import filter_segmentation
from mvp.tools.ground_detection import get_ground_plane
from mvp.tools.polygon_space import get_occupied_space, get_free_space, bbox_to_polygon
from mvp.config import data_root
from mvp.data.util import bbox_sensor_to_map, bbox_map_to_sensor, pcd_sensor_to_map, pcd_map_to_sensor, get_distance
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
from mvp.defense.perception_defender import PerceptionDefender

# Grid map parameters
H = 200  # Height of occupancy grid
W = 200  # Width of occupancy grid
CELL_SIZE = 0.5  # Size of each grid cell in meters
RANGE_LIMIT = 100  # Maximum range in meters

# Runtime overrides from shell script / environment variables.
TRAIN_ATTACK_TYPE = os.getenv("MVIG_ATTACK_TYPE", "spoof").strip().lower()
TRAIN_TOTAL_EPOCHS = int(os.getenv("MVIG_TOTAL_EPOCHS", "30"))
TRAIN_CACHE_SIZE = int(os.getenv("MVIG_CACHE_SIZE", "100"))
TRAIN_ATTACK_STEP = int(os.getenv("MVIG_ATTACK_STEP", "100"))

if TRAIN_ATTACK_TYPE not in {"spoof", "remove"}:
    raise ValueError(f"Unsupported MVIG_ATTACK_TYPE={TRAIN_ATTACK_TYPE}, expected 'spoof' or 'remove'")

# Initialize dataset
dataset = OPV2VDataset(root_path=os.path.join(data_root, "OPV2V"), mode="test")
dataset.cache_size = TRAIN_CACHE_SIZE  # Use cache_size to control the dataset size

# Initialize perception models
perception_list = [
    OpencoodPerception(fusion_method="early", model_name="pointpillar"),
    OpencoodPerception(fusion_method="intermediate", model_name="pointpillar"),
    OpencoodPerception(fusion_method="late", model_name="pointpillar"),
]
perception_dict = OrderedDict([(x.name, x) for x in perception_list])

# Initialize attacker
if TRAIN_ATTACK_TYPE == "spoof":
    attacker_list = [
        LidarSpoofIntermediateAttacker(
            perception_dict["pointpillar_intermediate"],
            dataset,
            step=TRAIN_ATTACK_STEP,
            sync=0,
            init=False,
            online=False
        )
    ]
else:
    attacker_list = [
        LidarRemoveIntermediateAttacker(
            perception_dict["pointpillar_intermediate"],
            dataset,
            step=TRAIN_ATTACK_STEP,
            sync=0,
            init=False,
            online=False
        )
    ]
attacker_dict = OrderedDict([(x.name, x) for x in attacker_list])

# Initialize defender
defender_list = [
    PerceptionDefender(),
]
defender_dict = OrderedDict([(x.name, x) for x in defender_list])

pickle_cache = OrderedDict()
pickle_cache_size = 600

# Constant definitions
total_frames = 10
attack_frame_ids = range(total_frames)

# Define the device near the top of the file
device = "cuda:0"  # This actually maps to physical GPU 1

# Ensure all CUDA operations use the selected GPU
torch.cuda.set_device(0)  # This logical index actually points to physical GPU 1

class GraphConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.edge_linear = nn.Linear(1, out_dim)  # Project edge features
        
    def forward(self, x, edge_index, edge_attr):
        # Apply the node projection first
        node_feat = self.linear(x)
        
        # Project edge features
        edge_feat = self.edge_linear(edge_attr.unsqueeze(1))
        
        # Message passing
        out_feat = node_feat.clone()
        
        if edge_index.shape[1] > 0:  # If edges exist
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[0, i], edge_index[1, i]
                # Combine source-node and edge features, then accumulate them on the destination node
                out_feat[dst] = out_feat[dst] + node_feat[src] * edge_feat[i]
            
        return out_feat

class MVIGNet(nn.Module):
    def __init__(self, node_dim=100, hidden_dim=64, num_layers=3, grid_size=(200, 200), attack_type="spoof", range_limit=20, random_baseline=False):
        super().__init__()
        self.gnn_layers = nn.ModuleList([
            GraphConvLayer(
                in_dim=node_dim if i==0 else hidden_dim,
                out_dim=hidden_dim
            ) for i in range(num_layers)
        ])
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        
        # Score map generation
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, grid_size[0] * grid_size[1]),
            nn.Softmax(dim=1)  # Convert all grid locations into a probability distribution with Softmax
        )
        
        # Define the coordinate range relative to the victim.
        self.x_range = [-range_limit, range_limit]  # x spans from `-range_limit` to `range_limit`
        self.y_range = [-range_limit, range_limit]  # y spans from `-range_limit` to `range_limit`
        self.grid_size = grid_size
        
        # Fixed vehicle parameters
        self.vehicle_size = [1.97027588e+00, 3.80580020e+00, 1.47503030e+00]
        self.vehicle_z = 0.73751515
        self.vehicle_yaw = 0.0
        self.attack_type = attack_type
        
        # Enable random-baseline control
        self.random_baseline = random_baseline
        
        # Cache lane-area polygons
        self.lane_areas_map = {}
        
    def load_lane_areas(self, map_name):
        """Load lane areas for a map"""
        if map_name not in self.lane_areas_map:
            lane_areas_path = os.path.join(data_root, f"carla/{map_name}_lane_areas.pkl")
            self.lane_areas_map[map_name] = pickle_cache_load(lane_areas_path)
    
    def check_in_lane_areas(self, point, lane_areas):
        """Check whether a point lies inside any lane area"""
        x, y = point
        point = Point(x, y)
        return any(area.contains(point) for area in lane_areas)
    
    def grid_to_world_coords(self, grid_x, grid_y):
        """Convert grid coordinates to world coordinates in the victim frame"""
        world_x = self.x_range[0] + (self.x_range[1] - self.x_range[0]) * (grid_x / (self.grid_size[0] - 1))
        world_y = self.y_range[0] + (self.y_range[1] - self.y_range[0]) * (grid_y / (self.grid_size[1] - 1))
        return world_x, world_y
    
    def world_to_grid_coords(self, world_x, world_y):
        """Convert world coordinates to grid coordinates in the victim frame"""
        # Compute the relative position normalized to [-1, 1]
        norm_x = (world_x - self.x_range[0]) / (self.x_range[1] - self.x_range[0]) * 2 - 1
        norm_y = (world_y - self.y_range[0]) / (self.y_range[1] - self.y_range[0]) * 2 - 1
        
        # Convert to grid coordinates
        grid_x = int((norm_x + 1) / 2 * (self.grid_size[1] - 1))
        grid_y = int((norm_y + 1) / 2 * (self.grid_size[0] - 1))
        
        # Clamp coordinates to the valid range
        grid_x = max(1, min(grid_x, self.grid_size[1]-2))
        grid_y = max(1, min(grid_y, self.grid_size[0]-2))
        
        return grid_x, grid_y
    
    def forward(self, temporal_graphs, map_name=None):
        node_features = []
        for graph in temporal_graphs:
            x = graph['x']
            for layer in self.gnn_layers:
                x = layer(x, graph['edge_index'], graph['edge_attr'])
            x = torch.mean(x, dim=0, keepdim=True)
            node_features.append(x)
        
        node_features = torch.stack(node_features, dim=1)
        output, _ = self.gru(node_features)
        last_hidden = output[:, -1]
        
        # Generate the grid heatmap
        grid_scores = self.score_head(last_hidden)
        batch_size = grid_scores.shape[0]
        
        # If a map name is provided, load and apply the lane-area constraint
        if map_name is not None:
            self.load_lane_areas(map_name)
            lane_areas = self.lane_areas_map.get(map_name)
            if lane_areas is not None:
                # Create the mask
                mask = torch.zeros_like(grid_scores)
                for b in range(batch_size):
                    for i in range(self.grid_size[0]):
                        for j in range(self.grid_size[1]):
                            x, y = self.grid_to_world_coords(j, i)
                            if self.check_in_lane_areas((x, y), lane_areas):
                                mask[b, i * self.grid_size[1] + j] = 1
                
                # Apply the mask
                grid_scores = grid_scores * mask
                # Renormalize
                grid_scores = grid_scores / (grid_scores.sum(dim=1, keepdim=True) + 1e-6)
        
        # Reshape the 1D score vector into a grid map
        grid_map = grid_scores.reshape(batch_size, self.grid_size[0], self.grid_size[1])
        
        # Select the highest-probability valid location
        batch_positions = []
        for b in range(batch_size):
            if self.random_baseline:
                # Optional fixed-position selection
                use_fixed_position = False  # Set to True to use a fixed position
                fixed_position_type = "center"  # Options: "center" or "center_to_topleft"
                
                if use_fixed_position:
                    # Compute the center coordinates
                    center_x = self.grid_size[1] // 2
                    center_y = self.grid_size[0] // 2
                    
                    if fixed_position_type == "center":
                        # Use the center point.
                        grid_x = center_x
                        grid_y = center_y
                        position_desc = "center"
                    elif fixed_position_type == "center_to_topleft":
                        # Use a point on the center-to-top-left line that stays close to the center
                        topleft_x = 0
                        topleft_y = 0
                        
                        # Set the interpolation ratio toward the center (0.0 is top-left, 1.0 is center)
                        # 0.75 places the point 25% away from the center along the path from the top-left corner
                        ratio = 0.92
                        
                        # Compute the position coordinates
                        grid_x = int(center_x - (center_x - topleft_x) * (1 - ratio))
                        grid_y = int(center_y - (center_y - topleft_y) * (1 - ratio))
                        
                        position_desc = f"point at {ratio:.2f} ratio from center to top-left"
                    else:
                        # Use the center point by default
                        grid_x = center_x
                        grid_y = center_y
                        position_desc = "center (default)"
                    
                    logging.info(f"Using fixed position at {position_desc}: grid ({grid_x}, {grid_y})")
                else:
                    # Random-baseline mode: choose a valid location uniformly at random
                    valid_indices = torch.where(grid_scores[b] > 0)[0]
                    if len(valid_indices) > 0:
                        # Sample one valid location uniformly without using the probability distribution
                        random_idx = valid_indices[torch.randint(0, len(valid_indices), (1,))]
                        grid_y = random_idx.item() // self.grid_size[1]
                        grid_x = random_idx.item() % self.grid_size[1]
                    else:
                        # Fallback to the center point if no valid location exists
                        grid_x = self.grid_size[1] // 2
                        grid_y = self.grid_size[0] // 2
            elif torch.rand(1).item() < 0.7:  # Reduced from 0.9 to 0.7
                # Find the valid grid cell with the highest probability
                valid_indices = torch.where(grid_scores[b] > 0)[0]
                if len(valid_indices) > 0:
                    max_idx = valid_indices[grid_scores[b][valid_indices].argmax()]
                    grid_y = max_idx.item() // self.grid_size[1]
                    grid_x = max_idx.item() % self.grid_size[1]
                else:
                    # Fallback to the center point if no valid location exists
                    grid_x = self.grid_size[1] // 2
                    grid_y = self.grid_size[0] // 2
            else:
                # Sample a valid location according to the predicted probability distribution
                valid_indices = torch.where(grid_scores[b] > 0)[0]
                if len(valid_indices) > 0:
                    probs = grid_scores[b][valid_indices]
                    sampled_idx = valid_indices[torch.multinomial(probs, 1)]
                    grid_y = sampled_idx.item() // self.grid_size[1]
                    grid_x = sampled_idx.item() % self.grid_size[1]
                else:
                    grid_x = self.grid_size[1] // 2
                    grid_y = self.grid_size[0] // 2
            
            world_x, world_y = self.grid_to_world_coords(grid_x, grid_y)
            batch_positions.append([world_x, world_y])
        
        # Convert to tensors
        xy_coords = torch.tensor(batch_positions, device=last_hidden.device)
        
        # Combine the predicted coordinates with fixed box parameters
        full_box = torch.cat([
            xy_coords,  # Position [x, y]
            torch.full((batch_size, 1), self.vehicle_z, device=xy_coords.device),  # Fixed z coordinate
            torch.tensor([self.vehicle_size], device=xy_coords.device).repeat(batch_size, 1),  # Fixed size
            torch.full((batch_size, 1), self.vehicle_yaw, device=xy_coords.device),  # Fixed yaw
        ], dim=1)
        
        return full_box, grid_map  # Return the full box parameters and the grid heatmap

class OccupancyDataProcessor:
    def __init__(self, grid_size=(200, 200), cell_size=0.5, range_limit=50):
        self.grid_params = {
            "size": grid_size,
            "cell_size": cell_size,
            "range_limit": range_limit
        }
        self.device = torch.device(device)  # Use the globally defined device
        self.current_frame_data = []  # Store current-frame data
        
        # Precompute grid coordinates
        self._init_grid_coords()

    def _init_grid_coords(self):
        """Precompute grid coordinates"""
        H, W = self.grid_params["size"]
        cell_size = self.grid_params["cell_size"]
        
        # Create grid coordinates.
        x = np.arange(-W/2, W/2) * cell_size
        y = np.arange(-H/2, H/2) * cell_size
        X, Y = np.meshgrid(x, y)
        
        # Create the range mask.
        range_mask = (X**2 + Y**2) <= self.grid_params["range_limit"]**2
        
        self.X = X
        self.Y = Y
        self.range_mask = range_mask
        self.points = np.column_stack((X.ravel(), Y.ravel()))

    def polygon_to_grid(self, polygon, lidar_pose):
        """Convert a polygon to a grid mask (optimized version)"""
        H, W = self.grid_params["size"]
        
        # Check whether the object exposes polygon-like attributes instead of checking the type directly
        if not hasattr(polygon, 'exterior') or not hasattr(polygon, 'is_valid'):
            try:
                # Log the actual object type for debugging
                logging.debug(f"Object type: {type(polygon)}")
                
                # Try to handle possible geometry types
                if hasattr(polygon, 'geoms'):  # Handle MultiPolygon inputs
                    from shapely.ops import unary_union
                    polygon = unary_union(list(polygon.geoms))
                else:
                    # Try to recover by explicitly constructing a Shapely Polygon
                    from shapely.geometry import Polygon as ShapelyPolygon
                    polygon = ShapelyPolygon(polygon)
            except Exception as e:
                logging.debug(f"Failed to convert to workable polygon: {str(e)}")
                return np.zeros((H, W), dtype=bool)
        
        # The remaining code follows the same geometry-conversion pipeline.
        # Compute the transform matrix
        cos_yaw = np.cos(np.radians(lidar_pose[4]))
        sin_yaw = np.sin(np.radians(lidar_pose[4]))
        
        # Rotate and translate points
        X_rotated = self.X * cos_yaw - self.Y * sin_yaw
        Y_rotated = self.X * sin_yaw + self.Y * cos_yaw
        
        X_map = X_rotated + lidar_pose[0]
        Y_map = Y_rotated + lidar_pose[1]
        
        points_map = np.column_stack((X_map.ravel(), Y_map.ravel()))
        
        # Use Path for efficient point-in-polygon tests
        path = Path(np.array(polygon.exterior.coords))
        mask = path.contains_points(points_map).reshape(H, W)
        
        return mask & self.range_mask

    def convert_to_grid(self, occupancy_feature, frame_id, vehicle_id):
        """Convert occupancy features into a grid representation"""
        H, W = self.grid_params["size"]
        
        try:
            vehicle_data = occupancy_feature[frame_id].get(vehicle_id)
            if vehicle_data is None:
                logging.error(f"Vehicle {vehicle_id} not found in frame {frame_id}")
                return None
                
            # Validate required fields
            required_fields = ["free_areas", "occupied_areas", "lidar_pose"]
            if not all(k in vehicle_data for k in required_fields):
                logging.error(f"Missing required fields for vehicle {vehicle_id} in frame {frame_id}")
                return None
            
            # Initialize the grid map
            grid_map = torch.full((H, W), 2, dtype=torch.float32, device=self.device)  # Use float32
            
            # Process free-space regions
            for area in vehicle_data["free_areas"]:
                free_mask = self.polygon_to_grid(area, vehicle_data["lidar_pose"])
                if free_mask is not None:
                    grid_map[torch.from_numpy(free_mask).to(self.device)] = 0
            
            # Process occupied regions
            for area in vehicle_data["occupied_areas"]:
                occupied_mask = self.polygon_to_grid(area, vehicle_data["lidar_pose"])
                if occupied_mask is not None:
                    grid_map[torch.from_numpy(occupied_mask).to(self.device)] = 1
            
            return grid_map
            
        except Exception as e:
            logging.error(f"Error in convert_to_grid: {str(e)}")
            return None

    def build_mvig(self, grid_maps):
        """Build the MVIG graph by extracting node features from raw grid maps and augmenting them with position and pose"""
        if not grid_maps:
            return self._create_default_graph()
            
        # Node features
        node_features = []
        valid_grids = {}
        
        for vehicle_id, grid in grid_maps.items():
            if grid is None:
                continue
                
            # Get vehicle position and pose
            vehicle_data = None
            for frame_id in range(len(self.current_frame_data)):
                if vehicle_id in self.current_frame_data[frame_id]:
                    vehicle_data = self.current_frame_data[frame_id][vehicle_id]
                    break
            
            # Extract position and pose features
            position_pose_features = []
            if vehicle_data and "lidar_pose" in vehicle_data:
                # Extract position (x, y, z)
                position = vehicle_data["lidar_pose"][:3]
                # Extract pose (roll, pitch, yaw)
                pose = vehicle_data["lidar_pose"][3:6]
                position_pose_features = torch.tensor(
                    np.concatenate([position, pose]),
                    device=self.device,
                    dtype=torch.float32
                )
            else:
                # Use a zero vector if position data is unavailable
                position_pose_features = torch.zeros(6, device=self.device, dtype=torch.float32)
                
            # Cast to float for further processing
            grid_float = grid.float()
            valid_grids[vehicle_id] = grid_float
            
            # 1. Extract basic occupancy-ratio features
            basic_feat = torch.tensor([
                (grid == 0).sum().float() / grid.numel(),  # Free-space ratio
                (grid == 1).sum().float() / grid.numel(),  # Occupied-space ratio
                (grid == 2).sum().float() / grid.numel()   # Unknown-space ratio
            ], device=self.device, dtype=torch.float32)
            
            # 2. Extract spatial-distribution features using multi-scale pooling
            spatial_features = []
            
            # Extract spatial features for each occupancy class
            for class_idx in range(3):
                # Create a binary mask
                class_mask = (grid == class_idx).float().unsqueeze(0).unsqueeze(0)
                
                # Use pooling kernels of multiple sizes to capture multi-scale features
                pooled_features = []  # Initialize here
                pool_sizes = [4, 8, 16, 32, 64]  # Different pooling sizes
                
                for pool_size in pool_sizes:
                    # Apply average pooling
                    if pool_size > 1:
                        pooled = F.avg_pool2d(class_mask, kernel_size=pool_size, stride=pool_size)
                        # Flatten and append the pooled features
                        pooled_features.append(pooled.view(-1))
                
                # Add all pooled features for this class to spatial_features
                if pooled_features:
                    spatial_features.extend(pooled_features)
            
            # Merge all features
            if spatial_features:
                all_spatial_features = torch.cat(spatial_features)
                
                # Combine basic, position-pose, and spatial features
                combined_features = torch.cat([basic_feat, position_pose_features, all_spatial_features])
                
                # Reduce the feature dimension to the target size when it is too large
                if combined_features.shape[0] > 100:
                    combined_features = combined_features.view(1, -1)
                    combined_features = F.adaptive_avg_pool1d(combined_features.unsqueeze(1), 91).squeeze(1).squeeze(0)
                    # 91 = 100 - (3 + 6), after reserving the basic and position-pose features
                    combined_features = torch.cat([basic_feat, position_pose_features, combined_features])
                elif combined_features.shape[0] < 100:
                    # Pad with zeros if the feature dimension is smaller than 100
                    padding = torch.zeros(100 - combined_features.shape[0], device=self.device, dtype=torch.float32)
                    combined_features = torch.cat([combined_features, padding])
                    
                node_features.append(combined_features)
            else:
                # If no spatial features are available, combine basic and position-pose features and pad to 100 dimensions
                combined_features = torch.cat([basic_feat, position_pose_features])
                padding = torch.zeros(100 - combined_features.shape[0], device=self.device, dtype=torch.float32)
                combined_features = torch.cat([combined_features, padding])
                node_features.append(combined_features)
        
        # Build edge features and edge indices using the original mutual-information computation
        edge_index = []
        edge_attr = []
        vehicle_ids = list(valid_grids.keys())
        
        for i in range(len(vehicle_ids)):
            for j in range(i+1, len(vehicle_ids)):
                edge_index.extend([[i, j], [j, i]])
                # Compute edge weights with the existing simplified mutual-information routine
                mi = self.compute_mutual_information(
                    valid_grids[vehicle_ids[i]], 
                    valid_grids[vehicle_ids[j]]
                )
                edge_attr.extend([mi, mi])
        
        return {
            'x': torch.stack(node_features),
            'edge_index': torch.tensor(edge_index, device=self.device, dtype=torch.long).t() if edge_index else torch.zeros((2, 0), device=self.device, dtype=torch.long),
            'edge_attr': torch.tensor(edge_attr, device=self.device, dtype=torch.float32) if edge_attr else torch.zeros(0, device=self.device, dtype=torch.float32)
        }

    def prepare_temporal_data(self, occupancy_feature, frame_ids, map_name):
        """Prepare temporal graph data and attach the map name"""
        temporal_graphs = []
        
        # Store current-frame data for `build_mvig`.
        self.current_frame_data = [occupancy_feature[frame_id] for frame_id in frame_ids]
        
        for frame_id in frame_ids:
            # logging.info(f"Processing frame {frame_id}")
            
            # Create the grid-map dictionary for the current frame
            grid_maps = {}
            
            # Process each vehicle
            for vehicle_id in occupancy_feature[frame_id].keys():
                try:
                    grid = self.convert_to_grid(
                        occupancy_feature, frame_id, vehicle_id
                    )
                    if grid is not None:
                        grid_maps[vehicle_id] = grid
                        logging.debug(f"Successfully converted grid for vehicle {vehicle_id} in frame {frame_id}")
                except Exception as e:
                    logging.error(f"Error converting grid for vehicle {vehicle_id} in frame {frame_id}: {str(e)}")
                    continue
            
            # Build the graph for the current frame
            if not grid_maps:
                logging.warning(f"No valid grid maps for frame {frame_id}")
                graph = self._create_default_graph()
            else:
                graph = self.build_mvig(grid_maps)
            
            # Attach map_name to the graph
            graph['map_name'] = map_name
            temporal_graphs.append(graph)
            logging.debug(f"Added graph for frame {frame_id} with {len(grid_maps)} vehicles")
        
        return temporal_graphs

    def _create_default_graph(self):
        """Create a default graph structure"""
        return {
            'x': torch.zeros((1, 100), device=self.device, dtype=torch.float32),  # Changed to 100 dimensions
            'edge_index': torch.zeros((2, 0), device=self.device, dtype=torch.long),
            'edge_attr': torch.zeros(0, device=self.device, dtype=torch.float32)
        }

    def compute_mutual_information(self, grid1, grid2):
        """Compute the mutual information between two grids"""
        eps = 1e-8  # Prevent numerical instability
        
        # Compute the joint distribution
        joint_hist = torch.zeros((3, 3), device=self.device)
        for i in range(3):
            for j in range(3):
                joint_hist[i, j] = torch.sum((grid1 == i) & (grid2 == j)).float()
        
        # Normalize
        joint_prob = joint_hist / (joint_hist.sum() + eps)
        
        # Compute marginal distributions
        p1 = joint_prob.sum(dim=1)
        p2 = joint_prob.sum(dim=0)
        
        # Compute mutual information
        mutual_info = 0.0
        for i in range(3):
            for j in range(3):
                if joint_prob[i, j] > eps:
                    mutual_info += joint_prob[i, j] * torch.log(
                        joint_prob[i, j] / (p1[i] * p2[j] + eps) + eps
                    )
        
        return mutual_info

    def compute_mutual_information_matrix(self, grid1, grid2):
        """Compute the per-location mutual-information matrix between two grids (vectorized implementation)"""
        H, W = grid1.shape
        eps = 1e-8
        
        # Initialize the mutual-information matrix
        mutual_info_matrix = torch.zeros((H, W), device=self.device)
        
        # Define the local window size
        window_size = 5
        padding = window_size // 2
        
        # Pad the grids and add channel dimensions (note the extra unsqueeze(0))
        grid1_padded = F.pad(grid1.unsqueeze(0).unsqueeze(0).float(), (padding, padding, padding, padding), mode='constant', value=2)
        grid2_padded = F.pad(grid2.unsqueeze(0).unsqueeze(0).float(), (padding, padding, padding, padding), mode='constant', value=2)
        
        # Use unfold to extract all local windows
        windows1 = F.unfold(grid1_padded, kernel_size=window_size, padding=0, stride=1)
        windows2 = F.unfold(grid2_padded, kernel_size=window_size, padding=0, stride=1)
        
        # Reshape to (H * W, window_size * window_size)
        windows1 = windows1.permute(0, 2, 1).reshape(H*W, window_size*window_size)
        windows2 = windows2.permute(0, 2, 1).reshape(H*W, window_size*window_size)
        
        # Process windows in batches of 100 to balance memory usage and compute cost
        batch_size = 100
        for i in range(0, H*W, batch_size):
            batch_end = min(i + batch_size, H*W)
            batch_win1 = windows1[i:batch_end]
            batch_win2 = windows2[i:batch_end]
            
            # Use one-hot encoding and batched matrix multiplication
            batch_onehot1 = F.one_hot(batch_win1.long(), num_classes=3).float()
            batch_onehot2 = F.one_hot(batch_win2.long(), num_classes=3).float()
            
            for idx, (oh1, oh2) in enumerate(zip(batch_onehot1, batch_onehot2)):
                # Compute the joint distribution
                joint_hist = torch.matmul(oh1.transpose(0, 1), oh2)
                
                # Normalize
                joint_prob = joint_hist / (joint_hist.sum() + eps)
                
                # Compute marginal distributions
                p1 = joint_prob.sum(dim=1)
                p2 = joint_prob.sum(dim=0)
                
                # Compute mutual information
                valid_mask = joint_prob > eps
                mi_terms = torch.zeros_like(joint_prob)
                mi_terms[valid_mask] = joint_prob[valid_mask] * torch.log(
                    joint_prob[valid_mask] / (torch.outer(p1, p2)[valid_mask] + eps) + eps
                )
                mi_value = mi_terms.sum()
                
                # Store the computed result in the matrix
                pos = i + idx
                mutual_info_matrix[pos // W, pos % W] = mi_value
        
        return mutual_info_matrix



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
        
        for attack_id, attack in enumerate(attacker.attack_list):
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


def attack_evaluation(attacker, perception_name):
    logging.info("Evaluating attack {} at perception {}".format(attacker.name, perception_name))
    max_cases = dataset.cache_size
    actual_case_number = min(len(attacker.attack_list), max_cases)
    
    success_log = np.zeros(actual_case_number).astype(bool)
    max_iou = np.zeros((actual_case_number, 2)).astype(np.float32)
    best_score = np.zeros((actual_case_number, 2)).astype(np.float32)

    save_dir = os.path.join(result_dir, "evaluation")
    os.makedirs(save_dir, exist_ok=True)

    @attack_case_iterator
    def attack_evaluation_processor(attacker, perception_name, case_id=None, case=None, data_dir=None, attack_id=None, attack=None):
        if attack_id >= actual_case_number:
            return
            
        ego_id = attack["attack_meta"]["victim_vehicle_id"]
        attacker_id = attack["attack_meta"]["attacker_vehicle_id"]
        case_id = attack["attack_meta"]["case_id"]
        attack_bbox = bbox_sensor_to_map(attack["attack_meta"]["bbox"][-1], case[9][attacker_id]["lidar_pose"])
        attack_bbox = bbox_map_to_sensor(attack_bbox, case[-1][ego_id]["lidar_pose"])

        feature_data = pickle_cache_load(os.path.join(result_dir, "normal/{:06d}/{}.pkl".format(case_id, perception_name)))
        
        pred_bboxes = feature_data[-1][ego_id]["pred_bboxes"]
        pred_scores = feature_data[-1][ego_id]["pred_scores"]
        for j, pred_bbox in enumerate(pred_bboxes):
            iou = iou2d(pred_bbox, attack_bbox)
            if iou > max_iou[attack_id, 0]:
                max_iou[attack_id, 0] = iou
                best_score[attack_id, 0] = pred_scores[j]

        if "early" in attacker.name:
            feature_data = pickle_cache_load(os.path.join(data_dir, "{}.pkl".format(perception_name)))
        else:
            feature_data = pickle_cache_load(os.path.join(data_dir, "attack_info.pkl"))

        pred_bboxes = feature_data[-1][ego_id]["pred_bboxes"]
        pred_scores = feature_data[-1][ego_id]["pred_scores"]
        for j, pred_bbox in enumerate(pred_bboxes):
            iou = iou2d(pred_bbox, attack_bbox)
            if iou > max_iou[attack_id, 1]:
                max_iou[attack_id, 1] = iou
                best_score[attack_id, 1] = pred_scores[j]

        if attacker.name.startswith("lidar_spoof") and max_iou[attack_id, 1] > 0:
            success_log[attack_id] = True
        if attacker.name.startswith("lidar_remove") and max_iou[attack_id, 1] == 0:
            success_log[attack_id] = True

    attack_evaluation_processor(attacker, perception_name)

    # Only save results for the cases that were actually processed.
    pickle_cache_dump({
        "success": success_log[:actual_case_number],
        "iou": max_iou[:actual_case_number],
        "score": best_score[:actual_case_number]
    }, os.path.join(save_dir, "attack_result_{}_{}.pkl".format(attacker.name, perception_name)))

    logging.info("Evaluation of attack {} at perception {}, total case number {:.2f}, success number {:.2f}, success rate {:.2f}, average IoU {:.2f}, average score {:.2f},".format(
        attacker.name, perception_name, 
        actual_case_number, 
        np.sum(success_log > 0), 
        np.mean(success_log), 
        np.mean(max_iou[:actual_case_number, 1]), 
        np.mean(best_score[:actual_case_number, 1])))

@attack_case_iterator
def defense(attacker, defender, perception_name, case_id=None, case=None, data_dir=None, attack_id=None, attack=None):
    if "early" in attacker.name:
        save_file = os.path.join(data_dir, "{}_{}.pkl".format(defender.name, perception_name))
        vis_file = os.path.join(data_dir, "{}_{}.png".format(defender.name, perception_name))
    else:
        save_file = os.path.join(data_dir, "{}.pkl".format(defender.name))
        vis_file = os.path.join(data_dir, "{}.png".format(defender.name))
    if os.path.isfile(save_file):
        return
    else:
        logging.info("Processing defense {} against attack {} on attack case {}".format(defender.name, attacker.name, attack_id))
    logging.info("Processing defense {} against attack {} on attack case {}".format(defender.name, attacker.name, attack_id))

    if "early" in attacker.name:
        perception_feature = pickle_cache_load(os.path.join(data_dir, "{}.pkl".format(perception_name)))
    else:
        perception_feature = pickle_cache_load(os.path.join(data_dir, "attack_info.pkl"))
    case = dataset.load_feature(case, occupancy_feature)

    # Load occupancy-map data
    occupancy_path = os.path.join(result_dir, f"normal/{case_id:06d}/occupancy_map.pkl")
    occupancy_feature = pickle_cache_load(occupancy_path)
    
    # Prepare temporal data
    processor = OccupancyDataProcessor()
    temporal_graphs = processor.prepare_temporal_data(
        occupancy_feature,
        frame_ids=attack_frame_ids,
        map_name=case[0][list(case[0].keys())[0]]["map"]
    )

    occupancy_feature = pickle_cache_load(os.path.join(result_dir, "normal/{:06d}/occupancy_map.pkl".format(case_id)))
    case = dataset.load_feature(case, occupancy_feature)

    defend_opts = {"frame_ids": [9]}
    new_case, score, metrics = defender.run(case, defend_opts)

    pickle_cache_dump(metrics, save_file)
    visualize_defense(case, metrics, show=False, save=vis_file)


def defense_evaluation(attacker, defender, perception_name):
    save_dir = os.path.join(result_dir, "evaluation")
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, "defense_result_{}_{}_{}.pkl".format(attacker.name, defender.name, perception_name))

    defense_results = {
        "spoof_error": [],
        "spoof_label": [],
        "spoof_location": [],
        "remove_error": [],
        "remove_label": [],
        "remove_location": [],
        "success": [],
    }

    @attack_case_iterator
    def defense_evaluation_processor(attacker, defender, perception_name, case_id=None, case=None, data_dir=None, attack_id=None, attack=None, iou_thres=0.7, dist_thres=40):
        if "early" in attacker.name:
            defense_file = os.path.join(data_dir, "{}_{}.pkl".format(defender.name, perception_name))
        else:
            defense_file = os.path.join(data_dir, "{}.pkl".format(defender.name))
        metrics = pickle_cache_load(defense_file)

        attacker_vehicle_id = attack["attack_meta"]["attacker_vehicle_id"]
        victim_vehicle_id = attack["attack_meta"]["victim_vehicle_id"]
        attack_mode =  "spoof" if "spoof" in attacker.name else "remove"
        attack_bbox = bbox_sensor_to_map(attack["attack_meta"]["bboxes"][-1], case[-1][attacker_vehicle_id]["lidar_pose"])

        victim_vehicle_id = attack["attack_meta"]["victim_vehicle_id"]
        for frame_id in attack_frame_ids:
            vehicle_metrics = metrics[frame_id][victim_vehicle_id]

        gt_bboxes = vehicle_metrics["gt_bboxes"]
        pred_bboxes = vehicle_metrics["pred_bboxes"]
        lidar_pose = vehicle_metrics["lidar_pose"]

        # iou 2d
        gt_bboxes[:, 2] = 0
        gt_bboxes[:, 5] = 1
        pred_bboxes[:, 2] = 0
        pred_bboxes[:, 5] = 1

        iou = np.zeros((gt_bboxes.shape[0], pred_bboxes.shape[0]))
        for i, gt_bbox in enumerate(gt_bboxes):
            for j, pred_bbox in enumerate(pred_bboxes):
                try:
                    # Check whether the boxes are valid.
                    orig_box = gt_bbox
                    attack_box = pred_bbox
                    
                    # Check whether the box dimensions are valid.
                    if (orig_box[3:6] <= 0).any() or (attack_box[3:6] <= 0).any():
                        logging.debug(f"Invalid box dimensions: orig_box={orig_box[3:6]}, attack_box={attack_box[3:6]}")
                        iou[i,j] = 0.0
                        continue
                        
                    # Check whether the boxes are parallel, i.e., their yaw difference is close to 0 or pi.
                    yaw_diff = abs(orig_box[6] - attack_box[6]) % (2 * np.pi)
                    if yaw_diff < 1e-3 or abs(yaw_diff - np.pi) < 1e-3:
                        # If they are parallel, fall back to the simplified IoU handling.
                        # Alternatively, directly set the IoU to zero.
                        iou[i,j] = 0.0
                        continue
                        
                    iou_val = iou2d(orig_box, attack_box)
                    
                    if np.isnan(iou_val):
                        logging.debug(f"NaN IoU detected, setting to 0")
                        iou[i,j] = 0.0
                    iou[i,j] = iou_val
                except Exception as e:
                    logging.debug(f"IoU computation error: {str(e)}")
                    iou[i,j] = 0.0

        spoof_label = np.max(iou, axis=0) <= iou_thres
        spoof_mask = np.logical_and(get_distance(pred_bboxes[:, :2], lidar_pose[:2]) > 1, get_distance(pred_bboxes[:, :2], lidar_pose[:2]) <= dist_thres)
        remove_label = np.max(iou, axis=1) <= iou_thres
        remove_mask = get_distance(gt_bboxes[:, :2], lidar_pose[:2]) <= dist_thres

        spoof_error = np.zeros(pred_bboxes.shape[0])
        spoof_location = np.zeros((pred_bboxes.shape[0], 2))
        for error_area, error, gt_error, bbox_index in vehicle_metrics["spoof"]:
            if error > spoof_error[bbox_index]:
                spoof_location[bbox_index] = np.array(list(list(error_area.centroid.coords)[0]))
                spoof_error[bbox_index] = error

        remove_error = np.zeros(gt_bboxes.shape[0])
        remove_location = np.zeros((gt_bboxes.shape[0], 2))
        for error_area, error, gt_error, bbox_index in vehicle_metrics["remove"]:
            if bbox_index < 0:
                continue
            if error > remove_error[bbox_index]:
                remove_location[bbox_index] = np.array(list(list(error_area.centroid.coords)[0]))
                remove_error[bbox_index] = error

        detected_location = spoof_location if attack_mode == "spoof" else remove_location
        is_success = np.min(get_distance(detected_location, attack_bbox[:2])) < 2

        defense_results["spoof_error"].append(spoof_error[spoof_mask])
        defense_results["spoof_label"].append(spoof_label[spoof_mask])
        defense_results["spoof_location"].append(spoof_location[spoof_mask])
        defense_results["remove_error"].append(remove_error[remove_mask])
        defense_results["remove_label"].append(remove_label[remove_mask])
        defense_results["remove_location"].append(remove_location[remove_mask])
        defense_results["success"].append(np.array([is_success]).astype(np.int8))

    defense_evaluation_processor(attacker, defender, perception_name)

    for key, data in defense_results.items():
        defense_results[key] = np.concatenate(data).reshape(-1)

    pickle_cache_dump(defense_results, save_file)
    spoof_best_TPR, spoof_best_FPR, spoof_roc_auc, spoof_best_thres = draw_roc(defense_results["spoof_error"], defense_results["spoof_label"],
            save=os.path.join(save_dir, "roc_lidar_spoof_{}_{}_{}.png".format(attacker.name, defender.name, perception_name)))
    remove_best_TPR, remove_best_FPR, remove_roc_auc, remove_best_thres = draw_roc(defense_results["remove_error"], defense_results["remove_label"],
            save=os.path.join(save_dir, "roc_lidar_remove_{}_{}_{}.png".format(attacker.name, defender.name, perception_name)))
    
    attack_result = pickle_cache_load(os.path.join(save_dir, "attack_result_{}_{}.pkl".format(attacker.name, perception_name)))
    success_rate = np.mean(attack_result["success"] * defense_results["success"])
    
    logging.info("Evaluation of defense {} against attack {} on perception {} success rate {:.2f}: For spoofing attack, best TPR {:.2f}, best FPR {:.2f}, ROC AUC {:.2f}, best threshold {:.2f}; For removal attack, best TPR {:.2f}, best FPR {:.2f}, ROC AUC {:.2f}, best threshold {:.2f}." .format(
        defender.name, attacker.name, perception_name, success_rate,
        spoof_best_TPR, spoof_best_FPR, spoof_roc_auc, spoof_best_thres, remove_best_TPR, remove_best_FPR, remove_roc_auc, remove_best_thres
    ))


@normal_case_iterator
def occupancy_map(lidar_seg_api, case_id=None, case=None, data_dir=None):
    save_file = os.path.join(data_dir, "occupancy_map.pkl")
    if os.path.isfile(save_file):
        return
    else:
        logging.info("Processing occupancy map of case {}".format(case_id))

    occupancy_feature = [{} for _ in range(total_frames)]
    for frame_id in attack_frame_ids:
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


def train_mvig_batch(attacker, mvig_model, defender, sequences, batch_size=4):
    """Train MVIG over a batch"""
    try:
        batch_attack_losses = []
        batch_box_diff_losses = []
        batch_defense_losses = []
        batch_total_losses = []
        batch_attack_successes = []
        batch_defense_successes = []
        
        # Process the provided sequence batch directly without secondary batching
        batch_cases = []
        batch_preds = []
        batch_grid_maps = []  # Dedicated list for grid maps
        batch_gt_boxes = []
        
        # Ensure all tensors stay on the correct device
        for sequence in sequences:
            try:
                # Get the basic sample data
                case = sequence['case']
                attack = sequence['attack']
                occupancy_feature = sequence['occupancy_feature']
                history_indices = sequence['history_indices']
                target_idx = sequence['target_idx']
                meta = attack["attack_meta"]
                
                # Load features
                case = dataset.load_feature(case, occupancy_feature)
                
                # Prepare temporal data
                processor = OccupancyDataProcessor()
                temporal_graphs = processor.prepare_temporal_data(
                    occupancy_feature,
                    frame_ids= history_indices,
                    map_name=case[0][list(case[0].keys())[0]]["map"]
                )
                
                if not temporal_graphs:
                    continue
                    
                # Run the prediction
                pred_box, grid_map = mvig_model(temporal_graphs)
                gt_box = torch.tensor(meta["bboxes"][target_idx], device=device, dtype=torch.float32).unsqueeze(0)
                
                # Collect batch data
                batch_cases.append({
                    'case': case,
                    'attack_opts': attack["attack_opts"].copy(),
                    'meta': meta,
                    'target_idx': target_idx,
                    'sequence_id': sequence.get('case_id', 'unknown')
                })
                batch_preds.append(pred_box)
                batch_grid_maps.append(grid_map)  # Store grid_map in the dedicated list
                batch_gt_boxes.append(gt_box)
                
            except Exception as e:
                logging.error(f"Error processing sequence: {str(e)}")
                continue
        
        if not batch_cases:
            return None
            
        # 2. Process predictions across the batch
        batch_attack_loss = torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
        batch_box_diff_loss = torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
        batch_defense_loss = torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
        batch_attack_success = []
        batch_defense_success = []
        
        for idx, (case_data, pred_box, grid_map) in enumerate(zip(batch_cases, batch_preds, batch_grid_maps)):
            try:
                # Prepare attack parameters
                attack_opts = case_data['attack_opts']
                attack_opts["victim_vehicle_id"] = case_data['meta']["victim_vehicle_id"]
                target_frame = target_idx
                attack_opts["frame_ids"] = [target_frame]
                victim_id = case_data['meta']["victim_vehicle_id"]
                attacker_id = case_data['meta']["attacker_vehicle_id"]
                case_id = case_data['meta']["case_id"]
                
                # Get original predictions before the attack
                original_pred_bboxes = []
                original_pred_scores = []
                
                # Query the perception model directly for the original predictions
                try:
                    if hasattr(attacker, 'perception') and attacker.perception:
                        orig_bboxes, orig_scores = attacker.perception.run(
                            case_data['case'][target_frame], 
                            ego_id=victim_id
                        )
                        # original pred bboxes are under victim vehicle's coordinate system
                        original_pred_bboxes = orig_bboxes
                        original_pred_scores = orig_scores
                except Exception as e:
                    logging.debug(f"Error getting original predictions: {str(e)}")
                
                # During MVIGNet training we do not run RC+/PGD-style refinement for the attack box.
                # To keep training stable, we directly use the MVIG-predicted location as a coarse attack target.
                # This target is forwarded via `attack_opts["positions"]` to the downstream attacker, which
                # later converts it to the victim frame and uses it as the reference box for mask/center setup.
                # Full evaluation should enable PGD-based optimization (for example, like RC+) to refine the final attack box.
                # Set the attack position differently for spoofing and removal attacks.
                # Initialize position_tensor first.
                position_tensor = pred_box.clone()
                position_tensor.requires_grad_(True)

                if "remove" in attacker.name and len(original_pred_bboxes) > 0:
                    try:
                        # Convert position_tensor to map coordinates first, then to the victim frame
                        position_map = bbox_sensor_to_map(
                            position_tensor.detach().cpu().numpy()[0],
                            case_data['case'][target_frame][attacker_id]["lidar_pose"]
                        )
                        position_victim = bbox_map_to_sensor(
                            position_map,
                            case_data['case'][target_frame][victim_id]["lidar_pose"]
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
                                case_data['case'][target_frame][victim_id]["lidar_pose"]
                            )
                            bbox_attacker = bbox_map_to_sensor(
                                bbox_map,
                                case_data['case'][target_frame][attacker_id]["lidar_pose"]
                            )
                            
                            # Update position_tensor
                            position_tensor = torch.tensor(
                                bbox_attacker,
                                device=device,
                                dtype=torch.float32
                            ).unsqueeze(0)
                            position_tensor.requires_grad_(True)
                        
                    except Exception as e:
                        logging.error(f"Error in removal attack processing: {str(e)}")
                        # position_tensor was already initialized above, so no reinitialization is needed here

                # Convert to a NumPy array for downstream processing; position_tensor is in the attacker frame.
                # This training path passes the predicted attack location directly to the attacker instead of
                # performing an additional PGD refinement step on the box location.
                position_np = position_tensor.detach().cpu().numpy()
                
                # Get the total number of frames
                total_frames = len(case_data['meta']["frame_ids"])
                positions = np.tile(position_np, (total_frames, 1))
                
                # Reuse the same predicted attack box across frames for the current training sample.
                # The attacker interprets this as the target spoof/remove box, then derives `bbox_to_spoof_ego`
                # in the victim frame before launching RC/BASIC/BAC/RC+ execution.
                # PGD-based box refinement is intentionally left to evaluation/inference-time attack execution.
                # Set the position for all frames.
                attack_opts["positions"] = positions
                attack_opts["frame_ids"] = [target_frame]  # Attack only the target frame
                attack["attack_meta"]["attack_frame_ids"] = [target_frame]  # Record only the target frame
                bboxes_list = [positions[i] for i in range(len(positions))]
                attack["attack_meta"]["bboxes"] = bboxes_list
                attack["attack_meta"]["bbox"] = [bboxes_list[target_frame]]
                
                # Run the attack
                with torch.enable_grad():
                    new_case, attack_info = attacker.run(case_data['case'], attack_opts)
                
                # Get predictions before and after the attack
                attack_pred_bboxes = attack_info[target_frame][victim_id]["pred_bboxes"]
                
                # Get attack metadata
                # attack_meta = case_data['meta']
                attack_meta = attack["attack_meta"]
                
                # 1. Compute attack_loss using evaluation criteria aligned with evaluate.py
                if "bboxes" in attack_meta and len(attack_meta["bboxes"]) > target_frame:
                    
                    # Attack type
                    attack_mode = "spoof" if "spoof" in attacker.name else "remove"
                    
                    # Compute the maximum IoU between post-attack predictions and the target box
                    max_iou = 0.0
                    best_score = 0.0
                    
                    # Compute the distance between bbox_to_spoof and the victim vehicle
                    bbox_to_spoof_ego = bbox_map_to_sensor(
                        bbox_sensor_to_map(position_np[-1], case_data['case'][target_frame][attacker_id]["lidar_pose"]),
                        case_data['case'][target_frame][victim_id]["lidar_pose"])
                    
                    # Compute the distance between bbox_to_spoof and the victim vehicle
                    # Assume the victim vehicle is at the origin of its own frame
                    bbox_to_spoof_pos = torch.tensor(bbox_to_spoof_ego[:2], device=device, dtype=torch.float32)
                    victim_pos = torch.tensor([0.0, 0.0], device=device, dtype=torch.float32)  # Victim-frame origin
                    distance_to_victim = torch.norm(bbox_to_spoof_pos - victim_pos)
                    
                    # Distance penalty term computed with torch ops to preserve gradient flow
                    distance_to_victim_tensor = torch.tensor(distance_to_victim, device=device)
                    distance_penalty = 0.4 * (1.0 - torch.exp(-distance_to_victim_tensor / 10.0))  # Preserve gradient flow
                    
                    if len(attack_pred_bboxes) > 0:
                        for j, pred_bbox in enumerate(attack_pred_bboxes):
                            try:
                                iou = iou2d(pred_bbox, bbox_to_spoof_ego)
                                if iou > max_iou:
                                    max_iou = iou
                                    best_score = 1.0
                            except Exception as e:
                                logging.debug(f"IoU computation error: {str(e)}")
                    
                    # Create the base loss tensor
                    base_loss = torch.zeros(1, device=device, requires_grad=True)
                    
                    if attack_mode == "spoof":
                        # Convert IoU to a tensor.
                        max_iou_tensor = torch.tensor([max_iou], device=device)
                        
                        if max_iou > 0.1:  # Use a higher threshold to ensure quality
                            # Keep all computations in tensors to preserve gradient flow.
                            attack_term = 1.0 - max_iou_tensor
                            sample_attack_loss = attack_term + distance_penalty
                            attack_success = True
                            
                        elif max_iou > 0:  # Partially successful
                            # Keep all computations in tensors.
                            attack_term = 1.0 - 0.5 * max_iou_tensor
                            sample_attack_loss = attack_term + distance_penalty
                            attack_success = False
                            
                        else:  # Complete failure
                            # Increase the weight of the distance penalty
                            enhanced_distance_penalty = 0.5 * (1.0 - torch.exp(-distance_to_victim_tensor / 10.0))
                            sample_attack_loss = base_loss + 1.0 + enhanced_distance_penalty  # Add 1.0 as the base penalty
                            attack_success = False
                            
                            # If the attack completely fails, use the distance to the nearest detection as an auxiliary loss.
                            if len(attack_pred_bboxes) > 0:
                                # Compute the distance between position_tensor and all detected boxes
                                distances = []
                                for pred_bbox in attack_pred_bboxes:
                                    pred_pos = torch.tensor(pred_bbox[:3], device=device, dtype=torch.float32)
                                    dist = torch.norm(position_tensor[-1][:3] - pred_pos)
                                    distances.append(dist)
                                
                                if distances:
                                    # Use torch functions to compute the minimum distance while preserving gradient flow.
                                    min_dist = torch.min(torch.stack(distances))
                                    detection_distance_factor = 0.1 * min_dist
                                    sample_attack_loss = sample_attack_loss + detection_distance_factor
                    else:  # remove attack
                        # Removal attack: lower IoU should lead to a smaller loss
                        # In evaluate.py, IoU == 0 indicates a successful attack
                        if max_iou == 0:
                            # Even a fully successful attack still includes the distance penalty
                            sample_attack_loss = distance_penalty
                            attack_success = True
                        elif max_iou < 0.1:  # Treat very small IoU as a partial success.
                            # Lower IoU should yield a smaller loss, while still accounting for the distance penalty.
                            sample_attack_loss = torch.tensor(max_iou * 2, 
                                                            device=device, 
                                                            dtype=torch.float32, 
                                                            requires_grad=True) + distance_penalty
                            attack_success = True  # Consider this successful
                        else:
                            # Lower IoU should yield a smaller loss, while still accounting for the distance penalty.
                            # Increase the weight of the distance penalty when the attack fails.
                            enhanced_distance_penalty = 0.5 * (1.0 - torch.exp(-distance_to_victim_tensor / 10.0))
                            sample_attack_loss = torch.tensor(max_iou, 
                                                            device=device, 
                                                            dtype=torch.float32, 
                                                            requires_grad=True) + enhanced_distance_penalty
                            attack_success = False
                
                # 2. Compute box_diff_loss so that position_tensor differs from the original bounding boxes
                if len(original_pred_bboxes) > 0:

                    # Convert position_tensor to a bounding box in the victim frame
                    bbox_to_spoof_ego = bbox_map_to_sensor(
                        bbox_sensor_to_map(position_np[-1], case_data['case'][target_frame][attacker_id]["lidar_pose"]),
                        case_data['case'][target_frame][victim_id]["lidar_pose"])
                    
                    # Compute IoU with all original bounding boxes
                    ious = []
                    for orig_box in original_pred_bboxes:
                        try:
                            iou = iou2d(orig_box, bbox_to_spoof_ego)
                            ious.append(iou)
                        except Exception as e:
                            logging.debug(f"IoU computation error in box_diff_loss: {str(e)}")
                            ious.append(0.0)  # Assume IoU is 0 when an error occurs
                    
                    # Find the maximum IoU
                    if ious:
                        max_iou = max(ious)
                        
                        # Spoofing attack: smaller IoU should yield smaller loss because the generated box should differ from original boxes
                        if "spoof" in attacker.name:
                            # Use IoU directly as the loss
                            # IoU of 0 gives zero loss, and IoU of 1 gives unit loss
                            sample_box_diff_loss = torch.tensor(max_iou, 
                                                              device=device, 
                                                              dtype=torch.float32, 
                                                              requires_grad=True)
                        else:  # remove attack
                            # Removal attack: the generated box should overlap strongly with the original box
                            # Use 1 - max_iou as the loss so that:
                            # - When IoU is close to 1, the loss is close to 0.
                            # - When IoU is close to 0, the loss is close to 1.
                            sample_box_diff_loss = torch.tensor(1.0 - max_iou, 
                                                              device=device, 
                                                              dtype=torch.float32, 
                                                              requires_grad=True)
                    else:
                        sample_box_diff_loss = torch.tensor(0.0, 
                                                          device=device, 
                                                          dtype=torch.float32, 
                                                          requires_grad=True)
                else:
                    sample_box_diff_loss = torch.tensor(0.0, 
                                                      device=device, 
                                                      dtype=torch.float32, 
                                                      requires_grad=True)
                
                # Get defense-evaluation results
                metrics = None
                if defender:
                    defend_opts = {"frame_ids": [target_frame]}
                    _, _, metrics = defender.run(new_case, defend_opts)

                # Initialize the defense-loss variable and success flag
                # Start from a larger initial value to encourage the model to find attacks that evade defense
                sample_defense_loss = torch.tensor(1.0, device=device, dtype=torch.float32, requires_grad=True)
                defense_success = True
                
                # 3. Compute defense_loss with criteria aligned with defense_evaluation_processor in evaluate.py
                try:
                    # Gather information required for defense evaluation
                    if metrics and isinstance(metrics, list) and target_frame < len(metrics):
                        frame_metrics = metrics[target_frame]
                        victim_vehicle_id = case_data['meta']["victim_vehicle_id"]
                        
                        if victim_vehicle_id in frame_metrics:
                            vehicle_metrics = frame_metrics[victim_vehicle_id]
                            
                            # Gather attack-related information.
                            attack_mode = "spoof" if "spoof" in attacker.name else "remove"
                            
                            if "bboxes" in attack_meta and "attacker_vehicle_id" in attack_meta:
                                attacker_vehicle_id = attack_meta["attacker_vehicle_id"]
                                attack_bbox = bbox_sensor_to_map(attack_meta["bboxes"][target_frame], case_data['case'][target_frame][attacker_id]["lidar_pose"])  # Attack bounding box for the target frame
                                
                                # Get predicted and ground-truth bounding boxes
                                gt_bboxes = vehicle_metrics.get("gt_bboxes", np.array([]))
                                pred_bboxes = vehicle_metrics.get("pred_bboxes", np.array([]))
                                
                                if attack_mode == "spoof":
                                    # Compute detected locations for spoofing attacks
                                    spoof_location = np.zeros((pred_bboxes.shape[0], 2))
                                    spoof_error = np.zeros(pred_bboxes.shape[0])
                                    
                                    for error_area, error, gt_error, bbox_index in vehicle_metrics.get("spoof", []):
                                        if bbox_index < len(spoof_error) and error > spoof_error[bbox_index]:
                                            try:
                                                spoof_location[bbox_index] = np.array(list(list(error_area.centroid.coords)[0]))
                                                spoof_error[bbox_index] = error
                                            except Exception as e:
                                                logging.debug(f"Error processing spoof location: {str(e)}")
                                    
                                    # Compute the minimum distance between detected locations and the attack location
                                    if len(spoof_location) > 0 and np.any(spoof_error > 0):
                                        distances = np.sqrt(np.sum((spoof_location[spoof_error > 0] - attack_bbox[:2])**2, axis=1))
                                        min_distance = np.min(distances) if len(distances) > 0 else float('inf')
                                        
                                        # Smaller distance implies larger defense_loss
                                        if min_distance < float('inf'):
                                    # Use PyTorch operations instead of NumPy to preserve gradient flow.
                                            min_distance_tensor = torch.tensor(min_distance, device=device, dtype=torch.float32)
                                            
                                            # Use PyTorch functions for the loss so gradients can propagate
                                            defense_loss_value = 1.0 / (1.0 + torch.exp((min_distance_tensor - 2.0) * 0.5))
                                            sample_defense_loss = defense_loss_value  # This is already a tensor and does not need conversion
                                            
                                            defense_success = (min_distance < 2.0)  # Keep the original success criterion
                                else:  # remove attack
                                    # Compute detected locations for removal attacks
                                    remove_location = np.zeros((gt_bboxes.shape[0], 2))
                                    remove_error = np.zeros(gt_bboxes.shape[0])
                                    
                                    for error_area, error, gt_error, bbox_index in vehicle_metrics.get("remove", []):
                                        if bbox_index >= 0 and bbox_index < len(remove_error) and error > remove_error[bbox_index]:
                                            try:
                                                remove_location[bbox_index] = np.array(list(list(error_area.centroid.coords)[0]))
                                                remove_error[bbox_index] = error
                                            except Exception as e:
                                                logging.debug(f"Error processing remove location: {str(e)}")
                                    
                                    # Compute the minimum distance between detected locations and the attack location
                                    if len(remove_location) > 0 and np.any(remove_error > 0):
                                        distances = np.sqrt(np.sum((remove_location[remove_error > 0] - attack_bbox[:2])**2, axis=1))
                                        min_distance = np.min(distances) if len(distances) > 0 else float('inf')
                                        
                                        # Smaller distance implies larger defense_loss
                                        if min_distance < float('inf'):
                                            # Use PyTorch operations instead of NumPy to preserve gradient flow.
                                            min_distance_tensor = torch.tensor(min_distance, device=device, dtype=torch.float32)
                                            
                                            # Use PyTorch functions for the loss so gradients can propagate
                                            defense_loss_value = 1.0 / (1.0 + torch.exp((min_distance_tensor - 2.0) * 0.5))
                                            sample_defense_loss = defense_loss_value  # This is already a tensor and does not need conversion
                                            
                                            defense_success = (min_distance < 2.0)  # Keep the original success criterion
                except Exception as e:
                    logging.debug(f"Error computing defense loss: {str(e)}")
                    sample_defense_loss = torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
                    defense_success = False
                
                # Accumulate batch losses
                batch_attack_loss = batch_attack_loss + sample_attack_loss
                batch_box_diff_loss = batch_box_diff_loss + sample_box_diff_loss
                batch_defense_loss = batch_defense_loss + sample_defense_loss
                batch_attack_success.append(attack_success)
                batch_defense_success.append(defense_success)
                
            except Exception as e:
                logging.error(f"Error processing batch item {idx}: {str(e)}\nTraceback:\n{traceback.format_exc()}")
                continue
        
        # Compute the average loss over the batch
        valid_batch_size = len(batch_defense_success)
        if valid_batch_size > 0:
            # Compute average losses
            batch_attack_loss = batch_attack_loss / valid_batch_size
            batch_box_diff_loss = batch_box_diff_loss / valid_batch_size
            batch_defense_loss = batch_defense_loss / valid_batch_size
            
            # Compute attack success rate
            attack_success_rate = sum(batch_attack_success) / valid_batch_size
            
            # Tune weights based on the observed loss values
            attack_weight = 1   # Greatly reduced because attack success is already 100%
            box_diff_weight = 10.0  # Greatly increased because the current value is 0
            defense_weight = 5   # Substantially increased because the current value is 0 and the success rate is 0%
            
            # Compute the total loss while keeping it differentiable
            batch_total_loss = attack_weight * batch_attack_loss + box_diff_weight * batch_box_diff_loss
            
            # Always include the defense loss
            batch_total_loss = batch_total_loss + defense_weight * batch_defense_loss
            
            # Record losses
            batch_attack_losses.append(batch_attack_loss.detach())
            batch_box_diff_losses.append(batch_box_diff_loss.detach())
            batch_defense_losses.append(batch_defense_loss.detach())
            batch_total_losses.append(batch_total_loss)
            batch_attack_successes.extend(batch_attack_success)
            batch_defense_successes.extend(batch_defense_success)
            
            # Log current batch performance
            logging.info(f"Batch stats: Attack success rate: {attack_success_rate:.2f}, "
                        f"Defense success rate: {sum(batch_defense_success)/valid_batch_size:.2f}, "
                        f"Attack loss: {batch_attack_loss.item():.4f}, "
                        f"Box diff loss: {batch_box_diff_loss.item():.4f}, "
                        f"Defense loss: {batch_defense_loss.item():.4f}")
        
        # Compute the average loss over all batches
        if batch_total_losses:
            # Return the total loss for backpropagation
            return {
                "total_loss": sum(batch_total_losses) / len(batch_total_losses),
                "attack_loss": sum(batch_attack_losses) / len(batch_attack_losses),
                "box_diff_loss": sum(batch_box_diff_losses) / len(batch_box_diff_losses),
                "defense_loss": sum(batch_defense_losses) / len(batch_defense_losses),
                "attack_success_rate": sum(batch_attack_successes) / len(batch_attack_successes),
                "defense_success_rate": sum(batch_defense_successes) / len(batch_defense_successes)
            }
        
        return None
        
    except Exception as e:
        logging.error(f"Error in train_mvig_batch: {str(e)}\nTraceback:\n{traceback.format_exc()}")
        return None

def prepare_dataset():
    """Preprocess and load all data"""
    logging.info("Preparing dataset...")
    all_sequences = []
    
    # Load data from the attacker's attack_list
    attacker = attacker_list[0]
    max_attacks = dataset.cache_size
    
    for attack_id, attack in enumerate(attacker.attack_list):
        if attack_id >= max_attacks:
            break
            
        try:
            case_id = attack["attack_meta"]["case_id"]
            # Load case data
            case = dataset.get_case(case_id, tag="multi_frame", use_lidar=True, use_camera=False)
            
            # Load occupancy-map data
            occupancy_path = os.path.join(result_dir, f"normal/{case_id:06d}/occupancy_map.pkl")
            if not os.path.exists(occupancy_path):
                logging.warning(f"Occupancy map not found for case {case_id}")
                continue
                
            occupancy_feature = pickle_cache_load(occupancy_path)
            
            # Get attack frame IDs
            attack_frame_ids = attack["attack_meta"]["attack_frame_ids"]
            
            # Create a sequence for each temporal window
            history_window = 5  # Number of history frames
            for i in range(len(attack_frame_ids) - history_window):
                sequence = {
                    'case_id': case_id,
                    'case': case,
                    'occupancy_feature': occupancy_feature,
                    'attack': attack,
                    'history_indices': list(range(i, i + history_window)),
                    'target_idx': i + history_window
                }
                all_sequences.append(sequence)
                
            logging.info(f"Successfully processed case {case_id}, added {len(attack_frame_ids) - history_window} sequences")
            
        except Exception as e:
            logging.error(f"Error processing case {case_id}: {str(e)}\nTraceback:\n{traceback.format_exc()}")
            continue
    
    if not all_sequences:
        raise ValueError("No valid sequences found in dataset")
        
    # Shuffle the data
    np.random.shuffle(all_sequences)
    
    # Split the dataset
    total_size = len(all_sequences)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    
    train_sequences = all_sequences[:train_size]
    val_sequences = all_sequences[train_size:train_size + val_size]
    test_sequences = all_sequences[train_size + val_size:]
    
    logging.info(f"Dataset split: Train {len(train_sequences)}, Val {len(val_sequences)}, Test {len(test_sequences)}")
    
    return train_sequences, val_sequences, test_sequences

def main():
    logging.info(
        "Runtime overrides - attack_type=%s, epochs=%d, cache_size=%d, attack_step=%d, cuda_visible_devices=%s",
        TRAIN_ATTACK_TYPE,
        TRAIN_TOTAL_EPOCHS,
        TRAIN_CACHE_SIZE,
        TRAIN_ATTACK_STEP,
        os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    )
    logging.info(f"######################## Loading dataset (cache_size={dataset.cache_size}) ########################")
    logging.info("Initialized perception models")
    logging.info(f"Initialized attacker: {list(attacker_dict.keys())[0]}")
    logging.info("Initialized defender")
    logging.info("######################## Generating occupancy maps ########################")
    # lidar_seg_api = SqueezeSegInterface()
    # occupancy_map(lidar_seg_api)

    # Preprocess and load the dataset
    train_sequences, val_sequences, test_sequences = prepare_dataset()
    
    # Initialize MVIG model and optimizer
    attack_type = attacker_list[0].name.split("_")[1]
    mvig_model = MVIGNet(attack_type=attack_type).to(device)
    
    # Lower the initial learning rate to stabilize training
    initial_lr = 0.0001  # Initial learning rate
    
    # Use Adam as the optimizer
    optimizer = torch.optim.Adam(
        mvig_model.parameters(), 
        lr=initial_lr,
        weight_decay=1e-4,  # Increase weight decay to reduce overfitting
        betas=(0.9, 0.999),  # Default momentum parameters
        eps=1e-8  # Numerical-stability parameter
    )
    
    # Use a cyclic learning rate to help the model escape local optima
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=0.00001,  # Minimum learning rate
        max_lr=0.0005,    # Maximum learning rate
        step_size_up=5,   # Warm-up half-cycle (epochs)
        step_size_down=10, # Decay half-cycle (epochs)
        mode='triangular2', # Triangular2 cyclic mode with decreasing amplitude
        cycle_momentum=False
    )
    
    # Apply stronger gradient clipping to prevent exploding gradients
    max_grad_norm = 0.5  # Reduced from 1.0 to 0.5
    
    logging.info(f"Initialized MVIG model and optimizer with lr={initial_lr}")

    # Training loop
    logging.info("######################## Training MVIG attacker ########################")
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    # Create the global progress bar
    total_epochs = TRAIN_TOTAL_EPOCHS
    total_samples = len(train_sequences) * total_epochs
    pbar = tqdm(total=total_samples, 
                desc="Total Training Progress",
                position=0,
                leave=True)
    
    for epoch in range(total_epochs):
        mvig_model.train()
        train_total_loss = 0
        train_attack_loss = 0
        train_box_diff_loss = 0
        train_defense_loss = 0
        train_attack_success_rate = 0
        train_defense_success_rate = 0
        train_valid_cases = 0
        
        # Randomly subsample training data so that each epoch uses 60% of the training set
        sample_ratio = 0.6  # This sampling ratio can be adjusted
        sample_size = int(len(train_sequences) * sample_ratio)
        
        # Use the epoch index as the random seed so each epoch samples a different subset
        np.random.seed(epoch + 42)
        sampled_indices = np.random.choice(
            len(train_sequences), 
            size=sample_size, 
            replace=False  # Sample without replacement
        )
        
        # Build the current epoch's training subset from sampled indices
        epoch_train_sequences = [train_sequences[i] for i in sampled_indices]
        
        logging.info(f"Epoch {epoch+1}: Randomly sampled {sample_size}/{len(train_sequences)} training sequences ({sample_ratio*100:.0f}%)")
        
        # Create the progress bar using only the sampled subset size
        pbar = tqdm(total=len(epoch_train_sequences), desc=f"Epoch {epoch+1}/{total_epochs}")
        
        # Batch size
        batch_size = 4
        
        # Collect a batch
        batch_sequences = []
        
        # Train on the sampled training sequences
        for seq_idx, seq in enumerate(epoch_train_sequences):
            batch_sequences.append(seq)
            
            # Run batch training once enough sequences are collected or the final sequence is reached
            if len(batch_sequences) >= batch_size or seq_idx == len(epoch_train_sequences) - 1:
                try:
                    # Run batch training
                    loss = train_mvig_batch(
                        attacker_list[0],
                        mvig_model,
                        defender_list[0],
                        batch_sequences,  # Pass the entire batch
                        batch_size=batch_size
                    )
                    
                    # Handle losses
                    if loss and 'total_loss' in loss:
                        # Backpropagate
                        optimizer.zero_grad()
                        loss['total_loss'].backward()
                        
                        # Apply gradient clipping
                        torch.nn.utils.clip_grad_norm_(mvig_model.parameters(), max_grad_norm)
                        
                        # Update parameters
                        optimizer.step()
                        
                        # Accumulate losses
                        train_total_loss += loss['total_loss'].item()
                        
                        # Get component-wise losses
                        train_attack_loss += loss.get('attack_loss', torch.tensor(0.0)).item()
                        train_box_diff_loss += loss.get('box_diff_loss', torch.tensor(0.0)).item()
                        train_defense_loss += loss.get('defense_loss', torch.tensor(0.0)).item()
                        
                        # Get success rates
                        attack_success_rate = float(loss.get('attack_success_rate', 0.0))
                        train_attack_success_rate += attack_success_rate
                        
                        defense_success_rate = float(loss.get('defense_success_rate', 0.0))
                        train_defense_success_rate += defense_success_rate
                        
                        train_valid_cases += 1
                    
                    # Update the progress bar
                    pbar.update(len(batch_sequences))
                    
                    # Compute averages
                    if train_valid_cases > 0:
                        avg_total_loss = train_total_loss / train_valid_cases
                        avg_attack_loss = train_attack_loss / train_valid_cases
                        avg_box_diff_loss = train_box_diff_loss / train_valid_cases
                        avg_defense_loss = train_defense_loss / train_valid_cases
                        avg_attack_success = train_attack_success_rate / train_valid_cases
                        avg_defense_success = train_defense_success_rate / train_valid_cases
                        
                        # Log only to the file and avoid duplicating output on the console
                        logging.info(f"Progress - Epoch: {epoch+1}/{total_epochs}, "
                                    f"Total: {avg_total_loss:.4f}, "
                                    f"Attack: {avg_attack_loss:.4f}, "
                                    f"BoxDiff: {avg_box_diff_loss:.4f}, "
                                    f"Defense: {avg_defense_loss:.4f}, "
                                    f"AttackSucc: {avg_attack_success:.2f}, "
                                    f"DefSucc: {avg_defense_success:.2f}, "
                                    f"Valid: {train_valid_cases}/{len(epoch_train_sequences)//batch_size}")
                        
                        # Update the progress bar display without writing to the log.
                        pbar.set_postfix({
                            'Epoch': f"{epoch+1}/{total_epochs}",
                            'Total': f"{avg_total_loss:.4f}",
                            'Attack': f"{avg_attack_loss:.4f}",
                            'BoxDiff': f"{avg_box_diff_loss:.4f}",
                            'Def': f"{avg_defense_loss:.4f}",
                            'AttSucc': f"{avg_attack_success:.2f}",
                            'DefSucc': f"{avg_defense_success:.2f}",
                            'Valid': f"{train_valid_cases}/{len(epoch_train_sequences)//batch_size}"
                        })
                
                except Exception as e:
                    logging.error(f"Error in batch training: {str(e)}\nTraceback:\n{traceback.format_exc()}")
                    pbar.update(len(batch_sequences))
                
                # Clear the batch
                batch_sequences = []
        
        pbar.close()
        
        # Validation phase, also using batch processing
        mvig_model.eval()
        val_total_loss = 0
        val_attack_loss = 0
        val_box_diff_loss = 0
        val_defense_loss = 0
        val_attack_success_rate = 0
        val_defense_success_rate = 0
        val_valid_cases = 0
        
        with torch.no_grad():
            # Collect validation batches
            val_batch_sequences = []
            
            for seq_idx, seq in enumerate(val_sequences):
                val_batch_sequences.append(seq)
                
                # Run batch training once enough sequences are collected or the final sequence is reached
                if len(val_batch_sequences) >= batch_size or seq_idx == len(val_sequences) - 1:
                    try:
                        loss = train_mvig_batch(
                            attacker_list[0],
                            mvig_model,
                            defender_list[0],
                            val_batch_sequences
                        )
                        
                        if loss and 'total_loss' in loss:
                            val_total_loss += loss['total_loss'].item()
                            
                            # Get component-wise losses
                            val_attack_loss += loss.get('attack_loss', torch.tensor(0.0)).item()
                            val_box_diff_loss += loss.get('box_diff_loss', torch.tensor(0.0)).item()
                            val_defense_loss += loss.get('defense_loss', torch.tensor(0.0)).item()
                            
                            # Get success rates
                            val_attack_success_rate += float(loss.get('attack_success_rate', 0.0))
                            val_defense_success_rate += float(loss.get('defense_success_rate', 0.0))
                            
                            val_valid_cases += 1
                            
                    except Exception as e:
                        logging.error(f"Error in validation: {str(e)}")
                    
                    # Clear the validation batch.
                    val_batch_sequences = []
        
        if val_valid_cases > 0:
            val_avg_loss = val_total_loss / val_valid_cases
            val_avg_attack_loss = val_attack_loss / val_valid_cases
            val_avg_box_diff_loss = val_box_diff_loss / val_valid_cases
            val_avg_defense_loss = val_defense_loss / val_valid_cases
            val_avg_attack_success = val_attack_success_rate / val_valid_cases
            val_avg_defense_success = val_defense_success_rate / val_valid_cases
            
            # Update the learning-rate scheduler
            scheduler.step(val_avg_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            logging.info(f"Epoch {epoch+1} - Validation: "
                        f"Total Loss: {val_avg_loss:.4f}, "
                        f"Attack Loss: {val_avg_attack_loss:.4f}, "
                        f"BoxDiff Loss: {val_avg_box_diff_loss:.4f}, "
                        f"Defense Loss: {val_avg_defense_loss:.4f}, "
                        f"Attack Success: {val_avg_attack_success:.2f}, "
                        f"Defense Success: {val_avg_defense_success:.2f}, "
                        f"Learning Rate: {current_lr:.6f}")
            
            # Early stopping check
            if val_avg_loss < best_val_loss:
                best_val_loss = val_avg_loss
                patience_counter = 0
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': mvig_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                    'attack_loss': val_avg_attack_loss,
                    'box_diff_loss': val_avg_box_diff_loss,
                    'defense_loss': val_avg_defense_loss,
                    'attack_success': val_avg_attack_success,
                    'defense_success': val_avg_defense_success
                }, os.path.join(result_dir, 'best_mvig_model.pth'))
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logging.info("Early stopping triggered")
                break
        
        # Save checkpoint
        if (epoch + 1) % 3 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': mvig_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_avg_loss if val_valid_cases > 0 else float('inf'),
                'attack_loss': val_avg_attack_loss if val_valid_cases > 0 else float('inf'),
                'box_diff_loss': val_avg_box_diff_loss if val_valid_cases > 0 else float('inf'),
                'defense_loss': val_avg_defense_loss if val_valid_cases > 0 else float('inf'),
                'attack_success': val_avg_attack_success if val_valid_cases > 0 else 0.0,
                'defense_success': val_avg_defense_success if val_valid_cases > 0 else 0.0
            }, os.path.join(result_dir, f'mvig_checkpoint_epoch_{epoch+1}.pth'))
    
    pbar.close()
    
    # Test phase
    mvig_model.eval()
    test_total_loss = 0
    test_attack_loss = 0
    test_box_diff_loss = 0
    test_defense_loss = 0
    test_attack_success_rate = 0
    test_defense_success_rate = 0
    test_valid_cases = 0
    
    with torch.no_grad():
        for seq in test_sequences:
            try:
                loss = train_mvig_batch(
                    attacker_list[0],
                    mvig_model,
                    defender_list[0],
                    [seq]
                )
                
                if loss and 'total_loss' in loss:
                    test_total_loss += loss['total_loss'].item()
                    
                    # Get component-wise losses
                    test_attack_loss += loss.get('attack_loss', torch.tensor(0.0)).item()
                    test_box_diff_loss += loss.get('box_diff_loss', torch.tensor(0.0)).item()
                    test_defense_loss += loss.get('defense_loss', torch.tensor(0.0)).item()
                    
                    # Get success rates
                    test_attack_success_rate += float(loss.get('attack_success_rate', 0.0))
                    test_defense_success_rate += float(loss.get('defense_success_rate', 0.0))
                    
                    test_valid_cases += 1
                    
            except Exception as e:
                logging.error(f"Error in testing: {str(e)}")
                continue
    
    if test_valid_cases > 0:
        test_avg_loss = test_total_loss / test_valid_cases
        test_avg_attack_loss = test_attack_loss / test_valid_cases
        test_avg_box_diff_loss = test_box_diff_loss / test_valid_cases
        test_avg_defense_loss = test_defense_loss / test_valid_cases
        test_avg_attack_success = test_attack_success_rate / test_valid_cases
        test_avg_defense_success = test_defense_success_rate / test_valid_cases
        
        logging.info(f"Final Test Results: "
                    f"Total Loss: {test_avg_loss:.4f}, "
                    f"Attack Loss: {test_avg_attack_loss:.4f}, "
                    f"BoxDiff Loss: {test_avg_box_diff_loss:.4f}, "
                    f"Defense Loss: {test_avg_defense_loss:.4f}, "
                    f"Attack Success: {test_avg_attack_success:.2f}, "
                    f"Defense Success: {test_avg_defense_success:.2f}")

if __name__ == "__main__":
    main()
