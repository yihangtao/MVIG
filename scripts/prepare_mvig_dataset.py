import os
import sys
import logging
import numpy as np
import pickle
from collections import OrderedDict
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.ops import unary_union
from matplotlib.path import Path
import torch

# Add root directory to system path
root = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../")
sys.path.append(root)

from mvp.config import data_root
from mvp.data.opv2v_dataset import OPV2VDataset
from mvp.data.util import pcd_sensor_to_map, bbox_sensor_to_map
from mvp.tools.squeezeseg.interface import SqueezeSegInterface
from mvp.defense.detection_util import filter_segmentation
from mvp.tools.lidar_seg import lidar_segmentation
from mvp.tools.ground_detection import get_ground_plane
from mvp.tools.polygon_space import get_occupied_space, get_free_space, bbox_to_polygon


class OccupancyMapGenerator:
    def __init__(self, grid_size=(200, 200), cell_size=0.5, range_limit=50):
        """Initialize occupancy map generator
        Args:
            grid_size: Tuple of (height, width) for the grid map
            cell_size: Size of each grid cell in meters
            range_limit: Maximum range to consider in meters
        """
        self.grid_params = {
            "size": grid_size,
            "cell_size": cell_size,
            "range_limit": range_limit
        }
        self.lane_areas_map = self._load_map()
        # Load attack-scene metadata
        self.attack_meta = self._load_attack_meta()
        
    def _load_map(self, map_names=None):
        """Load lane area information for all maps"""
        lane_areas_map = {}
        if map_names is None:
            map_names = ["Town01", "Town02", "Town03", "Town04", "Town05", "Town06", "Town07", "Town10HD"]
        
        for map_name in map_names:
            with open(os.path.join(data_root, f"carla/{map_name}_lane_areas.pkl"), "rb") as f:
                lane_areas_map[map_name] = pickle.load(f)
        return lane_areas_map

    def _load_attack_meta(self):
        """Load metadata for attack scenarios."""
        attack_meta_path = os.path.join(data_root, "OPV2V/attack/perception.pkl")
        if os.path.exists(attack_meta_path):
            with open(attack_meta_path, 'rb') as f:
                return pickle.load(f)
        return {}

    def polygon_to_grid(self, polygon, lidar_pose):
        """Convert a polygon to grid coordinates relative to vehicle pose
        Args:
            polygon: Shapely Polygon in map coordinates
            lidar_pose: [x, y, z, roll, yaw, pitch] in map coordinates
        Returns:
            mask: Boolean array of shape grid_size
        """
        H, W = self.grid_params["size"]
        cell_size = self.grid_params["cell_size"]
        
        # Create grid coordinates centered at vehicle position
        x = np.arange(-W/2, W/2) * cell_size
        y = np.arange(-H/2, H/2) * cell_size
        X, Y = np.meshgrid(x, y)
        
        # Transform grid points from vehicle coordinates to map coordinates
        cos_yaw = np.cos(np.radians(lidar_pose[4]))
        sin_yaw = np.sin(np.radians(lidar_pose[4]))
        
        # Rotate
        X_rotated = X * cos_yaw - Y * sin_yaw
        Y_rotated = X * sin_yaw + Y * cos_yaw
        
        # Translate
        X_map = X_rotated + lidar_pose[0]
        Y_map = Y_rotated + lidar_pose[1]
        
        # Stack coordinates for testing
        points_map = np.column_stack((X_map.ravel(), Y_map.ravel()))
        
        # Create range mask (circular observation area in vehicle coordinates)
        range_mask = (X**2 + Y**2) <= self.grid_params["range_limit"]**2
        
        # Convert polygon to path for efficient point containment test
        if isinstance(polygon, Polygon):
            path = Path(np.array(polygon.exterior.coords))
            mask = path.contains_points(points_map).reshape(H, W)
            return mask & range_mask
        
        return np.zeros((H, W), dtype=bool)

    def process_frame(self, frame_data, lidar_seg_api, case_id):
        """Process single frame data to generate occupancy maps for each vehicle"""
        vehicle_grids = {}
        
        # Get attack metadata for the current scenario
        attack_info = self.attack_meta.get(case_id, {})
        attacker_id = attack_info.get('attacker_vehicle_id', None)
        victim_id = attack_info.get('victim_vehicle_id', None)
        
        # Get map information
        map_name = frame_data[list(frame_data.keys())[0]]["map"]
        lane_areas = self.lane_areas_map.get(map_name, None)
        
        for vehicle_id, vehicle_data in frame_data.items():
            # Get vehicle pose and sensor data
            lidar = vehicle_data["lidar"]
            lidar_pose = vehicle_data["lidar_pose"]
            pcd = pcd_sensor_to_map(lidar, lidar_pose)
            
            # Initialize grid map (0:free, 1:occupied, 2:unknown)
            grid_map = np.full(self.grid_params["size"], 2, dtype=np.uint8)
            
            # Process ground and objects
            ground_indices, in_lane_mask, point_height = get_ground_plane(
                pcd, 
                lane_info=pickle.load(open(os.path.join(data_root, f"carla/{map_name}_lane_info.pkl"), "rb")),
                lane_areas=lane_areas,
                lane_planes=pickle.load(open(os.path.join(data_root, f"carla/{map_name}_ground_planes.pkl"), "rb")),
                method="map"
            )
            
            # Get object segments
            lidar_seg = lidar_segmentation(lidar, method="squeezeseq", interface=lidar_seg_api)
            object_segments = filter_segmentation(
                lidar, lidar_seg, lidar_pose,
                in_lane_mask=in_lane_mask,
                point_height=point_height,
                max_range=self.grid_params["range_limit"]
            )
            
            # Get occupied areas
            occupied_areas, occupied_heights = get_occupied_space(
                pcd, object_segments,
                point_height=point_height,
                height_thres=0
            )
            
            # Add ego vehicle area
            ego_bbox = vehicle_data["ego_bbox"]
            ego_area = bbox_to_polygon(ego_bbox)
            occupied_areas.append(ego_area)
            
            # Filter occupied areas by lane areas if available
            if lane_areas is not None:
                occupied_areas = [area for area in occupied_areas 
                                if self.check_in_lane_areas(area, lane_areas)]
            
            # Get free areas
            object_mask = np.zeros(pcd.shape[0], dtype=bool)
            if object_segments:
                object_indices = np.hstack(object_segments)
                object_mask[object_indices] = True
                
            free_areas = get_free_space(
                lidar, lidar_pose, object_mask,
                in_lane_mask=in_lane_mask,
                point_height=point_height,
                max_range=self.grid_params["range_limit"],
                height_thres=0,
                height_tolerance=0.2
            )
            
            # Convert areas to grid map
            # Mark free areas
            for area in free_areas:
                free_mask = self.polygon_to_grid(area, lidar_pose)
                if np.any(free_mask):  # Debug print
                    print(f"Found {np.sum(free_mask)} free cells")
                grid_map[free_mask] = 0
                
            # Mark occupied areas (overrides free areas)
            for area in occupied_areas:
                occupied_mask = self.polygon_to_grid(area, lidar_pose)
                if np.any(occupied_mask):  # Debug print
                    print(f"Found {np.sum(occupied_mask)} occupied cells")
                grid_map[occupied_mask] = 1
            
            # Debug visualization of the first free area
            if free_areas:
                plt.figure(figsize=(10, 10))
                first_area = free_areas[0]
                plt.plot(*first_area.exterior.xy, 'r-', label='Free Area')
                plt.plot(lidar_pose[0], lidar_pose[1], 'go', label='Vehicle Position')
                plt.axis('equal')
                plt.legend()
                plt.savefig(f'debug_free_area_{vehicle_id}.png')
                plt.close()
            
            # Store results
            vehicle_grids[vehicle_id] = {
                "grid_map": grid_map,
                "lidar_pose": lidar_pose,
                "ego_area": ego_area,
                "occupied_areas": occupied_areas,
                "free_areas": free_areas,
                "grid_params": self.grid_params,
                "is_attacker": vehicle_id == attacker_id,
                "is_victim": vehicle_id == victim_id,
                "attacker_id": attacker_id,
                "victim_id": victim_id
            }
            
        return vehicle_grids

    @staticmethod
    def check_in_lane_areas(area, lane_areas):
        """Check if an area is mostly within lane areas"""
        intersection = 0
        for lane_area in lane_areas:
            intersection += area.intersection(lane_area).area
        return intersection > 0.95 * area.area

    def visualize_grid_map(self, grid_map, save_path=None):
        """Visualize occupancy grid map"""
        plt.figure(figsize=(10, 10))
        plt.imshow(grid_map, cmap='viridis', origin='lower')
        
        # Add range circle
        H, W = grid_map.shape
        center = (W/2, H/2)
        range_circle = plt.Circle(
            center, 
            radius=self.grid_params["range_limit"]/self.grid_params["cell_size"], 
            fill=False, 
            color='red', 
            linestyle='--', 
            label='Sensor Range'
        )
        plt.gca().add_patch(range_circle)
        
        plt.colorbar(label='Occupancy State (0:free, 1:occupied, 2:unknown)')
        plt.title('Vehicle Occupancy Grid Map')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

def main():
    # Select the GPU device
    torch.cuda.set_device(1)  # This maps to physical GPU 0 in the current setup
    
    # Initialize
    dataset = OPV2VDataset(root_path=os.path.join(data_root, "OPV2V"), mode="test")
    lidar_seg_api = SqueezeSegInterface()
    generator = OccupancyMapGenerator(
        grid_size=(200, 200),
        cell_size=0.5,
        range_limit=50
    )
    
    # Setup result directory
    result_dir = os.path.join(root, "result/mvig")
    os.makedirs(result_dir, exist_ok=True)
    
    # Process each case
    for case_id, case in dataset.case_generator(tag="multi_frame", index=True, use_lidar=True, use_camera=False):
        case_dir = os.path.join(result_dir, f"{case_id:06d}")
        os.makedirs(case_dir, exist_ok=True)
        
        logging.info(f"Processing case {case_id}")
        
        # Process each frame
        for frame_id in range(len(case)):
            # Generate occupancy maps with case_id
            vehicle_grids = generator.process_frame(case[frame_id], lidar_seg_api, case_id)
            
            # Save results
            save_path = os.path.join(case_dir, f"frame_{frame_id:02d}.npz")
            np.savez_compressed(save_path, vehicle_grids=vehicle_grids)
            
            # Visualize first vehicle's grid map
            if frame_id == 0:
                first_vehicle_id = list(vehicle_grids.keys())[0]
                vis_path = os.path.join(case_dir, f"frame_{frame_id:02d}_vehicle_{first_vehicle_id}_vis.png")
                generator.visualize_grid_map(
                    vehicle_grids[first_vehicle_id]["grid_map"],
                    vis_path
                )

if __name__ == "__main__":
    main()
