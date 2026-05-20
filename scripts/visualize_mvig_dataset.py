import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add root directory to system path
root = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../")
sys.path.append(root)

def load_and_visualize_frame(case_id, frame_id=0, save_dir=None):
    """Load and visualize grid maps for a specific case and frame"""
    # Construct the path to the saved data
    result_dir = os.path.join(root, "result/mvig")
    case_dir = os.path.join(result_dir, f"{case_id:06d}")
    frame_path = os.path.join(case_dir, f"frame_{frame_id:02d}.npz")
    
    # Create save directory if specified
    if save_dir is None:
        save_dir = os.path.join(result_dir, "visualization")
    os.makedirs(save_dir, exist_ok=True)
    
    # Load the data
    data = np.load(frame_path, allow_pickle=True)
    grid_params = data['grid_params'].item()
    vehicle_grids = data['vehicle_grids'].item()
    
    # Print and save basic information
    info_str = f"Grid Parameters:\n"
    for key, value in grid_params.items():
        info_str += f"  {key}: {value}\n"
    info_str += f"\nNumber of vehicles: {len(vehicle_grids)}\n"
    
    # Visualize grid maps for all vehicles
    n_vehicles = len(vehicle_grids)
    n_cols = min(3, n_vehicles)  # Maximum 3 columns
    n_rows = (n_vehicles + n_cols - 1) // n_cols
    
    plt.figure(figsize=(5*n_cols, 5*n_rows))
    
    for i, (vehicle_id, data) in enumerate(vehicle_grids.items()):
        plt.subplot(n_rows, n_cols, i+1)
        grid_map = data['grid_map']
        
        plt.imshow(grid_map, cmap='viridis', origin='lower')
        plt.colorbar(label='Occupancy State\n(0:free, 1:occupied, 2:unknown)')
        plt.title(f'Vehicle {vehicle_id}')
        
        # Add vehicle pose to info string
        pose = data['lidar_pose']
        info_str += f"\nVehicle {vehicle_id} pose:\n"
        info_str += f"  Position: ({pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f})\n"
        info_str += f"  Rotation: ({pose[3]:.2f}, {pose[4]:.2f}, {pose[5]:.2f})\n"
    
    plt.tight_layout()
    
    # Save the visualization
    save_path = os.path.join(save_dir, f"case_{case_id:06d}_frame_{frame_id:02d}.png")
    plt.savefig(save_path)
    plt.close()
    
    # Save the information text
    info_path = os.path.join(save_dir, f"case_{case_id:06d}_frame_{frame_id:02d}_info.txt")
    with open(info_path, 'w') as f:
        f.write(info_str)
    
    print(f"Visualization saved to: {save_path}")
    print(f"Information saved to: {info_path}")
    print(info_str)

def main():
    # Example usage
    case_id = 0  # You can change this to view different cases
    frame_id = 0  # You can change this to view different frames
    save_dir = os.path.join(root, "result/mvig/visualization")
    
    try:
        load_and_visualize_frame(case_id, frame_id, save_dir)
    except FileNotFoundError:
        print(f"Data not found for case {case_id}, frame {frame_id}")
        # List available cases
        result_dir = os.path.join(root, "result/mvig")
        available_cases = sorted([int(d) for d in os.listdir(result_dir) 
                                if os.path.isdir(os.path.join(result_dir, d))])
        print("\nAvailable cases:", available_cases)

if __name__ == "__main__":
    main()
