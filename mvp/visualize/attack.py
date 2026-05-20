import numpy as np
import open3d as o3d
from mvp.data.util import rotation_matrix
import cv2
import matplotlib.pyplot as plt
import matplotlib

from .general import get_xylims
from mvp.config import model_3d_examples
from mvp.data.util import bbox_shift, bbox_rotate, pcd_sensor_to_map, bbox_sensor_to_map
from .general import draw_bbox_2d


def draw_attack(attack, normal_case, attack_case, mode="multi_frame", show=False, save=None, track_frames=None):
    if mode == "multi_frame":
        frame_ids = attack["attack_meta"]["attack_frame_ids"]
        frame_num = len(frame_ids)
        fig, axes = plt.subplots(frame_num, 2, figsize=(40, 20 * frame_num))

        # Create additional figures when trajectory visualization is requested
        if track_frames is not None:
            # Create the main trajectory figure
            track_fig, track_ax = plt.subplots(figsize=(20, 20))
            track_ax.set_title("Object Tracking Visualization", fontsize=20)
            
            # Create a zoomed view for detailed attack-box trajectories
            zoom_ax = None
            zoom_box = None
            zoom_pos = None
            
            # Cache point clouds and box positions from different frames
            all_pointclouds = []
            all_attack_boxes = []
            all_victim_boxes = []
            
            # Collect data from all requested frames
            for frame_id in track_frames:
                # Collect point-cloud data
                pointcloud_frame = np.vstack([pcd_sensor_to_map(vehicle_data["lidar"], vehicle_data["lidar_pose"])[:,:3] 
                                             for vehicle_id, vehicle_data in attack_case[frame_id].items()])
                all_pointclouds.append(pointcloud_frame)
                
                # Collect attack-box locations
                if frame_id in frame_ids:
                    frame_idx = frame_ids.index(frame_id)
                    attack_box = attack["attack_meta"]["bboxes"][frame_idx]
                    attacker_vehicle_id = attack["attack_meta"]["attacker_vehicle_id"]
                    attack_box_map = bbox_sensor_to_map(attack_box, attack_case[frame_id][attacker_vehicle_id]["lidar_pose"])
                    all_attack_boxes.append(attack_box_map)
                
                # Collect victim detection boxes
                victim_vehicle_id = attack["attack_meta"]["victim_vehicle_id"]
                if "pred_bboxes" in attack_case[frame_id][victim_vehicle_id]:
                    victim_boxes = bbox_sensor_to_map(attack_case[frame_id][victim_vehicle_id]["pred_bboxes"], 
                                                     attack_case[frame_id][victim_vehicle_id]["lidar_pose"])
                    all_victim_boxes.append(victim_boxes)
                else:
                    all_victim_boxes.append(None)
            
            # Draw the trajectory view
            if all_pointclouds:
                # Merge point clouds to determine plot limits
                try:
                    all_points = np.vstack([pc for pc in all_pointclouds if pc.size > 0])
                    xlim, ylim = get_xylims(all_points)
                    track_ax.set_xlim(xlim)
                    track_ax.set_ylim(ylim)
                except:
                    # Fall back to a default view when merging fails
                    track_ax.set_xlim([-100, 100])
                    track_ax.set_ylim([-100, 100])
                
                # # 1. Tighten the main view bounds
                # xlim = (-180, -70)  # Manually set the x-axis range
                # ylim = (50, 160)    # Manually set the y-axis range
                # track_ax.set_xlim(xlim)
                # track_ax.set_ylim(ylim)
                
                # Use the first frame point cloud as background
                track_ax.scatter(all_pointclouds[0][:,0], all_pointclouds[0][:,1], s=0.01, c="black", alpha=0.3)
                
                # 2. Draw detection boxes and trajectories for all vehicles
                # Draw the attack-box trajectory first
                if all_attack_boxes:
                    attack_x = [box[0] for box in all_attack_boxes]
                    attack_y = [box[1] for box in all_attack_boxes]
                    # 3. Use a thinner trajectory line
                    track_ax.plot(attack_x, attack_y, 'r-', linewidth=1.5, label="Attack Box Trajectory")
                    track_ax.scatter(attack_x, attack_y, c='red', s=40, zorder=5)
                    
                    # Determine the zoom window
                    zoom_pos = (min(attack_x), min(attack_y), max(attack_x) - min(attack_x), max(attack_y) - min(attack_y))
                    # Expand the zoom window slightly
                    zoom_margin = max(zoom_pos[2], zoom_pos[3]) * 0.5
                    zoom_pos = (zoom_pos[0] - zoom_margin, zoom_pos[1] - zoom_margin, 
                               zoom_pos[2] + 2*zoom_margin, zoom_pos[3] + 2*zoom_margin)
                    
                    # Create a dedicated zoomed figure
                    zoom_fig, zoom_ax = plt.subplots(figsize=(10, 10))
                    zoom_ax.set_title("Zoomed Attack Box Trajectory", fontsize=15)
                    zoom_ax.set_xlim(zoom_pos[0], zoom_pos[0] + zoom_pos[2])
                    zoom_ax.set_ylim(zoom_pos[1], zoom_pos[1] + zoom_pos[3])
                    zoom_ax.scatter(all_pointclouds[-1][:,0], all_pointclouds[-1][:,1], s=0.01, c="black", alpha=0.3)
                    zoom_ax.plot(attack_x, attack_y, 'r-', linewidth=2, label="Attack Box")
                    zoom_ax.scatter(attack_x, attack_y, c='red', s=80, zorder=5)
                    
                    # Also draw the first-frame attack box in the zoomed view
                    first_attack_box = all_attack_boxes[0]
                    attack_bboxes = [(np.array([first_attack_box]), None, 'red')]
                    draw_bbox_2d(zoom_ax, attack_bboxes)
                    
                    # Add a legend
                    zoom_ax.legend(fontsize=12)
                    
                    # Save the zoomed figure
                    if save is not None:
                        zoom_save = save.replace('.png', '_zoom.png')
                        zoom_fig.savefig(zoom_save, bbox_inches='tight')
                        plt.close(zoom_fig)
                    
                    # Draw the first-frame attack box in the main figure
                    first_attack_box = all_attack_boxes[0]
                    attack_bboxes = [(np.array([first_attack_box]), None, 'red')]
                    draw_bbox_2d(track_ax, attack_bboxes)
                
                # Track detected boxes with a simple nearest-neighbor matcher
                detection_color = 'green'  # Use green for all detection boxes
                has_drawn_label = False  # Track whether the legend label has been added
                
                # Use a more robust box-tracking routine
                tracked_boxes = []  # Store tracking results across frames
                
                # Initialize tracks from the first frame
                if all_victim_boxes[0] is not None and len(all_victim_boxes[0]) > 0:
                    for box_idx, box in enumerate(all_victim_boxes[0]):
                        tracked_boxes.append({
                            'id': box_idx,
                            'positions': [(track_frames[0], box)],
                            'x': [box[0]],
                            'y': [box[1]]
                        })
                        
                    # Draw the first-frame detection boxes
                    total_bboxes = [(all_victim_boxes[0], None, detection_color)]
                    draw_bbox_2d(track_ax, total_bboxes)
                
                # Track detection boxes in later frames
                for frame_idx in range(1, len(all_victim_boxes)):
                    if all_victim_boxes[frame_idx] is None or len(all_victim_boxes[frame_idx]) == 0:
                        continue
                        
                    current_boxes = all_victim_boxes[frame_idx]
                    current_frame = track_frames[frame_idx]
                    
                    # Match each current box to the nearest existing track
                    for box in current_boxes:
                        min_dist = float('inf')
                        best_track = None
                        
                        for track in tracked_boxes:
                            last_pos = track['positions'][-1][1]
                            dist = np.sqrt((box[0] - last_pos[0])**2 + (box[1] - last_pos[1])**2)
                            
                            if dist < min_dist:
                                min_dist = dist
                                best_track = track
                        
                        # Treat the box as the same object if it stays close enough
                        if min_dist < 10.0 and best_track is not None:  # 10-meter threshold, adjustable if needed
                            best_track['positions'].append((current_frame, box))
                            best_track['x'].append(box[0])
                            best_track['y'].append(box[1])
                        else:
                            # Start a new track
                            new_id = len(tracked_boxes)
                            tracked_boxes.append({
                                'id': new_id,
                                'positions': [(current_frame, box)],
                                'x': [box[0]],
                                'y': [box[1]]
                            })
                
                # Draw trajectories
                for track in tracked_boxes:
                    if len(track['x']) > 1:  # Only draw trajectories with multiple observations
                        if not has_drawn_label:
                            track_ax.plot(track['x'], track['y'], '-', color=detection_color, linewidth=1.5, 
                                         label="Vehicle Detections")
                            has_drawn_label = True
                        else:
                            track_ax.plot(track['x'], track['y'], '-', color=detection_color, linewidth=1.5)
                        
                        # Draw trajectory points
                        track_ax.scatter(track['x'], track['y'], c=detection_color, s=40, alpha=0.7)
                
                # Add a legend
                track_ax.legend(fontsize=12, loc='upper right')
                
                # Save the trajectory figure
                if save is not None:
                    track_save = save.replace('.png', '_trajectory.png')
                    track_fig.savefig(track_save, bbox_inches='tight')
                    plt.close(track_fig)

        # draw normal case first
        for case_id, case in enumerate([normal_case, attack_case]):
            for frame_id in frame_ids:
                if frame_num <= 1:
                    ax = axes[case_id]
                else:
                    ax = axes[frame_ids.index(frame_id)][case_id]

                # draw point clouds
                # pointcloud_all = pcd_sensor_to_map(case[frame_id][attack["attack_opts"]["attacker_vehicle_id"]]["lidar"], case[frame_id][attack["attack_opts"]["attacker_vehicle_id"]]["lidar_pose"])[:,:3]
                pointcloud_all = np.vstack([pcd_sensor_to_map(vehicle_data["lidar"], vehicle_data["lidar_pose"])[:,:3] for vehicle_id, vehicle_data in case[frame_id].items()])
                xlim, ylim = get_xylims(pointcloud_all)
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                # ax.set_aspect('equal', adjustable='box')
                ax.scatter(pointcloud_all[:,0], pointcloud_all[:,1], s=0.01, c="black")

                # label the location of attacker and victim
                attacker_vehicle_id = attack["attack_meta"]["attacker_vehicle_id"]
                attacker_vehicle_data = case[frame_id][attacker_vehicle_id]
                victim_vehicle_id = attack["attack_meta"]["victim_vehicle_id"]
                victim_vehicle_data = case[frame_id][victim_vehicle_id]
                ax.scatter(*victim_vehicle_data["lidar_pose"][:2].tolist(), s=100, c="green")
                ax.scatter(*attacker_vehicle_data["lidar_pose"][:2].tolist(), s=100, c="red")

                # draw gt/result bboxes
                total_bboxes = []
                if "gt_bboxes" in victim_vehicle_data:
                    total_bboxes.append((bbox_sensor_to_map(victim_vehicle_data["gt_bboxes"], victim_vehicle_data["lidar_pose"]), victim_vehicle_data["object_ids"], "g"))
                if "result_bboxes" in victim_vehicle_data:
                    total_bboxes.append((bbox_sensor_to_map(victim_vehicle_data["result_bboxes"], victim_vehicle_data["lidar_pose"]), None, "r"))
                
                # Only draw the attack box in the attacked scene
                if case_id == 1:  # attack_case
                    # label the position of spoofing/removal
                    bbox = attack["attack_meta"]["bboxes"][frame_ids.index(frame_id)]
                    bbox = bbox_sensor_to_map(bbox, attacker_vehicle_data["lidar_pose"])
                    total_bboxes.append((bbox[None,:], None, 'red'))

                draw_bbox_2d(ax, total_bboxes)
                
                # Add a title to distinguish the normal and attacked scenes
                title = f"Normal Scene (Frame {frame_id})" if case_id == 0 else f"Attack Scene (Frame {frame_id})"
                ax.set_title(title, fontsize=20)
    else:
        raise NotImplementedError()

    if show:
        plt.show()
    if save is not None:
        plt.savefig(save)
    plt.close()


def draw_attack_trace(trace, show=False, save=None):
    pass
