import numpy as np
import open3d as o3d
from mvp.data.util import rotation_matrix
import cv2
import matplotlib.pyplot as plt
import matplotlib

from .general import get_xylims, draw_bbox_2d, draw_bboxes_2d, draw_polygons, draw_pointclouds, show_or_save
from mvp.config import model_3d_examples
from mvp.data.util import bbox_shift, bbox_rotate, pcd_sensor_to_map, bbox_sensor_to_map
from mvp.config import color_map
from mvp.tools.sensor_calib import parse_lidar_bboxes, parse_camera_bboxes


def draw_ground_segmentation(pcd_data, inliers, show=False, save=None):
    fig, ax = plt.subplots(figsize=(30,30))
    pointcloud = pcd_data[:,:3]
    xlim, ylim = get_xylims(pointcloud)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal', adjustable='box')

    ax.scatter(pointcloud[:,0], pointcloud[:,1], s=0.01, c="black")
    ax.scatter(pointcloud[inliers,0], pointcloud[inliers,1], s=0.01, c="red") 
    if show:
        plt.show()
    if save is not None:
        plt.savefig(save)
    plt.clf()


def draw_sensor_calib(pcd_on_camera, camera_image, camera_seg=None, lidar_seg=None, show=False, save=None):
    image_shape = camera_image.shape
    fig, ax = plt.subplots()

    image_extent = [0, image_shape[1], image_shape[0], 0]
    if camera_seg is not None:
        camera_image = camera_seg["img"]
    ax.imshow(camera_image, origin="upper", extent=image_extent)

    if pcd_on_camera is not None:
        in_screen_mask = (pcd_on_camera[:,2] > 0)
        if lidar_seg is not None:
            # for i, info in enumerate(lidar_seg["info"]):
            #     points = pcd_on_camera[info["indices"]]
            #     ax.scatter(points[:,0], points[:,1], s=0.02, color=[0,0,(i+1)/17])
            classes = np.unique(lidar_seg["class"])
            for class_id in classes.tolist():
                class_mask = lidar_seg["class"] == class_id
                indices = np.argwhere(in_screen_mask * class_mask > 0).reshape(-1)
                points = pcd_on_camera[indices,:]
                ax.scatter(points[:,0], points[:,1], s=0.02, color=(np.array(color_map[class_id])/255).tolist())
        else:
            ax.scatter(pcd_on_camera[:,0], pcd_on_camera[:,1], s=0.02, c="blue")
    ax.set_xlim(image_extent[:2])
    ax.set_ylim(image_extent[-2:])

    if show:
        plt.show()
    if save is not None:
        plt.savefig(save)
    plt.clf()


def draw_polygon_areas(case, show=False, save=None, tag=""):
    fig, ax = plt.subplots(figsize=(30,30))
    color_map = ["r", "g", "b", "k", "y"]

    for i, vehicle_id in enumerate(case):
        vehicle_data = case[vehicle_id]

        if "lidar" in vehicle_data and vehicle_data["lidar"] is not None:
            lidar = vehicle_data["lidar"]
            lidar_pose = vehicle_data["lidar_pose"]
            pcd = pcd_sensor_to_map(lidar, lidar_pose)
            plt.scatter(pcd[:,0], pcd[:,1], s=0.1, c=color_map[i])
        
        if "gt_bboxes" in vehicle_data:
            bboxes_to_draw = [(bbox_sensor_to_map(vehicle_data["gt_bboxes"], vehicle_data["lidar_pose"]), None, "g")]
            draw_bbox_2d(ax, bboxes_to_draw)
        
        if "pred_bboxes" in vehicle_data:
            bboxes_to_draw = [(bbox_sensor_to_map(vehicle_data["pred_bboxes"], vehicle_data["lidar_pose"]), None, color_map[i])]
            draw_bbox_2d(ax, bboxes_to_draw)

        if "free_areas" + tag in vehicle_data:
            for area in vehicle_data["free_areas" + tag]:
                x, y = area.exterior.coords.xy
                plt.fill(x, y, color_map[i], alpha=0.2)
        if "occupied_areas" + tag in vehicle_data:
            for area in vehicle_data["occupied_areas" + tag]:
                x, y = area.exterior.coords.xy
                plt.plot(x, y, color_map[i], alpha=0.8)
        if "ego_area" in vehicle_data:
            area = vehicle_data["ego_area"]
            x, y = area.exterior.coords.xy
            plt.plot(x, y, color_map[i])
            plt.text(x[0], y[0], str(vehicle_id))

    ax.set_aspect('equal', adjustable='box')

    if show:
        plt.show()
    if save is not None:
        plt.savefig(save)
    plt.close()


def draw_object_tracking(point_clouds, detections, predictions, show=False, save=None):
    frame_num = len(detections)
    fig, axes = plt.subplots(frame_num, 1, figsize=(10, 10 * frame_num))

    for frame_id in range(frame_num):
        point_cloud = point_clouds[frame_id]
        ax = axes[frame_id]
        xlim, ylim = get_xylims(point_cloud)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal', adjustable='box')
        ax.scatter(point_cloud[:,0], point_cloud[:,1], s=0.01, c="black")
        ax.scatter(predictions[frame_id][:,0], predictions[frame_id][:,1], s=50, c="red")
        ax.scatter(detections[frame_id][:,0], detections[frame_id][:,1], s=50, c="blue")

    if show:
        plt.show()
    if save is not None:
        plt.savefig(save)
    plt.clf()


def visualize_defense(case, metrics, show=False, save=None, track_frames=None):
    # Check whether this is a multi-frame defense visualization.
    is_multi_frame = isinstance(metrics, list) and len(metrics) > 1
    
    # Create the main full-scene visualization.
    fig, ax = plt.subplots(figsize=(30,30))
    vehicle_color_map = ["r", "g", "b", "k", "y"]
    
    # Select the frame used for the main visualization.
    if is_multi_frame:
        # In multi-frame mode, use the first frame as the primary view.
        frame_data = case[0]
        frame_metrics = metrics[0]
    else:
        # In single-frame mode, use the provided frame.
        frame_data = case[-1]
        frame_metrics = metrics[0] if isinstance(metrics, list) else metrics
    
    vehicle_ids = list(frame_data.keys())
    

    # Create an auxiliary trajectory figure when requested.
    if track_frames is not None and is_multi_frame:
        # Create the main trajectory figure.
        track_fig, track_ax = plt.subplots(figsize=(20, 20))
        track_ax.set_title("Defense Tracking Visualization", fontsize=20)
        
        # Cache point clouds and box/area data across frames.
        all_pointclouds = []
        all_pred_boxes = {}  # Indexed by vehicle id.
        all_gt_boxes = {}    # Ground-truth boxes indexed by vehicle id.
        all_occupied_areas = {}  # Occupied areas indexed by vehicle id.
        all_free_areas = {}      # Free areas indexed by vehicle id.
        
        # Gather data from all requested frames.
        for frame_idx, frame_id in enumerate(track_frames):
            if frame_id >= len(case):
                continue
                
            # Collect point-cloud data.
            try:
                pointcloud_frame = np.vstack([
                    pcd_sensor_to_map(vehicle_data["lidar"], vehicle_data["lidar_pose"])[:,:3] 
                    for vehicle_id, vehicle_data in case[frame_id].items()
                    if "lidar" in vehicle_data and vehicle_data["lidar"] is not None
                ])
                all_pointclouds.append(pointcloud_frame)
            except:
                all_pointclouds.append(np.zeros((1, 3)))
            
            # Collect predicted boxes, ground-truth boxes, and occupancy areas.
            for vehicle_id in vehicle_ids:
                # Initialize per-vehicle storage.
                if vehicle_id not in all_pred_boxes:
                    all_pred_boxes[vehicle_id] = []
                if vehicle_id not in all_gt_boxes:
                    all_gt_boxes[vehicle_id] = []
                if vehicle_id not in all_occupied_areas:
                    all_occupied_areas[vehicle_id] = []
                if vehicle_id not in all_free_areas:
                    all_free_areas[vehicle_id] = []
                
                # Collect per-frame boxes and polygon areas.
                if frame_id < len(case) and vehicle_id in case[frame_id]:
                    vehicle_data = case[frame_id][vehicle_id]
                    
                    # Collect predicted boxes.
                    if "pred_bboxes" in vehicle_data:
                        pred_boxes = bbox_sensor_to_map(
                            vehicle_data["pred_bboxes"], 
                            vehicle_data["lidar_pose"]
                        )
                        all_pred_boxes[vehicle_id].append(pred_boxes)
                    else:
                        all_pred_boxes[vehicle_id].append(None)
                    
                    # Collect ground-truth boxes.
                    if "gt_bboxes" in vehicle_data:
                        gt_boxes = bbox_sensor_to_map(
                            vehicle_data["gt_bboxes"], 
                            vehicle_data["lidar_pose"]
                        )
                        all_gt_boxes[vehicle_id].append(gt_boxes)
                    else:
                        all_gt_boxes[vehicle_id].append(None)
                    
                    # Collect occupied and free areas.
                    if "occupied_areas" in vehicle_data:
                        all_occupied_areas[vehicle_id].append(vehicle_data["occupied_areas"])
                    else:
                        all_occupied_areas[vehicle_id].append([])
                    
                    if "free_areas" in vehicle_data:
                        all_free_areas[vehicle_id].append(vehicle_data["free_areas"])
                    else:
                        all_free_areas[vehicle_id].append([])
                else:
                    all_pred_boxes[vehicle_id].append(None)
                    all_gt_boxes[vehicle_id].append(None)
                    all_occupied_areas[vehicle_id].append([])
                    all_free_areas[vehicle_id].append([])
        
        # Draw the trajectory figure.
        if all_pointclouds:
            # Merge point clouds to determine the plot limits.
            try:
                all_points = np.vstack([pc for pc in all_pointclouds if pc.size > 0])
                xlim, ylim = get_xylims(all_points)
                track_ax.set_xlim(xlim)
                track_ax.set_ylim(ylim)
            except:
                # Fall back to a default view if merging fails.
                track_ax.set_xlim([-100, 100])
                track_ax.set_ylim([-100, 100])

            # 1. Tighten the main-view bounds.
            # xlim = (-180, -70)  # Manually set the x-axis range.
            # ylim = (50, 160)    # Manually set the y-axis range.
            # track_ax.set_xlim(xlim)
            # track_ax.set_ylim(ylim)
            
            # Draw the first-frame point cloud as the background.
            if len(all_pointclouds) > 0 and all_pointclouds[0].size > 0:
                track_ax.scatter(all_pointclouds[0][:,0], all_pointclouds[0][:,1], s=0.01, c="black", alpha=0.3)
            
            # Draw occupied and free regions from the first frame.
            for vehicle_idx, vehicle_id in enumerate(vehicle_ids):
                vehicle_color = vehicle_color_map[vehicle_idx % len(vehicle_color_map)]
                
                if vehicle_id in all_occupied_areas and len(all_occupied_areas[vehicle_id]) > 0:
                    # Draw occupied areas from the first frame.
                    first_occupied = all_occupied_areas[vehicle_id][0]
                    draw_polygons(track_ax, first_occupied, color=vehicle_color, alpha=0.4, fill=True, border=True, linewidth=0.5)
                
                if vehicle_id in all_free_areas and len(all_free_areas[vehicle_id]) > 0:
                    # Draw free areas from the first frame.
                    first_free = all_free_areas[vehicle_id][0]
                    draw_polygons(track_ax, first_free, color=vehicle_color, alpha=0.1, fill=True, border=False)
            
            # Draw detection boxes from the first frame.
            for vehicle_idx, vehicle_id in enumerate(vehicle_ids):
                # Draw ground-truth boxes in green.
                if vehicle_id in all_gt_boxes and len(all_gt_boxes[vehicle_id]) > 0 and all_gt_boxes[vehicle_id][0] is not None:
                    boxes = all_gt_boxes[vehicle_id][0]
                    draw_bboxes_2d(track_ax, boxes, None, color="green", linewidth=0.8)
                
                # Draw predicted boxes in red.
                if vehicle_id in all_pred_boxes and len(all_pred_boxes[vehicle_id]) > 0 and all_pred_boxes[vehicle_id][0] is not None:
                    boxes = all_pred_boxes[vehicle_id][0]
                    draw_bboxes_2d(track_ax, boxes, None, color="red", linewidth=0.8)
            
            # Build trajectories for predicted boxes.
            pred_tracked_boxes = []
            
            # Initialize tracks from the first frame.
            for vehicle_id in vehicle_ids:
                if vehicle_id in all_pred_boxes and len(all_pred_boxes[vehicle_id]) > 0 and all_pred_boxes[vehicle_id][0] is not None:
                    boxes = all_pred_boxes[vehicle_id][0]
                    for box_idx, box in enumerate(boxes):
                        # Start a new track.
                        pred_tracked_boxes.append({
                            'positions': [(0, box)],  # (frame_idx, box)
                            'vehicle_id': vehicle_id,
                            'box_idx': box_idx,
                            'x': [box[0]],  # List of x coordinates.
                            'y': [box[1]]   # List of y coordinates.
                        })
            
            # Match predicted boxes from later frames to existing tracks.
            for frame_idx in range(1, len(track_frames)):
                # Gather all predicted boxes in the current frame.
                current_boxes = []
                for vehicle_id in vehicle_ids:
                    if (vehicle_id in all_pred_boxes and 
                        frame_idx < len(all_pred_boxes[vehicle_id]) and 
                        all_pred_boxes[vehicle_id][frame_idx] is not None):
                        
                        boxes = all_pred_boxes[vehicle_id][frame_idx]
                        for box_idx, box in enumerate(boxes):
                            current_boxes.append({
                                'box': box,
                                'vehicle_id': vehicle_id,
                                'box_idx': box_idx
                            })
                
                # Match current-frame boxes to existing tracks.
                matched_indices = set()
                
                # Find the best match for each existing track.
                for track_idx, track in enumerate(pred_tracked_boxes):
                    if not current_boxes:  # No boxes are available to match.
                        continue
                        
                    last_frame_idx, last_box = track['positions'][-1]
                    
                    # Compute distances to all candidate boxes.
                    distances = []
                    for box_idx, box_info in enumerate(current_boxes):
                        if box_idx in matched_indices:  # Skip boxes that are already matched.
                            distances.append(float('inf'))
                            continue
                            
                        box = box_info['box']
                        # Compute Euclidean distance.
                        dist = np.sqrt((last_box[0] - box[0])**2 + (last_box[1] - box[1])**2)
                        distances.append(dist)
                    
                    # Select the closest box.
                    min_dist_idx = np.argmin(distances)
                    min_dist = distances[min_dist_idx]
                    
                    # Accept the match when the distance is below the threshold.
                    if min_dist < 10.0:  # Distance threshold; tune it if needed.
                        matched_box = current_boxes[min_dist_idx]
                        matched_indices.add(min_dist_idx)
                        
                        # Update the matched track.
                        track['positions'].append((frame_idx, matched_box['box']))
                        track['x'].append(matched_box['box'][0])
                        track['y'].append(matched_box['box'][1])
                
                # Start new tracks for unmatched boxes.
                for box_idx, box_info in enumerate(current_boxes):
                    if box_idx not in matched_indices:
                        # Start a new track.
                        pred_tracked_boxes.append({
                            'positions': [(frame_idx, box_info['box'])],
                            'vehicle_id': box_info['vehicle_id'],
                            'box_idx': box_info['box_idx'],
                            'x': [box_info['box'][0]],
                            'y': [box_info['box'][1]]
                        })
            
            # Draw trajectories last so they remain visible on top.
            has_drawn_label = False
            for track in pred_tracked_boxes:
                if len(track['positions']) > 1:  # Only draw trajectories with multiple observations.
                    if not has_drawn_label:
                        track_ax.plot(track['x'], track['y'], '-', color='red', linewidth=0.8, 
                                     label="Prediction Trajectories")
                        has_drawn_label = True
                    else:
                        track_ax.plot(track['x'], track['y'], '-', color='red', linewidth=0.8)
                    
                    # Draw trajectory points.
                    track_ax.scatter(track['x'], track['y'], c='red', s=10, alpha=0.9)
            
            # Add legend.
            # track_ax.legend(fontsize=12, loc='upper right')
            
            # Save the trajectory figure.
            if save is not None:
                track_save = save.replace('.png', '_trajectory.png')
                track_fig.savefig(track_save, bbox_inches='tight')
                plt.close(track_fig)

    # Draw the main visualization.
    for i, vehicle_id in enumerate(vehicle_ids):
        if frame_data[vehicle_id]["lidar"] is not None:
            draw_pointclouds(ax, pcd_sensor_to_map(frame_data[vehicle_id]["lidar"], frame_data[vehicle_id]["lidar_pose"]), color=vehicle_color_map[i])
        draw_polygons(ax, frame_data[vehicle_id]["free_areas"], color=vehicle_color_map[i], alpha=0.2, border=False)
        draw_polygons(ax, frame_data[vehicle_id]["occupied_areas"], color=vehicle_color_map[i], alpha=0.6, fill=True, border=False, linewidth=0.5)
        draw_polygons(ax, frame_data[vehicle_id]["ego_area"], color=vehicle_color_map[i], alpha=0.6, fill=True, border=False, linewidth=0.5)
        draw_bboxes_2d(ax, frame_data[vehicle_id]["ego_bbox"][np.newaxis, :], None, color="g", linewidth=1.0)

    for i, vehicle_id in enumerate(vehicle_ids):
        if "pred_bboxes" in frame_data[vehicle_id]:
            draw_bboxes_2d(ax, bbox_sensor_to_map(frame_data[vehicle_id]["gt_bboxes"], frame_data[vehicle_id]["lidar_pose"]), None, color="g", linewidth=0.8)
            draw_bboxes_2d(ax, bbox_sensor_to_map(frame_data[vehicle_id]["pred_bboxes"], frame_data[vehicle_id]["lidar_pose"]), None, color="r", linewidth=0.8)
    
    error_areas = []
    for vehicle_id in frame_metrics:
        for t in ["spoof", "remove"]:
            if t in frame_metrics[vehicle_id]:
                error_areas += [x[0] for x in frame_metrics[vehicle_id][t]]
    draw_polygons(ax, error_areas, color="y", alpha=0.8, border=False)

    ax.set_aspect('equal', adjustable='box')
    
    # Add title.
    if is_multi_frame:
        plt.title("Multi-Frame Defense Visualization (First Frame)", fontsize=20)
    else:
        plt.title("Defense Visualization", fontsize=20)
    
    # Save the full visualization.
    if save:
        plt.savefig(save)
        # Save point cloud only visualization
        pointcloud_save = save.replace('.png', '_pointcloud.png')
        save_pointcloud_only(case, pointcloud_save, is_multi_frame=is_multi_frame)
    if show:
        plt.show()
    plt.close(fig)
    
    # Create a grid-only visualization.
    if save:
        grid_save = save.replace('.png', '_grid.png')
        fig_grid, ax_grid = plt.subplots(figsize=(30,30))
        
        # Draw occupied grids and free-space regions.
        for i, vehicle_id in enumerate(vehicle_ids):
            # Draw free areas with lower opacity.
            draw_polygons(ax_grid, frame_data[vehicle_id]["free_areas"], color=vehicle_color_map[i], alpha=0.2, border=False)
            # Draw occupied areas with higher opacity.
            draw_polygons(ax_grid, frame_data[vehicle_id]["occupied_areas"], color=vehicle_color_map[i], alpha=0.6, fill=True, border=False, linewidth=0.5)
            # Draw the ego area of each vehicle.
            draw_polygons(ax_grid, frame_data[vehicle_id]["ego_area"], color=vehicle_color_map[i], alpha=0.6, fill=True, border=False, linewidth=0.5)
            # Draw the ego-vehicle box.
            draw_bboxes_2d(ax_grid, frame_data[vehicle_id]["ego_bbox"][np.newaxis, :], None, color="g", linewidth=1.0)
        
        # Draw ground-truth and predicted boxes.
        for i, vehicle_id in enumerate(vehicle_ids):
            if "pred_bboxes" in frame_data[vehicle_id]:
                draw_bboxes_2d(ax_grid, bbox_sensor_to_map(frame_data[vehicle_id]["gt_bboxes"], frame_data[vehicle_id]["lidar_pose"]), None, color="g", linewidth=0.8)
                draw_bboxes_2d(ax_grid, bbox_sensor_to_map(frame_data[vehicle_id]["pred_bboxes"], frame_data[vehicle_id]["lidar_pose"]), None, color="r", linewidth=0.8)
        
        # Draw error regions.
        draw_polygons(ax_grid, error_areas, color="y", alpha=0.8, border=False)
        
        # Add title.
        if is_multi_frame:
            ax_grid.set_title("Multi-Frame Defense Grid Visualization (First Frame)", fontsize=20)
        else:
            ax_grid.set_title("Defense Grid Visualization", fontsize=20)
            
        ax_grid.set_aspect('equal', adjustable='box')
        plt.savefig(grid_save)
        plt.close(fig_grid)
        
        # # In multi-frame mode, optionally create per-frame visualizations.
        # if is_multi_frame and track_frames is not None:
        #     for frame_idx, frame_id in enumerate(track_frames):
        #         if frame_id >= len(case) or frame_idx >= len(metrics):
        #             continue
                    
        #         frame_save = save.replace('.png', f'_frame{frame_id}.png')
        #         frame_fig, frame_ax = plt.subplots(figsize=(20, 20))
                
        #         # Draw point cloud.
        #         if frame_idx < len(all_pointclouds) and all_pointclouds[frame_idx].size > 0:
        #             frame_ax.scatter(all_pointclouds[frame_idx][:,0], all_pointclouds[frame_idx][:,1], s=0.01, c="black", alpha=0.5)
                
        #         # Draw occupied and free regions.
        #         for vehicle_idx, vehicle_id in enumerate(vehicle_ids):
        #             vehicle_color = vehicle_color_map[vehicle_idx % len(vehicle_color_map)]
                    
        #             # Draw occupied areas.
        #             if vehicle_id in all_occupied_areas and frame_idx < len(all_occupied_areas[vehicle_id]):
        #                 occupied_areas = all_occupied_areas[vehicle_id][frame_idx]
        #                 draw_polygons(frame_ax, occupied_areas, color=vehicle_color, alpha=0.4, fill=True, border=True, linewidth=0.5)
                    
        #             # Draw free areas.
        #             if vehicle_id in all_free_areas and frame_idx < len(all_free_areas[vehicle_id]):
        #                 free_areas = all_free_areas[vehicle_id][frame_idx]
        #                 draw_polygons(frame_ax, free_areas, color=vehicle_color, alpha=0.1, fill=True, border=False)
                
        #         # Draw detection boxes.
        #         for vehicle_idx, vehicle_id in enumerate(vehicle_ids):
        #             # Draw ground-truth boxes in green.
        #             if vehicle_id in all_gt_boxes and frame_idx < len(all_gt_boxes[vehicle_id]) and all_gt_boxes[vehicle_id][frame_idx] is not None:
        #                 boxes = all_gt_boxes[vehicle_id][frame_idx]
        #                 draw_bboxes_2d(frame_ax, boxes, None, color="green", linewidth=0.8)
                    
        #             # Draw predicted boxes in red.
        #             if vehicle_id in all_pred_boxes and frame_idx < len(all_pred_boxes[vehicle_id]) and all_pred_boxes[vehicle_id][frame_idx] is not None:
        #                 boxes = all_pred_boxes[vehicle_id][frame_idx]
        #                 draw_bboxes_2d(frame_ax, boxes, None, color="red", linewidth=0.8)
                
        #         frame_ax.set_title(f"Defense Frame {frame_id}", fontsize=20)
        #         frame_ax.set_aspect('equal', adjustable='box')
                
        #         # Set axis limits.
        #         try:
        #             xlim, ylim = get_xylims(all_pointclouds[frame_idx])
        #             frame_ax.set_xlim(xlim)
        #             frame_ax.set_ylim(ylim)
        #         except:
        #             # Fall back to default limits.
        #             frame_ax.set_xlim([-100, 100])
        #             frame_ax.set_ylim([-100, 100])
                
        #         plt.savefig(frame_save, bbox_inches='tight')
        #         plt.close(frame_fig)


def draw_roc(value, label, show=False, save=None, multi_frame=False, case_ids=None):
    """
    Draw an ROC curve for either single-frame or multi-frame data.

    Args:
        value: Score array.
        label: Label array where 1 indicates attacker and 0 indicates normal.
        show: Whether to display the figure.
        save: Optional output path.
        multi_frame: Whether the inputs come from multi-frame data.
        case_ids: Case-id array used to group multi-frame samples. When None,
            grouping is skipped and the function falls back to single-frame mode.

    Returns:
        best_TPR, best_FPR, roc_auc, best_thres
    """
    # Process multi-frame data.
    if multi_frame and case_ids is not None:
        # Group samples by case id.
        case_groups = {}
        for i, case_id in enumerate(case_ids):
            if case_id not in case_groups:
                case_groups[case_id] = {'values': [], 'label': None}
            
            case_groups[case_id]['values'].append(value[i])
            # All samples from the same case should share the same label.
            case_groups[case_id]['label'] = label[i]
        
        # Determine the threshold range.
        thres_min = np.min(value) - 0.02
        thres_max = np.max(value) + 0.02
        thresholds = np.arange(thres_min, thres_max, 0.02).tolist()
        
        tpr_data = []
        fpr_data = []
        roc_auc = 0
        best_thres = 0
        best_TPR = 0
        best_FPR = 0
        
        for thres in thresholds:
            # Mark a case as positive if any frame exceeds the threshold.
            TP = 0  # True positives.
            FP = 0  # False positives.
            P = 0   # Total actual positives.
            N = 0   # Total actual negatives.
            
            for case_id, data in case_groups.items():
                # A case is positive when any frame score exceeds the threshold.
                case_positive = any(v > thres for v in data['values'])
                case_is_attacker = data['label'] > 0
                
                if case_is_attacker:
                    P += 1
                    if case_positive:
                        TP += 1
                else:
                    N += 1
                    if case_positive:
                        FP += 1
            
            # Compute true-positive and false-positive rates.
            TPR = TP / max(P, 1)  # Avoid division by zero.
            FPR = FP / max(N, 1)  # Avoid division by zero.
            
            # Update the best threshold.
            if TPR * (1 - FPR) > roc_auc:
                roc_auc = TPR * (1 - FPR)
                best_thres = thres
                best_TPR = TPR
                best_FPR = FPR
            
            tpr_data.append(TPR)
            fpr_data.append(FPR)
            
    else:
        # Original single-frame processing logic.
        tpr_data = []
        fpr_data = []
        roc_auc = 0
        best_thres = 0
        best_TPR = 0
        best_FPR = 0
        
        for thres in np.arange(value.min()-0.02, value.max()+0.02, 0.02).tolist():
            TP = np.sum((value > thres) * (label > 0))
            FP = np.sum((value > thres) * (label <= 0))
            P = np.sum(label > 0)
            N = np.sum(label <= 0)
            
            TPR = TP / max(P, 1)  # Avoid division by zero.
            FPR = FP / max(N, 1)  # Avoid division by zero.
            
            if TPR * (1 - FPR) > roc_auc:
                roc_auc = TPR * (1 - FPR)
                best_thres = thres
                best_TPR = TPR
                best_FPR = FPR
                
            tpr_data.append(TPR)
            fpr_data.append(FPR)
    
    # Draw the ROC curve.
    plt.figure(figsize=(8, 8))
    plt.plot(fpr_data, tpr_data, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('ROC Curve' + (' (Multi-Frame)' if multi_frame else ''))
    
    # Save or display the figure.
    show_or_save(show=show, save=save)
    
    return best_TPR, best_FPR, roc_auc, best_thres


def save_pointcloud_only(case, save_path, is_multi_frame=False):
    """
    Creates and saves a visualization with only the point cloud data, no detection boxes or other elements.
    
    Args:
        case: The case data containing vehicle information
        save_path: Path where to save the visualization
        is_multi_frame: Whether this is a multi-frame case
    """
    # Determine which frame to use
    if is_multi_frame:
        frame_data = case[0]  # Use first frame for multi-frame cases
    else:
        frame_data = case[-1]  # Use last frame for single-frame cases
    
    vehicle_ids = list(frame_data.keys())
    vehicle_color_map = ["r", "g", "b", "k", "y"]
    
    # Create figure for point cloud only
    fig_pc, ax_pc = plt.subplots(figsize=(20, 20))
    
    # Draw only point clouds
    for i, vehicle_id in enumerate(vehicle_ids):
        if "lidar" in frame_data[vehicle_id] and frame_data[vehicle_id]["lidar"] is not None:
            # Convert point cloud to map coordinates
            pc_map = pcd_sensor_to_map(frame_data[vehicle_id]["lidar"], frame_data[vehicle_id]["lidar_pose"])
            # Use more aesthetic colors and point size
            ax_pc.scatter(pc_map[:,0], pc_map[:,1], s=0.5, 
                         c=vehicle_color_map[i % len(vehicle_color_map)], alpha=0.7,
                         label=f"Vehicle {vehicle_id}")
    
    # Add title and legend
    if is_multi_frame:
        ax_pc.set_title("Multi-Frame Point Cloud Visualization", fontsize=20)
    else:
        ax_pc.set_title("Point Cloud Visualization", fontsize=20)
    
    # Add legend
    ax_pc.legend(fontsize=12, loc='upper right')
    
    # Set equal aspect ratio
    ax_pc.set_aspect('equal', adjustable='box')
    
    # Set axis labels
    ax_pc.set_xlabel('X (m)', fontsize=14)
    ax_pc.set_ylabel('Y (m)', fontsize=14)
    
    # Add grid lines to enhance visualization
    ax_pc.grid(True, linestyle='--', alpha=0.3)
    
    # Save the image
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig_pc)
