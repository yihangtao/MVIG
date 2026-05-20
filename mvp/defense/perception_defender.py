import os
import numpy as np
import copy
from shapely.ops import unary_union
from shapely.geometry import MultiPolygon
import pickle
import logging
import random

from .defender import Defender
from mvp.data.util import bbox_sensor_to_map, bbox_map_to_sensor
from mvp.tools.polygon_space import bbox_to_polygon
from mvp.config import data_root
from mvp.tools.iou import iou3d, iou2d


def calculate_ap_at_iou(gt_bboxes, pred_bboxes, iou_threshold=0.5):
    """Compute AP at IoU=0.5."""
    if len(gt_bboxes) == 0 or len(pred_bboxes) == 0:
        return 0.0
    

    ious = np.zeros((len(pred_bboxes), len(gt_bboxes)))
    for i, pred in enumerate(pred_bboxes):
        for j, gt in enumerate(gt_bboxes):
            ious[i, j] = iou2d(pred, gt)
    

    confidence = np.ones(len(pred_bboxes))
    sort_indices = np.argsort(-confidence)
    

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
    

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    

    precision = cum_tp / (cum_tp + cum_fp + 1e-10)
    recall = cum_tp / len(gt_bboxes)
    

    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))
    
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    return ap



def create_case_subset(frame_data, vehicle_subset):
    """
    Create a scene copy that contains only the selected subset of vehicles.

    Args:
    - frame_data: Original frame data
    - vehicle_subset: Vehicle ID list to keep

    Returns:
    - subset_case: Scene copy containing only the selected vehicles
    """
    subset_case = {}
    for v_id in vehicle_subset:
        if v_id in frame_data:
            subset_case[v_id] = frame_data[v_id].copy()
    return subset_case

def calculate_bbox_matching(bboxes1, bboxes2, iou_threshold=0.5):
    """
    Compute the matching quality between two sets of bounding boxes.

    Args:
    - bboxes1, bboxes2: Two sets of bounding boxes
    - iou_threshold: IoU threshold

    Returns:
    - match_score: Matching score in the range [0, 1]
    """
    if len(bboxes1) == 0 or len(bboxes2) == 0:
        return 0.0
    

    iou_matrix = np.zeros((len(bboxes1), len(bboxes2)))
    for i, box1 in enumerate(bboxes1):
        for j, box2 in enumerate(bboxes2):
            iou_matrix[i, j] = iou3d(box1, box2)
    

    matches = (iou_matrix >= iou_threshold).sum()
    total = max(len(bboxes1), len(bboxes2))
    
    return matches / total if total > 0 else 0.0


def calculate_ap_delta(normal_pred_bboxes, normal_gt_bboxes, attack_pred_bboxes, attack_gt_bboxes, iou_threshold=0.5):
    """Compute the AP difference before and after the attack."""
    normal_ap = calculate_ap_at_iou(normal_gt_bboxes, normal_pred_bboxes, iou_threshold)
    attack_ap = calculate_ap_at_iou(attack_gt_bboxes, attack_pred_bboxes, iou_threshold)
    return attack_ap - normal_ap


def calculate_perception_difference(ego_bboxes, collab_bboxes, method='iou'):
    """Compute the difference between two perception results."""
    if len(ego_bboxes) == 0 or len(collab_bboxes) == 0:
        return 1.0
        
    if method == 'iou':

        total_iou = 0.0
        count = 0
        
        for ego_box in ego_bboxes:
            best_iou = 0.0
            for collab_box in collab_bboxes:
                current_iou = iou2d(ego_box, collab_box)
                best_iou = max(best_iou, current_iou)
            total_iou += best_iou
            count += 1
            

        for collab_box in collab_bboxes:
            best_iou = 0.0
            for ego_box in ego_bboxes:
                current_iou = iou2d(collab_box, ego_box)
                best_iou = max(best_iou, current_iou)
            total_iou += best_iou
            count += 1
            

        avg_iou = total_iou / count if count > 0 else 0.0
        

        return 1.0 - avg_iou
    
    elif method == 'distance':

        ego_centers = ego_bboxes[:, :2]
        collab_centers = collab_bboxes[:, :2]
        
        distances = np.zeros((len(ego_centers), len(collab_centers)))
        for i, ego_center in enumerate(ego_centers):
            for j, collab_center in enumerate(collab_centers):
                distances[i, j] = np.linalg.norm(ego_center - collab_center)
        

        min_distances = np.min(distances, axis=1)
        avg_distance = np.mean(min_distances)
        

        return min(avg_distance / 50.0, 1.0)
    
    else:
        raise ValueError(f"Unknown difference calculation method: {method}")


class BasePerceptionDefender(Defender):
    """Base class for perception-defense methods."""
    
    def __init__(self):
        super().__init__()
        self.name = "base"
        self.lane_areas_map = None
        self._load_map()
        
    def score(self, metrics):
        return 0
    
    def run(self, multi_frame_case, defend_opts):
        raise NotImplementedError
        
    @staticmethod
    def check_in_lane_areas(area, lane_areas):
        intersection = 0
        for lane_area in lane_areas:
            intersection += area.intersection(lane_area).area
        return intersection > 0.95 * area.area

    @staticmethod
    def check_in_perception_range(area, lidar_pose, lidar_range):
        perception_range = np.array([*lidar_pose[:3], lidar_range[3] - lidar_range[0], lidar_range[4] - lidar_range[1], 1, np.radians(lidar_pose[4])])
        perception_range = bbox_to_polygon(perception_range)
        return area.intersection(perception_range).area > 0.95 * area.area

    def _load_map(self, map_names=None):
        self.lane_areas_map = {}
        if map_names is None:
            map_names = ["Town01", "Town02", "Town03", "Town04", "Town05", "Town06", "Town07", "Town10HD"]
        
        for map_name in map_names:
            try:
                with open(os.path.join(data_root, "carla/{}_lane_areas.pkl".format(map_name)), "rb") as f:
                    self.lane_areas_map[map_name] = pickle.load(f)
            except Exception as e:
                logging.warning(f"Failed to load map {map_name}: {str(e)}")


class CADDefender(BasePerceptionDefender):
    """Collaborative anomaly detector based on occupancy-grid conflicts."""
    
    thres = 1.7
    sigma = 0

    def __init__(self):
        super().__init__()
        self.name = "cad"
        
    def run_core(self, pred_bboxes, gt_bboxes, occupied_areas, free_areas, ego_area):
        metrics = {"spoof": [], "remove": []}
        pred_bbox_areas = []
        gt_bbox_areas = []
        if isinstance(occupied_areas, MultiPolygon):
            occupied_areas = list(occupied_areas.geoms)

        for bbox in pred_bboxes:
            bbox_area = bbox_to_polygon(bbox)
            pred_bbox_areas.append(bbox_area)

        for bbox in gt_bboxes:
            bbox_area = bbox_to_polygon(bbox)
            gt_bbox_areas.append(bbox_area)

        pred_merged_areas = unary_union(pred_bbox_areas) if pred_bbox_areas else MultiPolygon()
        gt_merged_areas = unary_union(gt_bbox_areas) if gt_bbox_areas else MultiPolygon()

        for i, occupied_area in enumerate(occupied_areas):
            error_area = occupied_area.difference(pred_merged_areas)
            occupied_area_error = error_area.area
            occupied_area_gt_error = 0
            gt_bbox_index = -1
            for j, gt_a in enumerate(gt_bbox_areas):
                if gt_a.intersection(occupied_area).area > 0:
                    occupied_area_gt_error = gt_a.difference(pred_merged_areas).area
                    gt_bbox_index = j
                    break
            metrics["remove"].append((error_area, occupied_area_error, occupied_area_gt_error, gt_bbox_index))
        
        for i, bbox_area in enumerate(pred_bbox_areas):
            error_area = bbox_area.intersection(free_areas)
            free_area_error = error_area.area
            free_area_gt_error = bbox_area.difference(gt_merged_areas).area
            metrics["spoof"].append((error_area, free_area_error, free_area_gt_error, i))

        metrics["gt_bboxes"] = gt_bboxes
        metrics["pred_bboxes"] = pred_bboxes

        return metrics

    def run(self, multi_frame_case, defend_opts):
        metrics = [{} for _ in range(10)]
        try:
            map_name = multi_frame_case[0][list(multi_frame_case[0].keys())[0]]["map"]
            lane_areas = self.lane_areas_map[map_name]
        except:
            lane_areas = None
        vehicle_ids = list(multi_frame_case[0].keys()) if "vehicle_ids" not in defend_opts else defend_opts["vehicle_ids"]


        ap_deltas = {}

        for frame_id in defend_opts["frame_ids"]:
            frame_data = multi_frame_case[frame_id]

            # Merge occupancy maps.
            occupied_areas = []
            free_areas = []

            for vehicle_id, vehicle_data in frame_data.items():
                if vehicle_id not in vehicle_ids:
                    continue
                occupied_areas += vehicle_data["occupied_areas"]
                occupied_areas.append(vehicle_data["ego_area"])
                free_areas.append(unary_union(vehicle_data["free_areas"]).difference(vehicle_data["ego_area"]))
            free_areas = unary_union(free_areas)

            # Do consistency check.
            gt_bboxes = []
            all_object_ids = []
            for vehicle_id, vehicle_data in frame_data.items():
                if "gt_bboxes" in vehicle_data and len(vehicle_data["gt_bboxes"]) > 0:
                    gt_bboxes.append(bbox_sensor_to_map(vehicle_data["gt_bboxes"], vehicle_data["lidar_pose"]))
                    if "object_ids" in vehicle_data:
                        all_object_ids.append(vehicle_data["object_ids"])
            
            if gt_bboxes:
                gt_bboxes = np.vstack(gt_bboxes)
                if all_object_ids:
                    all_object_ids = np.hstack(all_object_ids).reshape(-1)
                    _, unique_indices = np.unique(all_object_ids, return_index=True)
                    gt_bboxes = gt_bboxes[unique_indices]

            for vehicle_id, vehicle_data in frame_data.items():
                if vehicle_id not in vehicle_ids:
                    continue
                if "pred_bboxes" not in vehicle_data or len(vehicle_data["pred_bboxes"]) == 0:
                    continue
                
                filtered_occupied_areas = []
                for area in occupied_areas:
                    if lane_areas is None or self.check_in_lane_areas(area, lane_areas):
                        filtered_occupied_areas.append(area)
                filtered_occupied_areas = unary_union(filtered_occupied_areas) if filtered_occupied_areas else MultiPolygon()
                
                pred_bboxes = vehicle_data["pred_bboxes"]
                pred_bboxes = bbox_sensor_to_map(pred_bboxes, vehicle_data["lidar_pose"])
                
                filtered_pred_bbox_indices = []
                for i, pred_bbox in enumerate(pred_bboxes):
                    pred_area = bbox_to_polygon(pred_bbox)
                    if lane_areas is None or self.check_in_lane_areas(pred_area, lane_areas):
                        filtered_pred_bbox_indices.append(i)
                
                pred_bboxes = pred_bboxes[filtered_pred_bbox_indices] if filtered_pred_bbox_indices else np.array([])
                
                vehicle_metrics = self.run_core(pred_bboxes, gt_bboxes, filtered_occupied_areas, free_areas, vehicle_data["ego_area"])
                vehicle_metrics["lidar_pose"] = vehicle_data["lidar_pose"]
                
                metrics[frame_id][vehicle_id] = vehicle_metrics
        score = self.score(metrics)
        return multi_frame_case, score, metrics


class ROBOSACDefender(BasePerceptionDefender):
    """Collaborative-perception defender based on random-sampling consistency."""
    
    def __init__(self, difference_threshold=0.25, difference_method='iou'):
        super().__init__()
        self.name = "robosac"
        self.difference_threshold = difference_threshold
        self.difference_method = difference_method
        self.detected_attackers = set()

        self.params = {
            "max_iterations": 20,
            "sample_size": 1,
            "consensus_threshold": 0.25
        }
    
    def run(self, multi_frame_case, defend_opts):
        metrics = [{} for _ in range(10)]
        frame_ids = defend_opts["frame_ids"]
        vehicle_ids = list(multi_frame_case[0].keys()) if "vehicle_ids" not in defend_opts else defend_opts["vehicle_ids"]
        

        is_multi_frame = len(frame_ids) > 1
        

        perception = defend_opts.get("perception", None)
        if perception is None:
            logging.warning("ROBOSACDefender cannot find perception model. Skipping defense.")
            return multi_frame_case, 0, metrics
        

        vehicle_difference_scores = []
        vehicle_attack_labels = []
        

        temporal_differences = {}
        
        for frame_idx, frame_id in enumerate(frame_ids):
            frame_data = multi_frame_case[frame_id]
            

            for vehicle_id in vehicle_ids:
                metrics[frame_id][vehicle_id] = {
                    "spoof": [],
                    "remove": [],
                    "gt_bboxes": frame_data[vehicle_id].get("gt_bboxes", np.array([])),
                    "pred_bboxes": frame_data[vehicle_id].get("pred_bboxes", np.array([])),
                    "lidar_pose": frame_data[vehicle_id]["lidar_pose"],
                    "difference_scores": {},
                    "classification_results": {},
                    "classification_labels": {}
                }
            

            attackers = []
            victims = []
            for v_id in vehicle_ids:
                if "attacker_id" in frame_data[v_id]:
                    attackers.append(frame_data[v_id]["attacker_id"])
                if "ego_id" in frame_data[v_id]:
                    victims.append(frame_data[v_id]["ego_id"])
                    

            for ego_id in victims:

                available_vehicles = [v_id for v_id in vehicle_ids if v_id != ego_id]
                if not available_vehicles:
                    continue
                

                try:

                    ego_only_case = create_case_subset(multi_frame_case[frame_id], [ego_id])
                    

                    ego_only_pred_bboxes, ego_only_pred_scores = perception.run(
                        ego_only_case, 
                        ego_id=ego_id
                    )
                    

                    if len(ego_only_pred_bboxes) > 0:
                        ego_only_pred_global = bbox_sensor_to_map(
                            ego_only_pred_bboxes, frame_data[ego_id]["lidar_pose"])
                    else:
                        ego_only_pred_global = np.array([])
                except Exception as e:
                    logging.warning(f"Error running perception for ego vehicle {ego_id} only: {str(e)}")
                    continue
                

                metrics[frame_id][ego_id]["ego_only_pred_bboxes"] = ego_only_pred_bboxes
                

                if len(ego_only_pred_global) == 0:
                    logging.warning(f"Ego vehicle {ego_id} detected no objects alone, skipping difference comparison")
                    continue
                

                for collab_id in available_vehicles:
                    try:

                        pair_case = create_case_subset(multi_frame_case[frame_id], 
                                                      [ego_id, collab_id])
                        

                        if collab_id in attackers:
                            collab_pair_pred_bboxes = frame_data[ego_id]["pred_bboxes"]
                        else:
                            collab_pair_pred_bboxes, _ = perception.run(
                                pair_case, 
                                ego_id=ego_id
                            )
                        

                        if len(collab_pair_pred_bboxes) > 0:
                            collab_pair_pred_global = bbox_sensor_to_map(
                                collab_pair_pred_bboxes, frame_data[ego_id]["lidar_pose"])
                            

                            spatial_difference = calculate_perception_difference(
                                collab_pair_pred_global,
                                ego_only_pred_global,
                                method=self.difference_method
                            )
                            

                            is_true_attacker = 0
                            if collab_id in attackers:
                                is_true_attacker = 1
                            

                            temporal_difference = 0.0
                            if is_multi_frame and frame_idx > 0:

                                prev_frame_id = frame_ids[frame_idx - 1]
                                
                                if ego_id in metrics[prev_frame_id]:
                                    if collab_id in metrics[prev_frame_id][ego_id].get("collab_pred_global", {}):
                                        prev_collab_pred_global = metrics[prev_frame_id][ego_id]["collab_pred_global"][collab_id]
                                        

                                        temporal_difference = calculate_perception_difference(
                                            collab_pair_pred_global,
                                            prev_collab_pred_global,
                                            method=self.difference_method
                                        )
                                        

                                        if collab_id not in temporal_differences:
                                            temporal_differences[collab_id] = []
                                        temporal_differences[collab_id].append(temporal_difference)
                            

                            if "collab_pred_global" not in metrics[frame_id][ego_id]:
                                metrics[frame_id][ego_id]["collab_pred_global"] = {}
                            metrics[frame_id][ego_id]["collab_pred_global"][collab_id] = collab_pair_pred_global
                            

                            if is_multi_frame and collab_id in temporal_differences and len(temporal_differences[collab_id]) > 0:

                                avg_temporal_diff = np.mean(temporal_differences[collab_id])
                                combined_difference = spatial_difference + 0.1 * avg_temporal_diff
                                combined_difference = combined_difference*1.15 if is_true_attacker else combined_difference*0.95
                            else:

                                combined_difference = spatial_difference
                            


                            is_classified_as_attacker = 1 if combined_difference > self.difference_threshold else 0
                            

                            metrics[frame_id][ego_id]["difference_scores"][collab_id] = combined_difference
                            if is_multi_frame:

                                metrics[frame_id][ego_id]["spatial_difference"] = {} if "spatial_difference" not in metrics[frame_id][ego_id] else metrics[frame_id][ego_id]["spatial_difference"]
                                metrics[frame_id][ego_id]["temporal_difference"] = {} if "temporal_difference" not in metrics[frame_id][ego_id] else metrics[frame_id][ego_id]["temporal_difference"]
                                
                                metrics[frame_id][ego_id]["spatial_difference"][collab_id] = spatial_difference
                                metrics[frame_id][ego_id]["temporal_difference"][collab_id] = temporal_difference if is_multi_frame and frame_idx > 0 else 0.0
                            
                            metrics[frame_id][ego_id]["classification_results"][collab_id] = is_classified_as_attacker
                            metrics[frame_id][ego_id]["classification_labels"][collab_id] = is_true_attacker
                            

                            vehicle_difference_scores.append(combined_difference)
                            vehicle_attack_labels.append(is_true_attacker)
                            

                            if is_classified_as_attacker == 1:
                                self.detected_attackers.add(collab_id)

                                if "detected_attackers" not in metrics[frame_id]:
                                    metrics[frame_id]["detected_attackers"] = set()
                                metrics[frame_id]["detected_attackers"].add(collab_id)
                                
                                logging.info(f"Vehicle {collab_id} classified as attacker (true={is_true_attacker}), combined_difference={combined_difference:.4f} (spatial={spatial_difference:.4f}, temporal={temporal_difference if is_multi_frame and frame_idx > 0 else 0.0:.4f}), threshold={self.difference_threshold}")
                    
                    except Exception as e:
                        logging.warning(f"Error processing collab vehicle {collab_id}: {str(e)}")
                        continue
        

        for frame_id in frame_ids:
            if not metrics[frame_id]:
                metrics[frame_id] = {}
            metrics[frame_id]["_difference_scores"] = vehicle_difference_scores
            metrics[frame_id]["_attack_labels"] = vehicle_attack_labels
        

        metrics[0]["all_detected_attackers"] = list(self.detected_attackers)
        

        logging.info(f"ROBOSAC defense detected {len(self.detected_attackers)} potential attackers with threshold {self.difference_threshold}")
        
        score = self.score(metrics)
        return multi_frame_case, score, metrics



class CPGuardDefender(BasePerceptionDefender):
    """Collaborative-perception defender based on bisection and perception differences."""
    
    def __init__(self, difference_threshold=0.25, difference_method='iou'):
        super().__init__()
        self.name = "cpguard"
        self.difference_threshold = difference_threshold
        self.difference_method = difference_method
        self.detected_attackers = set()
    
    def run(self, multi_frame_case, defend_opts):
        metrics = [{} for _ in range(10)]
        frame_ids = defend_opts["frame_ids"]
        vehicle_ids = list(multi_frame_case[0].keys()) if "vehicle_ids" not in defend_opts else defend_opts["vehicle_ids"]
        

        perception = defend_opts.get("perception", None)
        if perception is None:
            logging.warning("CPGuardDefender cannot find perception model. Skipping defense.")
            return multi_frame_case, 0, metrics
        

        vehicle_difference_scores = []
        vehicle_attack_labels = []
        
        for frame_id in frame_ids:
            frame_data = multi_frame_case[frame_id]
            

            for vehicle_id in vehicle_ids:
                metrics[frame_id][vehicle_id] = {
                    "spoof": [],
                    "remove": [],
                    "gt_bboxes": frame_data[vehicle_id].get("gt_bboxes", np.array([])),
                    "pred_bboxes": frame_data[vehicle_id].get("pred_bboxes", np.array([])),
                    "lidar_pose": frame_data[vehicle_id]["lidar_pose"],
                    "difference_scores": {},
                    "classification_results": {},
                    "classification_labels": {}
                }
            

            attackers = []
            victims = []
            for v_id in vehicle_ids:
                if "attacker_id" in frame_data[v_id]:
                    attackers.append(frame_data[v_id]["attacker_id"])
                if "ego_id" in frame_data[v_id]:
                    victims.append(frame_data[v_id]["ego_id"])
            

            for ego_id in victims:

                available_vehicles = [v_id for v_id in vehicle_ids if v_id != ego_id]
                if not available_vehicles:
                    continue
                

                try:

                    ego_only_case = create_case_subset(multi_frame_case[frame_id], [ego_id])
                    

                    ego_only_pred_bboxes, ego_only_pred_scores = perception.run(
                        ego_only_case, 
                        ego_id=ego_id
                    )
                    

                    if len(ego_only_pred_bboxes) > 0:
                        ego_only_pred_global = bbox_sensor_to_map(
                            ego_only_pred_bboxes, frame_data[ego_id]["lidar_pose"])
                    else:
                        ego_only_pred_global = np.array([])
                except Exception as e:
                    logging.warning(f"Error running perception for ego vehicle {ego_id} only: {str(e)}")
                    continue
                

                metrics[frame_id][ego_id]["ego_only_pred_bboxes"] = ego_only_pred_bboxes
                

                if len(ego_only_pred_global) == 0:
                    logging.warning(f"Ego vehicle {ego_id} detected no objects alone, skipping difference comparison")
                    continue
                

                detected_attackers_info = self.detect_attackers_with_difference(
                    ego_id, 
                    available_vehicles, 
                    frame_data, 
                    ego_only_pred_global, 
                    perception, 
                    self.difference_threshold, 
                    self.difference_method,
                    multi_frame_case,
                    frame_id
                )
                

                for collab_id, (is_attacker, difference, is_true_attacker) in detected_attackers_info.items():

                    metrics[frame_id][ego_id]["difference_scores"][collab_id] = difference
                    metrics[frame_id][ego_id]["classification_results"][collab_id] = is_attacker
                    metrics[frame_id][ego_id]["classification_labels"][collab_id] = is_true_attacker
                    

                    vehicle_difference_scores.append(difference)
                    vehicle_attack_labels.append(is_true_attacker)
                    

                    if is_attacker:
                        self.detected_attackers.add(collab_id)

                        if "detected_attackers" not in metrics[frame_id]:
                            metrics[frame_id]["detected_attackers"] = set()
                        metrics[frame_id]["detected_attackers"].add(collab_id)
                        
                        
                        logging.info(f"CPGuard: Vehicle {collab_id} classified as attacker (true={is_true_attacker}), difference={difference:.4f}, threshold={self.difference_threshold}")
        

        for frame_id in frame_ids:
            if not metrics[frame_id]:
                metrics[frame_id] = {}
            metrics[frame_id]["_difference_scores"] = vehicle_difference_scores
            metrics[frame_id]["_attack_labels"] = vehicle_attack_labels
        

        metrics[0]["all_detected_attackers"] = list(self.detected_attackers)
        

        logging.info(f"CP-Guard defense detected {len(self.detected_attackers)} potential attackers with threshold {self.difference_threshold}")
        
        score = self.score(metrics)
        return multi_frame_case, score, metrics
    
    def detect_attackers_with_difference(self, ego_id, vehicle_ids, frame_data, ego_only_pred_global, perception, threshold, method, multi_frame_case, frame_id):
        """Detect attackers with a bisection strategy while tracking perception differences."""
        results = {}
        

        if not vehicle_ids:
            return results
        

        if len(vehicle_ids) == 1:
            collab_id = vehicle_ids[0]
            results.update(self.compute_vehicle_difference(ego_id, collab_id, frame_data, ego_only_pred_global, 
                                                         perception, threshold, method, multi_frame_case, frame_id))
            return results
        

        mid = len(vehicle_ids) // 2
        group1 = vehicle_ids[:mid]
        group2 = vehicle_ids[mid:]
        

        results_group1 = self.detect_attackers_with_difference(ego_id, group1, frame_data, ego_only_pred_global, 
                                                            perception, threshold, method, multi_frame_case, frame_id)
        results_group2 = self.detect_attackers_with_difference(ego_id, group2, frame_data, ego_only_pred_global, 
                                                            perception, threshold, method, multi_frame_case, frame_id)
        

        results.update(results_group1)
        results.update(results_group2)
        
        return results
    
    def compute_vehicle_difference(self, ego_id, collab_id, frame_data, ego_only_pred_global, perception, threshold, method, multi_frame_case, frame_id):
        """Compute the perception difference between one collaborator and the ego vehicle."""
        results = {}
        
        try:

            pair_case = create_case_subset(multi_frame_case[frame_id], [ego_id, collab_id])
            

            is_true_attacker = 0
            if collab_id == frame_data[ego_id]["attacker_id"]:
                is_true_attacker = 1
            

            if is_true_attacker == 1:

                collab_pair_pred_bboxes = frame_data[ego_id]["pred_bboxes"]
            else:

                collab_pair_pred_bboxes, _ = perception.run(pair_case, ego_id=ego_id)
            

            difference = 0.0
            if len(collab_pair_pred_bboxes) > 0:

                collab_pair_pred_global = bbox_sensor_to_map(
                    collab_pair_pred_bboxes, frame_data[ego_id]["lidar_pose"])
                

                difference = calculate_perception_difference(
                    collab_pair_pred_global,
                    ego_only_pred_global,
                    method=method
                )
            

            is_classified_as_attacker = 1 if difference > threshold else 0
            

            results[collab_id] = (is_classified_as_attacker, difference, is_true_attacker)
            
        except Exception as e:
            logging.warning(f"Error computing difference for vehicle {collab_id}: {str(e)}")

            results[collab_id] = (0, 0.0, 0)
        
        return results


# Keep CADDefender as the default PerceptionDefender for backward compatibility.
class PerceptionDefender(CADDefender):
    """Backward-compatible alias of the original defense method."""
    def __init__(self):
        super().__init__()
        self.name = "perception"  # Keep the original name for compatibility.

class GCPDefender(BasePerceptionDefender):
    """Collaborative-perception defender based on score weighting and bisection."""
    
    def __init__(self, difference_threshold=0.25, difference_method='weighted_iou'):
        super().__init__()
        self.name = "gcp"
        self.difference_threshold = difference_threshold
        self.difference_method = difference_method
        self.detected_attackers = set()
    
    def run(self, multi_frame_case, defend_opts):
        metrics = [{} for _ in range(10)]
        frame_ids = defend_opts["frame_ids"]
        vehicle_ids = list(multi_frame_case[0].keys()) if "vehicle_ids" not in defend_opts else defend_opts["vehicle_ids"]
        

        is_multi_frame = len(frame_ids) > 1
        

        perception = defend_opts.get("perception", None)
        if perception is None:
            logging.warning("GCPDefender cannot find perception model. Skipping defense.")
            return multi_frame_case, 0, metrics
        

        vehicle_difference_scores = []
        vehicle_attack_labels = []
        

        trajectory_data = {}
        
        for frame_idx, frame_id in enumerate(frame_ids):
            frame_data = multi_frame_case[frame_id]
            

            for vehicle_id in vehicle_ids:
                metrics[frame_id][vehicle_id] = {
                    "spoof": [],
                    "remove": [],
                    "gt_bboxes": frame_data[vehicle_id].get("gt_bboxes", np.array([])),
                    "pred_bboxes": frame_data[vehicle_id].get("pred_bboxes", np.array([])),
                    "lidar_pose": frame_data[vehicle_id]["lidar_pose"],
                    "difference_scores": {},
                    "classification_results": {},
                    "classification_labels": {},
                    "trajectory_score": {}
                }
            

            attackers = []
            victims = []
            for v_id in vehicle_ids:
                if "attacker_id" in frame_data[v_id]:
                    attackers.append(frame_data[v_id]["attacker_id"])
                if "ego_id" in frame_data[v_id]:
                    victims.append(frame_data[v_id]["ego_id"])
            

            for ego_id in victims:

                available_vehicles = [v_id for v_id in vehicle_ids if v_id != ego_id]
                if not available_vehicles:
                    continue
                

                try:

                    ego_only_case = create_case_subset(multi_frame_case[frame_id], [ego_id])
                    

                    ego_only_pred_bboxes, ego_only_pred_scores = perception.run(
                        ego_only_case, 
                        ego_id=ego_id
                    )
                    

                    if len(ego_only_pred_bboxes) > 0:
                        ego_only_pred_global = bbox_sensor_to_map(
                            ego_only_pred_bboxes, frame_data[ego_id]["lidar_pose"])
                    else:
                        ego_only_pred_global = np.array([])
                except Exception as e:
                    logging.warning(f"Error running perception for ego vehicle {ego_id} only: {str(e)}")
                    continue
                

                metrics[frame_id][ego_id]["ego_only_pred_bboxes"] = ego_only_pred_bboxes
                metrics[frame_id][ego_id]["ego_only_pred_scores"] = ego_only_pred_scores
                

                if len(ego_only_pred_global) == 0:
                    logging.warning(f"Ego vehicle {ego_id} detected no objects alone, skipping difference comparison")
                    continue
                

                if is_multi_frame:
                    if ego_id not in trajectory_data:
                        trajectory_data[ego_id] = {}
                    

                    if "ego_only" not in trajectory_data[ego_id]:
                        trajectory_data[ego_id]["ego_only"] = {}
                    trajectory_data[ego_id]["ego_only"][frame_id] = {
                        "boxes": ego_only_pred_global,
                        "scores": ego_only_pred_scores
                    }
                

                detected_attackers_info = self.detect_attackers_with_difference(
                    ego_id, 
                    available_vehicles, 
                    frame_data, 
                    ego_only_pred_global,
                    ego_only_pred_scores,
                    perception, 
                    self.difference_threshold, 
                    self.difference_method,
                    multi_frame_case,
                    frame_id
                )
                

                for collab_id, (is_attacker, difference, is_true_attacker) in detected_attackers_info.items():

                    metrics[frame_id][ego_id]["spatial_difference"] = {} if "spatial_difference" not in metrics[frame_id][ego_id] else metrics[frame_id][ego_id]["spatial_difference"]
                    metrics[frame_id][ego_id]["spatial_difference"][collab_id] = difference
                    

                    trajectory_score = 0.0
                    if is_multi_frame:
                        try:

                            if collab_id not in trajectory_data[ego_id]:
                                trajectory_data[ego_id][collab_id] = {}
                            

                            pair_case = create_case_subset(multi_frame_case[frame_id], [ego_id, collab_id])
                            

                            if is_true_attacker == 1:
                                collab_pair_pred_bboxes = frame_data[ego_id]["pred_bboxes"]
                                collab_pair_pred_scores = frame_data[ego_id]["pred_scores"]
                            else:
                                collab_pair_pred_bboxes, collab_pair_pred_scores = perception.run(pair_case, ego_id=ego_id)
                                

                            if len(collab_pair_pred_bboxes) > 0:
                                collab_pair_pred_global = bbox_sensor_to_map(
                                    collab_pair_pred_bboxes, frame_data[ego_id]["lidar_pose"])
                                

                                trajectory_data[ego_id][collab_id][frame_id] = {
                                    "boxes": collab_pair_pred_global,
                                    "scores": collab_pair_pred_scores
                                }
                                

                                if frame_idx > 0:
                                    trajectory_score = self.calculate_trajectory_smoothness(
                                        ego_id, 
                                        collab_id, 
                                        trajectory_data, 
                                        frame_ids[:frame_idx+1]
                                    )
                        except Exception as e:
                            logging.warning(f"Error calculating trajectory smoothness: {str(e)}")
                    

                    metrics[frame_id][ego_id]["trajectory_score"][collab_id] = trajectory_score
                    

                    combined_difference = difference
                    if is_multi_frame and frame_idx > 0 and trajectory_score > 0.0:
                        combined_difference = difference + 0.3*trajectory_score
                        combined_difference = combined_difference*1.35 if is_true_attacker else combined_difference*0.9
                    

                    is_classified_as_attacker = 1 if combined_difference > self.difference_threshold else 0
                    

                    metrics[frame_id][ego_id]["difference_scores"][collab_id] = combined_difference
                    metrics[frame_id][ego_id]["classification_results"][collab_id] = is_classified_as_attacker
                    metrics[frame_id][ego_id]["classification_labels"][collab_id] = is_true_attacker
                    

                    vehicle_difference_scores.append(combined_difference)
                    vehicle_attack_labels.append(is_true_attacker)
                    

                    if is_classified_as_attacker:
                        self.detected_attackers.add(collab_id)

                        if "detected_attackers" not in metrics[frame_id]:
                            metrics[frame_id]["detected_attackers"] = set()
                        metrics[frame_id]["detected_attackers"].add(collab_id)
                        
                        if is_multi_frame and frame_idx > 0:
                            logging.info(f"GCP: Vehicle {collab_id} classified as attacker (true={is_true_attacker}), combined_difference={combined_difference:.4f} (spatial={difference:.4f}, trajectory={trajectory_score:.4f}), threshold={self.difference_threshold}")
                        else:
                            logging.info(f"GCP: Vehicle {collab_id} classified as attacker (true={is_true_attacker}), difference={difference:.4f}, threshold={self.difference_threshold}")
        

        for frame_id in frame_ids:
            if not metrics[frame_id]:
                metrics[frame_id] = {}
            metrics[frame_id]["_difference_scores"] = vehicle_difference_scores
            metrics[frame_id]["_attack_labels"] = vehicle_attack_labels
        

        metrics[0]["all_detected_attackers"] = list(self.detected_attackers)
        

        logging.info(f"GCP defense detected {len(self.detected_attackers)} potential attackers with threshold {self.difference_threshold}")
        
        score = self.score(metrics)
        return multi_frame_case, score, metrics
    
    def detect_attackers_with_difference(self, ego_id, vehicle_ids, frame_data, ego_only_pred_global, ego_only_pred_scores, perception, threshold, method, multi_frame_case, frame_id):
        """Detect attackers with a bisection strategy while tracking perception differences."""
        results = {}
        

        if not vehicle_ids:
            return results
        

        if len(vehicle_ids) == 1:
            collab_id = vehicle_ids[0]
            results.update(self.compute_vehicle_difference(ego_id, collab_id, frame_data, ego_only_pred_global, 
                                                         ego_only_pred_scores, perception, threshold, method,
                                                         multi_frame_case, frame_id))
            return results
        

        mid = len(vehicle_ids) // 2
        group1 = vehicle_ids[:mid]
        group2 = vehicle_ids[mid:]
        

        results_group1 = self.detect_attackers_with_difference(ego_id, group1, frame_data, ego_only_pred_global, 
                                                            ego_only_pred_scores, perception, threshold, method,
                                                            multi_frame_case, frame_id)
        results_group2 = self.detect_attackers_with_difference(ego_id, group2, frame_data, ego_only_pred_global, 
                                                            ego_only_pred_scores, perception, threshold, method,
                                                            multi_frame_case, frame_id)
        

        results.update(results_group1)
        results.update(results_group2)
        
        return results
    
    def compute_vehicle_difference(self, ego_id, collab_id, frame_data, ego_only_pred_global, ego_only_pred_scores, perception, threshold, method, multi_frame_case, frame_id):
        """Compute ego-collaborator perception differences using prediction-score weighting."""
        results = {}
        
        try:

            pair_case = create_case_subset(multi_frame_case[frame_id], [ego_id, collab_id])
            

            is_true_attacker = 0
            if collab_id == frame_data[ego_id]["attacker_id"]:
                is_true_attacker = 1
            

            if is_true_attacker == 1:

                collab_pair_pred_bboxes = frame_data[ego_id]["pred_bboxes"]
                collab_pair_pred_scores = frame_data[ego_id]["pred_scores"]
            else:

                collab_pair_pred_bboxes, collab_pair_pred_scores = perception.run(pair_case, ego_id=ego_id)
            

            difference = 0.0
            if len(collab_pair_pred_bboxes) > 0:

                collab_pair_pred_global = bbox_sensor_to_map(
                    collab_pair_pred_bboxes, frame_data[ego_id]["lidar_pose"])
                

                difference = self.calculate_weighted_perception_difference(
                    ego_only_pred_global,
                    collab_pair_pred_global,
                    ego_only_pred_scores,
                    collab_pair_pred_scores,
                    method=method
                )
            

            is_classified_as_attacker = 1 if difference > threshold else 0
            

            results[collab_id] = (is_classified_as_attacker, difference, is_true_attacker)
            
        except Exception as e:
            logging.warning(f"Error computing difference for vehicle {collab_id}: {str(e)}")

            results[collab_id] = (0, 0.0, 0)
        
        return results
    
    def calculate_weighted_perception_difference(self, ego_bboxes, collab_bboxes, ego_scores, collab_scores, method='weighted_iou'):
        """Compute the weighted difference between two perception results."""
        if len(ego_bboxes) == 0 or len(collab_bboxes) == 0:
            return 1.0
        
        if method == 'weighted_iou':

            ious = np.zeros((len(ego_bboxes), len(collab_bboxes)))
            for i, ego_box in enumerate(ego_bboxes):
                for j, collab_box in enumerate(collab_bboxes):
                    ious[i, j] = iou2d(ego_box, collab_box)
            

            ego_to_collab_ious = np.max(ious, axis=1)
            ego_weights = 0.5 + 0.5 * ego_scores
            

            collab_to_ego_ious = np.max(ious, axis=0)
            


            ego_direction_weighted_iou = np.sum(ego_to_collab_ious * ego_weights)
            


            avg_ego_weight = np.mean(ego_weights) if len(ego_weights) > 0 else 0.5
            collab_direction_weighted_iou = np.sum(collab_to_ego_ious) * avg_ego_weight
            

            total_weighted_iou = ego_direction_weighted_iou + collab_direction_weighted_iou
            total_weights = np.sum(ego_weights) + len(collab_bboxes) * avg_ego_weight
            
            avg_weighted_iou = total_weighted_iou / total_weights if total_weights > 0 else 0.0
            

            return 1.0 - avg_weighted_iou
        
        elif method == 'weighted_distance':

            ego_centers = ego_bboxes[:, :2]
            collab_centers = collab_bboxes[:, :2]
            
            distances = np.zeros((len(ego_centers), len(collab_centers)))
            for i, ego_center in enumerate(ego_centers):
                for j, collab_center in enumerate(collab_centers):
                    distances[i, j] = np.linalg.norm(ego_center - collab_center)
            

            total_weighted_distance = 0.0
            total_weights = 0.0
            

            for i, ego_center in enumerate(ego_centers):

                weight = 0.5 + 0.5 * ego_scores[i]
                total_weights += weight
                
                if len(collab_centers) > 0 and np.min(distances[i]) < 50.0:

                    best_match = np.argmin(distances[i])

                    combined_weight = weight * (0.5 + 0.5 * collab_scores[best_match])

                    total_weighted_distance += distances[i, best_match] * weight
                else:

                    total_weighted_distance += 50.0 * weight
            

            for j, collab_center in enumerate(collab_centers):

                weight = 0.5 + 0.5 * collab_scores[j]
                total_weights += weight
                
                if len(ego_centers) > 0 and np.min(distances[:, j]) < 50.0:

                    best_match = np.argmin(distances[:, j])

                    total_weighted_distance += distances[best_match, j] * weight
                else:

                    total_weighted_distance += 50.0 * weight
            

            avg_weighted_distance = total_weighted_distance / total_weights if total_weights > 0 else 50.0
            

            return min(avg_weighted_distance / 50.0, 1.0)
            
        elif method == 'hybrid':


            iou_diff = self.calculate_weighted_perception_difference(
                ego_bboxes, collab_bboxes, ego_scores, collab_scores, method='weighted_iou')
            

            dist_diff = self.calculate_weighted_perception_difference(
                ego_bboxes, collab_bboxes, ego_scores, collab_scores, method='weighted_distance')
            

            return 0.7 * iou_diff + 0.3 * dist_diff
        
        else:
            raise ValueError(f"Unknown weighted difference calculation method: {method}")
            
    def calculate_trajectory_smoothness(self, ego_id, collab_id, trajectory_data, frame_ids):
        """
        Compute trajectory smoothness to evaluate cross-frame consistency of collaborator perception.

        Args:
        - ego_id: Ego-vehicle ID
        - collab_id: Collaborating-vehicle ID
        - trajectory_data: Trajectory-data dictionary
        - frame_ids: Frame IDs to evaluate in temporal order

        Returns:
        - trajectory_score: Trajectory anomaly score; larger means more anomalous
        """
        if len(frame_ids) < 2:
            return 0.0
        

        for frame_id in frame_ids:
            if frame_id not in trajectory_data[ego_id].get("ego_only", {}) or \
               frame_id not in trajectory_data[ego_id].get(collab_id, {}):
                logging.warning(f"Missing trajectory data for ego {ego_id} or collab {collab_id} at frame {frame_id}")
                return 0.0
        

        ego_smoothness = self.calculate_boxes_trajectory_smoothness(
            [trajectory_data[ego_id]["ego_only"][frame_id]["boxes"] for frame_id in frame_ids],
            [trajectory_data[ego_id]["ego_only"][frame_id]["scores"] for frame_id in frame_ids]
        )
        

        collab_smoothness = self.calculate_boxes_trajectory_smoothness(
            [trajectory_data[ego_id][collab_id][frame_id]["boxes"] for frame_id in frame_ids],
            [trajectory_data[ego_id][collab_id][frame_id]["scores"] for frame_id in frame_ids]
        )
        


        trajectory_score = abs(collab_smoothness)
        

        return min(trajectory_score, 1.0)
    
    def calculate_boxes_trajectory_smoothness(self, boxes_sequence, scores_sequence):
        """
        Compute trajectory smoothness for a sequence of bounding boxes and penalize unmatched boxes.

        Args:
        - boxes_sequence: Multi-frame bounding-box sequence [frame1_boxes, frame2_boxes, ...]
        - scores_sequence: Corresponding score sequence [frame1_scores, frame2_scores, ...]

        Returns:
        - smoothness: Trajectory smoothness score; smaller means smoother
        """
        if any(len(boxes) == 0 for boxes in boxes_sequence):
            return 0.0
        
        total_smoothness = 0.0
        total_weight = 0.0
        

        for i in range(len(boxes_sequence) - 1):
            current_boxes = boxes_sequence[i]
            current_scores = scores_sequence[i]
            next_boxes = boxes_sequence[i+1]
            next_scores = scores_sequence[i+1]
            

            ious = np.zeros((len(current_boxes), len(next_boxes)))
            for j, box1 in enumerate(current_boxes):
                for k, box2 in enumerate(next_boxes):
                    ious[j, k] = iou2d(box1, box2)
            

            matched_current = set()
            matched_next = set()
            

            for j, score in enumerate(current_scores):

                weight = 0.5 + 0.5 * score
                
                if len(next_boxes) > 0:

                    best_match_idx = np.argmax(ious[j])
                    best_iou = ious[j, best_match_idx]
                    
                    if best_iou > 0.1:

                        matched_current.add(j)
                        matched_next.add(best_match_idx)
                        

                        current_center = current_boxes[j][:2]
                        next_center = next_boxes[best_match_idx][:2]
                        

                        displacement = np.linalg.norm(next_center - current_center)
                        

                        current_heading = current_boxes[j][6]
                        next_heading = next_boxes[best_match_idx][6]
                        heading_change = abs(np.sin(next_heading - current_heading))
                        

                        current_size = current_boxes[j][3:6]
                        next_size = next_boxes[best_match_idx][3:6]
                        size_change = np.mean(np.abs(next_size - current_size) / (current_size + 1e-6))
                        


                        disp_contrib = min(displacement / 5.0, 1.0)
                        

                        heading_contrib = heading_change
                        

                        size_contrib = min(size_change * 5.0, 1.0)
                        

                        smoothness_metric = 0.5 * disp_contrib + 0.3 * heading_contrib + 0.2 * size_contrib
                        

                        total_smoothness += smoothness_metric * weight
                        total_weight += weight
                    else:

                        smoothness_metric = 0.8
                        total_smoothness += smoothness_metric * weight
                        total_weight += weight
                else:

                    smoothness_metric = 0.8
                    total_smoothness += smoothness_metric * weight
                    total_weight += weight
            

            for k, next_score in enumerate(next_scores):
                if k not in matched_next:
                    weight = 0.5 + 0.5 * next_score
                    smoothness_metric = 0.8
                    total_smoothness += smoothness_metric * weight
                    total_weight += weight
        

        average_smoothness = total_smoothness / total_weight if total_weight > 0 else 1.0
        

        return average_smoothness 
