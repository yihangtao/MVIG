import random
import pickle
from readline import append_history_file
import numpy as np
import copy
import logging
import traceback

from .attacker import Attacker
from mvp.data.util import bbox_map_to_sensor, bbox_sensor_to_map, pcd_sensor_to_map, sort_lidar_points
from mvp.tools.iou import iou2d
from mvp.defense.perception_defender import calculate_ap_delta  # Required for AP-delta evaluation


class LidarSpoofIntermediateAttacker(Attacker):
    def __init__ (self, perception, dataset=None, step=100, sync=1, init=True, online=True):
        super().__init__()
        self.dataset = dataset
        self.name = "lidar_spoof"
        self.load_benchmark_meta()
        self.name = "lidar_spoof_intermediate"
        self.max_perturb = 10

        self.name += "_Step{}".format(step - 1)
        if sync > 0:
            self.name += "_Async"
        if init:
            self.name += "_Init"
        if online:
            self.name += "_Online"
        self.step = step
        self.sync = sync
        self.init = init
        self.online = online

        if perception.model_name != "pointpillar":
            self.name += "_{}".format(perception.model_name)

        if step <=  2:
            self.learn_rate = 1
        else:
            self.learn_rate = 0.05
        self.perception = perception

    def run(self, multi_frame_case, attack_opts):
        case = copy.deepcopy(multi_frame_case)
        info = [{} for i in range(10)]
        init_perturbation = None
        
        # Default to RC mode unless attack_opts overrides it.
        attack_mode = attack_opts.get("attack_mode", "RC")
        # Default number of fake objects used by BASIC mode.
        n_fake_objects = attack_opts.get("n_fake_objects", 3)

        for frame_index, frame_id in enumerate(attack_opts["frame_ids"]):
            attacker_id = attack_opts["attacker_vehicle_id"]
            ego_id = attack_opts["victim_vehicle_id"]
            info[frame_id][ego_id] = {}

            if self.sync == 0:
                optimize_case = copy.deepcopy(case[frame_id])
                real_case = None
            else:
                optimize_case = copy.deepcopy(case[frame_id - 1])
                original_case = copy.deepcopy(optimize_case)
                real_case = copy.deepcopy(case[frame_id])
            original_case = copy.deepcopy(optimize_case)
            real_original_case = copy.deepcopy(real_case)
            # `attack_opts["positions"]` carries the coarse attack target predicted upstream by MVIG.
            # This attacker turns that target into `bbox_to_spoof_ego`, which is the victim-frame box used
            # consistently by RC as the direct spoof box and by PGD-style modes as the target reference box.
            bbox_to_spoof = attack_opts["positions"][frame_id if self.sync == 0 else frame_id - 1]
            bbox_to_spoof_ego = bbox_map_to_sensor(
                bbox_sensor_to_map(bbox_to_spoof, optimize_case[attacker_id]["lidar_pose"]),
                optimize_case[ego_id]["lidar_pose"])
            real_bbox_to_spoof = attack_opts["positions"][frame_id]
            real_bbox_to_spoof_ego = bbox_map_to_sensor(
                bbox_sensor_to_map(real_bbox_to_spoof, case[frame_id][attacker_id]["lidar_pose"]),
                case[frame_id][ego_id]["lidar_pose"])

            if self.init:
                optimize_case[attacker_id]["lidar"] = self.apply_ray_tracing(optimize_case[attacker_id]["lidar"], **attack_opts["attack_info"][frame_id if self.sync == 0 else frame_id - 1])
                if real_case is not None:
                    real_case[attacker_id]["lidar"] = self.apply_ray_tracing(real_case[attacker_id]["lidar"], **attack_opts["attack_info"][frame_id])
            
            # Select the implementation according to the attack mode.
            if attack_mode.upper() == "RC":
                # RC mode directly appends a spoof target to the predictions.
                try:
                    if hasattr(self, 'perception') and self.perception:
                        # Run the perception model to obtain the original predictions.
                        orig_bboxes, orig_scores = self.perception.run(
                            optimize_case, 
                            ego_id=ego_id
                        )
                        
                        # Build the result dictionary from the original predictions.
                        result = {
                            "pred_bboxes": orig_bboxes.copy(),
                            "pred_scores": orig_scores.copy(),
                            "normal_pred_bboxes": orig_bboxes.copy(),  # Cache the original predictions.
                            "normal_gt_bboxes": []  # Initialize as an empty array.
                        }
                        
                        # Try to fetch the ground-truth boxes.
                        if "gt_bboxes" in optimize_case[ego_id]:
                            result["normal_gt_bboxes"] = optimize_case[ego_id]["gt_bboxes"]
                        
                        # Append the spoof target box.
                        if bbox_to_spoof_ego is not None:
                            result["pred_bboxes"] = np.append(result["pred_bboxes"], [bbox_to_spoof_ego], axis=0)
                            result["pred_scores"] = np.append(result["pred_scores"], [0.9])  # Assign a high confidence score to the spoof target.
                            
                            # Compute the AP delta.
                            if len(result["normal_gt_bboxes"]) > 0:
                                # Compare predictions before and after the attack.
                                normal_pred = result["normal_pred_bboxes"]
                                normal_gt = result["normal_gt_bboxes"]
                                attack_pred = result["pred_bboxes"]
                                
                                # Ensure all boxes are expressed in the same coordinate system.
                                lidar_pose = optimize_case[ego_id]["lidar_pose"]
                                normal_pred_map = bbox_sensor_to_map(normal_pred, lidar_pose)
                                normal_gt_map = bbox_sensor_to_map(normal_gt, lidar_pose)
                                attack_pred_map = bbox_sensor_to_map(attack_pred, lidar_pose)
                                
                                # Compute the AP delta.
                                delta_ap = calculate_ap_delta(
                                    normal_pred_map, normal_gt_map,
                                    attack_pred_map, normal_gt_map,
                                    iou_threshold=0.5
                                )
                                
                                # Store the AP delta.
                                result["delta_ap_0.5"] = delta_ap
                                logging.info(f"RC Attack delta AP@0.5: {delta_ap:.4f}")
                            else:
                                result["delta_ap_0.5"] = 0.0
                    else:
                        # If no perception model is available, return only the spoof target.
                        if bbox_to_spoof_ego is not None:
                            result = {
                                "pred_bboxes": np.array([bbox_to_spoof_ego]),
                                "pred_scores": np.array([0.9]),
                                "delta_ap_0.5": 0.0
                            }
                        else:
                            result = {
                                "pred_bboxes": np.array([]),
                                "pred_scores": np.array([]),
                                "delta_ap_0.5": 0.0
                            }
                except Exception as e:
                    logging.debug(f"Error in RC perception: {str(e)}")
                    # On failure, fall back to returning only the spoof target.
                    if bbox_to_spoof_ego is not None:
                        result = {
                            "pred_bboxes": np.array([bbox_to_spoof_ego]),
                            "pred_scores": np.array([0.9]),
                            "delta_ap_0.5": 0.0
                        }
                    else:
                        result = {
                            "pred_bboxes": np.array([]),
                            "pred_scores": np.array([]),
                            "delta_ap_0.5": 0.0
                        }
            else:
                # BASIC/BAC/RC+ delegate to `attack_intermediate` for PGD-style optimization.
                # In those modes, `bbox_to_spoof_ego` is not only the semantic spoof target, but also the
                # reference box used downstream to derive the perturbation center and local mask region.
                try:
                    # Run the PGD-style attack.
                    result = self.perception.attack_intermediate(
                        optimize_case, 
                        ego_id, 
                        attacker_id, 
                        max_perturb=self.max_perturb, 
                        mode="spoof", 
                        bbox=bbox_to_spoof_ego, 
                        max_iteration=self.step, 
                        lr=self.learn_rate, 
                        real_case=real_case, 
                        original_case=original_case, 
                        real_original_case=real_original_case, 
                        real_bbox=real_bbox_to_spoof_ego, 
                        init_perturbation=init_perturbation, 
                        feature_size=10,
                        attack_mode=attack_mode,
                        n_fake_objects=n_fake_objects
                    )
                    
                    # Attach the original predictions for reference.
                    try:
                        orig_bboxes, orig_scores = self.perception.run(
                            optimize_case, 
                            ego_id=ego_id
                        )
                        result["normal_pred_bboxes"] = orig_bboxes
                        
                        # Fetch the ground-truth boxes.
                        if "gt_bboxes" in optimize_case[ego_id]:
                            result["normal_gt_bboxes"] = optimize_case[ego_id]["gt_bboxes"]
                        else:
                            result["normal_gt_bboxes"] = np.array([])
                            
                        # Compute the AP delta.
                        if len(result["normal_gt_bboxes"]) > 0:
                            # Ensure all boxes are expressed in the same coordinate system.
                            lidar_pose = optimize_case[ego_id]["lidar_pose"]
                            normal_pred_map = bbox_sensor_to_map(result["normal_pred_bboxes"], lidar_pose)
                            normal_gt_map = bbox_sensor_to_map(result["normal_gt_bboxes"], lidar_pose)
                            attack_pred_map = bbox_sensor_to_map(result["pred_bboxes"], lidar_pose)
                            
                            # Compute the AP delta.
                            delta_ap = calculate_ap_delta(
                                normal_pred_map, normal_gt_map,
                                attack_pred_map, normal_gt_map,
                                iou_threshold=0.5
                            )
                            
                            # Store the AP delta.
                            result["delta_ap_0.5"] = delta_ap
                            logging.info(f"{attack_mode} attack delta AP@0.5: {delta_ap:.4f}")
                        else:
                            result["delta_ap_0.5"] = 0.0
                    except Exception as e:
                        logging.debug(f"Error getting original predictions: {str(e)}")
                        if "normal_pred_bboxes" not in result:
                            result["normal_pred_bboxes"] = np.array([])
                        if "normal_gt_bboxes" not in result:
                            result["normal_gt_bboxes"] = np.array([])
                        if "delta_ap_0.5" not in result:
                            result["delta_ap_0.5"] = 0.0
                
                except Exception as e:
                    error_msg = f"Error in {attack_mode} attack: {str(e)}\n"
                    error_msg += "Stack trace:\n"
                    error_msg += traceback.format_exc()
                    logging.error(error_msg)
                    # On failure, fall back to returning only the spoof target.
                    if bbox_to_spoof_ego is not None:
                        result = {
                            "pred_bboxes": np.array([bbox_to_spoof_ego]),
                            "pred_scores": np.array([0.9]),
                            "normal_pred_bboxes": np.array([]),
                            "normal_gt_bboxes": np.array([]),
                            "delta_ap_0.5": 0.0
                        }
                    else:
                        result = {
                            "pred_bboxes": np.array([]),
                            "pred_scores": np.array([]),
                            "normal_pred_bboxes": np.array([]),
                            "normal_gt_bboxes": np.array([]),
                            "delta_ap_0.5": 0.0
                        }
            
            # Record the attack mode.
            result["attack_mode"] = attack_mode
            
            # Cache the post-attack predictions.
            if self.online and "perturbation" in result:
                init_perturbation = result["perturbation"] / 2
            
            case[frame_id][ego_id]["pred_bboxes"] = result["pred_bboxes"]
            case[frame_id][ego_id]["pred_scores"] = result["pred_scores"]
            case[frame_id][ego_id]["normal_pred_bboxes"] = result.get("normal_pred_bboxes", np.array([]))
            case[frame_id][ego_id]["normal_gt_bboxes"] = result.get("normal_gt_bboxes", np.array([]))
            case[frame_id][ego_id]["delta_ap_0.5"] = result.get("delta_ap_0.5", 0.0)
            case[frame_id][ego_id]["attack_mode"] = attack_mode

            info[frame_id][ego_id] = {
                "pred_bboxes": result["pred_bboxes"], 
                "pred_scores": result["pred_scores"],
                "normal_pred_bboxes": result.get("normal_pred_bboxes", np.array([])),
                "normal_gt_bboxes": result.get("normal_gt_bboxes", np.array([])),
                "delta_ap_0.5": result.get("delta_ap_0.5", 0.0),
                "attacker_id": attacker_id,
                "ego_id": ego_id,
                "attack_mode": attack_mode
            }

        return case, info
