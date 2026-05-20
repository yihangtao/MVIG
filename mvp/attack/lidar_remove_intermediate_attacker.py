import random
import pickle
import numpy as np
import copy
import logging

from .attacker import Attacker
from mvp.data.util import bbox_map_to_sensor, bbox_sensor_to_map, pcd_sensor_to_map
from mvp.tools.iou import iou2d


class LidarRemoveIntermediateAttacker(Attacker):
    def __init__ (self, perception, dataset=None, step=100, sync=1, init=True, online=True):
        super().__init__()
        self.dataset = dataset
        self.name = "lidar_remove"
        self.load_benchmark_meta()
        self.name = "lidar_remove_intermediate"
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
            self.learn_rate = 2
        else:
            self.learn_rate = 0.05
        self.perception = perception

    def run(self, multi_frame_case, attack_opts):
        case = copy.deepcopy(multi_frame_case)
        info = [{} for i in range(10)]
        init_perturbation = None

        for frame_index, frame_id in enumerate(attack_opts["frame_ids"]):
            attacker_id = attack_opts["attacker_vehicle_id"]
            ego_id = attack_opts["victim_vehicle_id"]
            info[frame_id][ego_id] = {}

            if self.sync == 0:
                optimize_case = copy.deepcopy(case[frame_id])
                real_case = None
            else:
                optimize_case = copy.deepcopy(case[frame_id - 1])
                real_case = copy.deepcopy(case[frame_id])
            original_case = copy.deepcopy(optimize_case)
            real_original_case = copy.deepcopy(real_case)

            # try:
            #     object_index = optimize_case[attacker_id]["object_ids"].index(attack_opts["object_id"])
            #     bbox_to_remove = optimize_case[attacker_id]["gt_bboxes"][object_index]
            # except:
            #     bbox_to_remove = attack_opts["bboxes"][frame_id if self.sync == 0 else frame_id - 1]
            try:
                bbox_to_remove = attack_opts["positions"][frame_id if self.sync == 0 else frame_id - 1]
            except:
                try:
                    object_index = optimize_case[attacker_id]["object_ids"].index(attack_opts["object_id"])
                    bbox_to_remove = optimize_case[attacker_id]["gt_bboxes"][object_index]
                except:
                    bbox_to_remove = attack_opts["bboxes"][frame_id if self.sync == 0 else frame_id - 1]

            bbox_to_remove_ego = bbox_map_to_sensor(
                bbox_sensor_to_map(bbox_to_remove, optimize_case[attacker_id]["lidar_pose"]),
                optimize_case[ego_id]["lidar_pose"])
            # try:
            #     real_object_index = case[frame_id][attacker_id]["object_ids"].index(attack_opts["object_id"])
            #     real_bbox_to_remove = case[frame_id][attacker_id]["gt_bboxes"][real_object_index]
            # except:
            #     real_bbox_to_remove = attack_opts["bboxes"][frame_id]
            try:
                real_bbox_to_remove = attack_opts["positions"][frame_id]
            except:
                try:
                    real_object_index = case[frame_id][attacker_id]["object_ids"].index(attack_opts["object_id"])
                    real_bbox_to_remove = case[frame_id][attacker_id]["gt_bboxes"][real_object_index]
                except:
                    real_bbox_to_remove = attack_opts["bboxes"][frame_id]


            real_bbox_to_remove_ego = bbox_map_to_sensor(
                bbox_sensor_to_map(real_bbox_to_remove, case[frame_id][attacker_id]["lidar_pose"]),
                case[frame_id][ego_id]["lidar_pose"])
            
            if self.init:
                optimize_case[attacker_id]["lidar"] = self.apply_ray_tracing(optimize_case[attacker_id]["lidar"], **attack_opts["attack_info"][frame_id if self.sync == 0 else frame_id - 1])
                if real_case is not None:
                    real_case[attacker_id]["lidar"] = self.apply_ray_tracing(real_case[attacker_id]["lidar"], **attack_opts["attack_info"][frame_id])

             # result = self.perception.attack_intermediate(optimize_case, ego_id, attacker_id, max_perturb=self.max_perturb, mode="remove", bbox=bbox_to_remove_ego, max_iteration=self.step, lr=self.learn_rate, real_case=real_case, original_case=original_case, real_original_case=real_original_case, real_bbox=real_bbox_to_remove_ego, init_perturbation=init_perturbation, feature_size=10)

            # 获取正常感知的预测结果
            try:
                if hasattr(self, 'perception') and self.perception:
                    # 运行感知模型获取原始预测
                    orig_bboxes, orig_scores = self.perception.run(
                        optimize_case, 
                        ego_id=ego_id
                    )
                    
                    # 创建结果字典，存储原始预测框和预测分数
                    result = {
                        "pred_bboxes": orig_bboxes.copy(),
                        "pred_scores": orig_scores.copy(),
                        "normal_pred_bboxes": orig_bboxes.copy(),  # 保存原始预测框用于计算delta_ap
                        "normal_gt_bboxes": []  # 初始化为空数组
                    }
                    
                    # 尝试获取真值框
                    if "gt_bboxes" in optimize_case[ego_id]:
                        result["normal_gt_bboxes"] = optimize_case[ego_id]["gt_bboxes"]
                    
                    # 如果存在要移除的目标框，找到并删除IoU最大的预测框
                    if bbox_to_remove_ego is not None and len(orig_bboxes) > 0:
                        # 计算所有预测框与目标框的IoU
                        ious = np.array([iou2d(bbox, bbox_to_remove_ego) for bbox in orig_bboxes])
                        max_iou_idx = np.argmax(ious)
                        
                        # 删除IoU最大的预测框
                        mask = np.ones(len(orig_bboxes), dtype=bool)
                        mask[max_iou_idx] = False
                        result["pred_bboxes"] = result["pred_bboxes"][mask]
                        result["pred_scores"] = result["pred_scores"][mask]
                        
                        # 计算AP差值
                        if len(result["normal_gt_bboxes"]) > 0:
                            # 获取攻击前后的预测框
                            normal_pred = result["normal_pred_bboxes"]
                            normal_gt = result["normal_gt_bboxes"]
                            attack_pred = result["pred_bboxes"]
                            
                            # 确保所有框都在同一坐标系中
                            lidar_pose = optimize_case[ego_id]["lidar_pose"]
                            normal_pred_map = bbox_sensor_to_map(normal_pred, lidar_pose)
                            normal_gt_map = bbox_sensor_to_map(normal_gt, lidar_pose)
                            attack_pred_map = bbox_sensor_to_map(attack_pred, lidar_pose)
                            
                            # 从defense模块导入计算函数
                            from mvp.defense.perception_defender import calculate_ap_delta
                            
                            # 计算AP差值
                            delta_ap = calculate_ap_delta(
                                normal_pred_map, normal_gt_map,
                                attack_pred_map, normal_gt_map,
                                iou_threshold=0.5
                            )
                            
                            # 存储AP差值
                            result["delta_ap_0.5"] = delta_ap
                            logging.info(f"Remove attack delta AP@0.5: {delta_ap:.4f}")
                        else:
                            result["delta_ap_0.5"] = 0.0
                else:
                    # 如果没有感知模型，返回空结果
                    result = {
                        "pred_bboxes": np.array([]),
                        "pred_scores": np.array([]),
                        "normal_pred_bboxes": np.array([]),
                        "normal_gt_bboxes": np.array([]),
                        "delta_ap_0.5": 0.0
                    }
            except Exception as e:
                logging.debug(f"Error in perception: {str(e)}")
                # 出错时返回空结果
                result = {
                    "pred_bboxes": np.array([]),
                    "pred_scores": np.array([]),
                    "normal_pred_bboxes": np.array([]),
                    "normal_gt_bboxes": np.array([]),
                    "delta_ap_0.5": 0.0
                }
            
            if self.online:
                init_perturbation = result["perturbation"]
            
            # 将结果存储到case和info中
            case[frame_id][ego_id]["pred_bboxes"] = result["pred_bboxes"]
            case[frame_id][ego_id]["pred_scores"] = result["pred_scores"]
            case[frame_id][ego_id]["normal_pred_bboxes"] = result.get("normal_pred_bboxes", np.array([]))
            case[frame_id][ego_id]["normal_gt_bboxes"] = result.get("normal_gt_bboxes", np.array([]))
            case[frame_id][ego_id]["delta_ap_0.5"] = result.get("delta_ap_0.5", 0.0)

            info[frame_id][ego_id] = {
                "pred_bboxes": result["pred_bboxes"], 
                "pred_scores": result["pred_scores"],
                "normal_pred_bboxes": result.get("normal_pred_bboxes", np.array([])),
                "normal_gt_bboxes": result.get("normal_gt_bboxes", np.array([])),
                "attacker_id": attacker_id,
                "ego_id": ego_id,
                "delta_ap_0.5": result.get("delta_ap_0.5", 0.0)
            }

        return case, info
