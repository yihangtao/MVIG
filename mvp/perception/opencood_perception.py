from concurrent.futures import process
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Must be set before importing torch
from mvp.config import third_party_root
opencood_root = os.path.join(third_party_root, "OpenCOOD")
sys.path.append(opencood_root)
import numpy as np
from collections import OrderedDict
import torch
import math
import copy
import random
import logging
import torch
import torch.nn.functional as F

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import box_utils
from opencood.utils.pcd_utils import mask_points_by_range
from opencood.utils.transformation_utils import x1_to_x2
from opencood.utils.common_utils import torch_tensor_to_numpy

from .perception import Perception
from mvp.config import model_root, data_root
from mvp.tools.iou import iou3d, iou2d
from mvp.evaluate.detection import iou3d_batch
from .iou_util import oriented_box_intersection_2d
from mvp.data.util import pcd_sensor_to_map, pcd_map_to_sensor, pose_to_transformation


class OpencoodPerception(Perception):
    def __init__(self, fusion_method="early", model_name="pointpillar", device_id=0):
        super().__init__()
        assert(model_name in ["pixor", "voxelnet", "second", "pointpillar", "v2vnet", "fpvrcnn"])
        assert(fusion_method in ["early", "intermediate", "late"])
        self.name = "{}_{}".format(model_name, fusion_method)
        self.devices = f"cuda:{device_id}"
        self.model_name = model_name
        self.fusion_method = fusion_method
        if self.model_name == "v2vnet":
            self.model_dir = os.path.join(model_root, "OpenCOOD/v2vnet")
            self.fusion_method = "intermediate"
        else:
            self.model_dir = os.path.join(model_root, "OpenCOOD/{}_{}_fusion".format(self.model_name, self.fusion_method if self.fusion_method != "intermediate" else "attentive"))
        self.config_file = os.path.join(self.model_dir, "config.yaml")
        self.preprocessors = {
            "early": self.early_preprocess,
            "intermediate": self.intermediate_preprocess,
            "late": self.late_preprocess,
        }
        self.inference_processors = {
            "early": inference_utils.inference_early_fusion,
            "intermediate": inference_utils.inference_intermediate_fusion,
            "late": inference_utils.inference_late_fusion,
        }

        hypes = yaml_utils.load_yaml(self.config_file, None)
        hypes["root_dir"] = os.path.join(data_root, "OPV2V/train")
        hypes["validate_dir"] = os.path.join(data_root, "OPV2V/validate")
        self.dataset = build_dataset(hypes, visualize=False, train=False)
        self.model = train_utils.create_model(hypes)
        # we assume gpu is available
        if torch.cuda.is_available():
            self.model.cuda(device_id)
        self.device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
        ret = train_utils.load_saved_model(self.model_dir, self.model)
        self.model = ret[1]
        self.model.eval()
        self.is_visualize = True

        # Added for visualization
        self.enable_visualization = True
        self.visualization_dir = os.path.join(model_root, "visualizations")
        os.makedirs(self.visualization_dir, exist_ok=True)

    def run(self, multi_vehicle_case, ego_id):
        batch = self.preprocessors[self.fusion_method](multi_vehicle_case, ego_id)
        batch_data = self.dataset.collate_batch_test([batch])
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, self.device)
            pred_box_tensor, pred_score, gt_box_tensor = \
                self.inference_processors[self.fusion_method](batch_data,
                                                              self.model,
                                                              self.dataset)
        pred_bboxes = pred_box_tensor.cpu().numpy()
        pred_bboxes = box_utils.corner_to_center(pred_bboxes, order="lwh")
        pred_bboxes[:,2] -= 0.5 * pred_bboxes[:,5]
        pred_scores = pred_score.cpu().numpy()
        return pred_bboxes, pred_scores
    
    def run_multi_vehicle(self, multi_vehicle_case, ego_id):
        pred_bboxes, pred_scores = self.run(multi_vehicle_case, ego_id)
        if pred_bboxes.shape[0] == 0:
            multi_vehicle_case[ego_id]["pred_bboxes"] = np.array([])
            multi_vehicle_case[ego_id]["pred_scores"] = np.array([])
        else:
            multi_vehicle_case[ego_id]["pred_bboxes"] = pred_bboxes
            multi_vehicle_case[ego_id]["pred_scores"] = pred_scores
        return multi_vehicle_case

    def attack_late(self, multi_vehicle_case, ego_id, attacker_id, bbox=None, mode="spoof"):
        batch = self.preprocessors[self.fusion_method](multi_vehicle_case, ego_id)
        batch_data = self.dataset.collate_batch_test([batch])
        if bbox is not None:
            bbox = np.copy(bbox)
            bbox[3:6] = bbox[[5,4,3]]
            bbox[2] += 0.5 * bbox[3]
            bbox = torch.from_numpy(bbox).type(torch.float32).to(self.device)

        with torch.no_grad():
            data_dict = train_utils.to_device(batch_data, self.device)
            output_dict = OrderedDict()
            for cav_id, cav_content in data_dict.items():
                output_dict[cav_id] = self.model(cav_content)

            # the final bounding box list
            pred_box3d_list = []
            pred_box2d_list = []

            for cav_id, cav_content in data_dict.items():
                transformation_matrix = cav_content['transformation_matrix']
                anchor_box = cav_content['anchor_box']
                prob = output_dict[cav_id]['psm']
                prob = F.sigmoid(prob.permute(0, 2, 3, 1))
                prob = prob.reshape(1, -1)
                reg = output_dict[cav_id]['rm']
                batch_box3d = self.dataset.post_processor.delta_to_boxes3d(reg, anchor_box)
                mask = \
                    torch.gt(prob, self.dataset.post_processor.params['target_args']['score_threshold'])
                mask = mask.view(1, -1)
                mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)

                boxes3d = torch.masked_select(batch_box3d[0],
                                            mask_reg[0]).view(-1, 7)
                scores = torch.masked_select(prob[0], mask[0])

                # convert output to bounding box
                if len(boxes3d) != 0:
                    if cav_id == attacker_id:
                        if mode == "spoof":
                            boxes3d = torch.vstack([boxes3d, torch.reshape(bbox, (1, 7))])
                            scores = torch.hstack([scores, torch.tensor([1.0]).type(scores.dtype).to(self.device)])
                        elif mode == "remove":
                            keep_index = torch.sum((boxes3d[:, :2] - bbox[:2]) ** 2, dim=1) > 4
                            boxes3d = boxes3d[keep_index]
                            scores = scores[keep_index]

                    # (N, 8, 3)
                    boxes3d_corner = \
                        box_utils.boxes_to_corners_3d(boxes3d,
                                                    order=self.dataset.post_processor.params['order'])
                    # (N, 8, 3)
                    projected_boxes3d = \
                        box_utils.project_box3d(boxes3d_corner,
                                                transformation_matrix)
                    # convert 3d bbx to 2d, (N,4)
                    projected_boxes2d = \
                        box_utils.corner_to_standup_box_torch(projected_boxes3d)
                    # (N, 5)
                    boxes2d_score = \
                        torch.cat((projected_boxes2d, scores.unsqueeze(1)), dim=1)

                    pred_box2d_list.append(boxes2d_score)
                    pred_box3d_list.append(projected_boxes3d)

            if len(pred_box2d_list) ==0 or len(pred_box3d_list) == 0:
                raise Exception("no detection result")
            # shape: (N, 5)
            pred_box2d_list = torch.vstack(pred_box2d_list)
            # scores
            scores = pred_box2d_list[:, -1]
            # predicted 3d bbx
            pred_box3d_tensor = torch.vstack(pred_box3d_list)
            # remove large bbx
            keep_index_1 = box_utils.remove_large_pred_bbx(pred_box3d_tensor)
            keep_index_2 = box_utils.remove_bbx_abnormal_z(pred_box3d_tensor)
            keep_index = torch.logical_and(keep_index_1, keep_index_2)
            pred_box3d_tensor = pred_box3d_tensor[keep_index]
            scores = scores[keep_index]

            # nms
            keep_index = box_utils.nms_rotated(pred_box3d_tensor,
                                            scores,
                                            self.dataset.post_processor.params['nms_thresh']
                                            )
            pred_box3d_tensor = pred_box3d_tensor[keep_index]

            # select cooresponding score
            scores = scores[keep_index]

            # filter out the prediction out of the range.
            mask = \
                box_utils.get_mask_for_boxes_within_range_torch(pred_box3d_tensor)
            pred_box3d_tensor = pred_box3d_tensor[mask, :, :]
            scores = scores[mask]
            assert scores.shape[0] == pred_box3d_tensor.shape[0]

        pred_box = pred_box3d_tensor.cpu().numpy()
        pred_box = box_utils.corner_to_center(pred_box, order="lwh")
        pred_box[:,2] -= 0.5 * pred_box[:,5]
        return {
            "pred_bboxes": pred_box,
            "pred_scores": scores.cpu().numpy()
        }

    def attack_intermediate_forward(self, batch_data, attacker_index, perturbation=None, feature=None, max_perturb=10, center=[0, 0, 0], feature_size=15, perturb_func=None):
        # Ensure center is a NumPy array instead of a Python list.
        if center is not None and isinstance(center, list):
            center = np.array(center)
        
        if perturbation is not None:
            clipped_perturbation = torch.clip(perturbation, min=-max_perturb, max=max_perturb)
        else:
            clipped_perturbation = None
        
        voxel_features = batch_data['ego']['processed_lidar']['voxel_features']
        voxel_coords = batch_data['ego']['processed_lidar']['voxel_coords']
        voxel_num_points = batch_data['ego']['processed_lidar']['voxel_num_points']
        record_len = batch_data['ego']['record_len']

        pairwise_t_matrix = batch_data['ego']['pairwise_t_matrix']

        batch_dict = {'voxel_features': voxel_features,
                    'voxel_coords': voxel_coords,
                    'voxel_num_points': voxel_num_points,
                    'record_len': record_len}

        if self.model_name == "v2vnet":
            batch_dict['voxel_features'] = batch_dict['voxel_features'].float()
        
        if self.model_name in ["pointpillar", "v2vnet"]:
            # n, 4 -> n, c
            self.model.pillar_vfe(batch_dict)
            # n, c -> N, C, H, W
            self.model.scatter(batch_dict)

            spatial_features = batch_dict['spatial_features']
        elif self.model_name == "voxelnet":
            if voxel_coords.is_cuda:
                record_len_tmp = record_len.cpu()

            record_len_tmp = list(record_len_tmp.numpy())

            self.model.N = sum(record_len_tmp)

            # feature learning network
            vwfs = self.model.svfe(batch_dict)['pillar_features']

            voxel_coords = torch_tensor_to_numpy(voxel_coords)
            vwfs = self.model.voxel_indexing(vwfs, voxel_coords)

            # convolutional middle network
            vwfs = self.model.cml(vwfs)
            # convert from 3d to 2d N C H W
            vmfs = vwfs.view(self.model.N, -1, self.model.H, self.model.W)

            # compression layer
            if self.model.compression:
                vmfs = self.model.compression_layer(vmfs)
            
            spatial_features = vmfs
        else:
            raise NotImplementedError()

        if perturb_func is not None:
            x = torch.clone(spatial_features).detach()
            spatial_features[attacker_index] = perturb_func(x[attacker_index].unsqueeze(0))[0]
        elif feature is not None:
            # Directly overwrite the feature tensor.
            if center is not None and feature_size is not None:
                spatial_features[attacker_index][:, center[1]-feature_size:center[1]+feature_size, center[0]-feature_size:center[0]+feature_size] = feature
            else:
                # Directly overwrite the entire feature map.
                spatial_features[attacker_index] = feature
            clipped_perturbation = None
        elif perturbation is not None:
            # Apply additive perturbation.
            x = torch.clone(spatial_features).detach()
            
            # Handle perturbations applied to the full feature map.
            if center is None or feature_size is None:
                # Add the perturbation to the entire feature map.
                spatial_features[attacker_index] = x[attacker_index] + clipped_perturbation
            else:
                # Ensure center is a NumPy array.
                if isinstance(center, list):
                    center = np.array(center)
                # Original local-region perturbation logic.
                aligned_center = center.astype(np.int32)
                C, H, W = spatial_features[attacker_index].size()
                perturbation_features = torch.zeros_like(spatial_features[attacker_index]).to(self.device)
                perturbation_features[:, aligned_center[1]-feature_size:aligned_center[1]+feature_size,
                                       aligned_center[0]-feature_size:aligned_center[0]+feature_size] = clipped_perturbation
                theta = torch.tensor([[[1, 0, (center[1] - aligned_center[1]) * 2 / W],
                                     [0, 1, (center[0] - aligned_center[0]) * 2 / H]]], dtype=torch.float).repeat(1, 1, 1).to(self.device)
                grid = torch.nn.functional.affine_grid(theta, (1, C, H, W))
                perturbation_features = torch.nn.functional.grid_sample(perturbation_features.unsqueeze(0), grid)[0]
                spatial_features[attacker_index] = x[attacker_index] + perturbation_features

        if self.model_name in ["pointpillar", "v2vnet"]:
            batch_dict["spatial_features"] = spatial_features
            self.model.backbone(batch_dict)
            spatial_features_2d = batch_dict['spatial_features_2d']

            if self.model_name == "v2vnet":
                # downsample feature to reduce memory
                if self.model.shrink_flag:
                    spatial_features_2d = self.model.shrink_conv(spatial_features_2d)
                # compressor
                if self.model.compression:
                    spatial_features_2d = self.model.naive_compressor(spatial_features_2d)
                
                fused_feature = self.model.fusion_net(spatial_features_2d,
                                                record_len,
                                                pairwise_t_matrix)
                psm = self.model.cls_head(fused_feature)
                rm = self.model.reg_head(fused_feature)
            else:
                psm = self.model.cls_head(spatial_features_2d)
                rm = self.model.reg_head(spatial_features_2d)

        elif self.model_name == "voxelnet":
            # information naive fusion
            vmfs_fusion = self.model.fusion_net(spatial_features, record_len)
            # map and regression map
            psm, rm = self.model.rpn(vmfs_fusion)
        else:
            raise NotImplementedError()

        output_dict = OrderedDict()
        output_dict['ego'] = {'psm': psm,
                            'rm': rm}

        return output_dict, clipped_perturbation, spatial_features

    def attack_intermediate(self, multi_vehicle_case, ego_id, attacker_id, max_perturb=10, lr=0.2, max_iteration=25, 
                            bbox=None, mode="spoof", real_case=None, original_case=None, real_original_case=None, 
                            real_bbox=None, init_perturbation=None, feature_size=10, attack_mode="RC", n_fake_objects=3):
        """
        Attack an intermediate-fusion perception model.

        Notes:
            MVIG training may pass a predicted attack box directly for stability, without
            running PGD-based box refinement. Evaluation-time attacks that need stronger
            attack quality should enable a PGD-style mode such as "RC+", "BAC", or "BASIC".
        
        Args:
            multi_vehicle_case: Multi-vehicle scene data.
            ego_id: Ego vehicle ID.
            attacker_id: Attacker vehicle ID.
            max_perturb: Maximum perturbation magnitude.
            lr: PGD learning rate.
            max_iteration: Number of PGD iterations.
            bbox: Target bounding box used in RC and spoof modes.
            mode: Spoof or remove mode used by the RC attack.
            real_case, original_case, real_original_case: Reference scene data.
            real_bbox: Ground-truth box in the reference scene.
            init_perturbation: Initial perturbation.
            feature_size: Local feature patch size.
            attack_mode: Attack mode, one of "RC", "BASIC", "BAC", or "RC+".
            n_fake_objects: Number of fake objects to generate in BASIC mode.
        """
        torch.manual_seed(1)
        np.random.seed(1)
        random.seed(1)

        if attack_mode == "BASIC":
            max_iteration = 5
        elif attack_mode == "BAC" or attack_mode == "RC+":
            max_iteration = 50
      

        logging.info(f"Starting {attack_mode} attack with {max_iteration} iterations, lr={lr}")
        
        base_data_dict = self.retrieve_base_data(multi_vehicle_case, ego_id)
        attacker_index = list(base_data_dict.keys()).index(attacker_id)
        assert(attacker_index >= 0)
        
        optimize_batch = self.preprocessors[self.fusion_method](multi_vehicle_case, ego_id)
        optimize_batch_data = train_utils.to_device(self.dataset.collate_batch_test([optimize_batch]), self.device)
        anchor_box = optimize_batch_data['ego']['anchor_box']

        if self.model_name in ["pointpillar", "v2vnet"]:
            feature_dim = 64
        elif self.model_name == "voxelnet":
            feature_dim = 128
        else:
            raise NotImplementedError()

        # `bbox` is the victim-frame target box produced upstream from the MVIG-predicted attack location.
        # PGD-style modes convert this box to a voxel index and use that index as the local perturbation
        # center, which also anchors the BAC/RC+ mask placement on the feature map.
        # Use the target box voxel index as the perturbation center.
        if bbox is not None:
            bbox_tensor = torch.from_numpy(bbox).to(self.device).type(torch.float32)
            bbox_tensor[2] += 0.5 * bbox_tensor[5]
            center = self.point_to_voxel_index(bbox)
        else:
            # Use the feature-map center as the default reference point.
            with torch.no_grad():
                _, _, feature_map = self.attack_intermediate_forward(optimize_batch_data, attacker_index)
                C, H, W = feature_map[attacker_index].size()
                center = np.array([W//2, H//2, 0])
                bbox_tensor = None

        if real_bbox is not None:
            real_center = self.point_to_voxel_index(real_bbox)

        # Initialize perturbation-related tensors.
        with torch.no_grad():
            optimize_output_dict, _, optimize_feature = self.attack_intermediate_forward(optimize_batch_data, attacker_index)

            # Add visualization of initial feature map
            if self.is_visualize and attack_mode.upper() in ["BASIC", "BAC", "RC+"]:
                self.visualize_feature_maps(
                    optimize_feature[attacker_index], 
                    save_path=f"./visualization/{attack_mode}_initial_feature.png",
                    center=center if center is not None else None,
                    feature_size=feature_size if feature_size is not None else None
                )

            if real_case is not None:
                real_batch = self.preprocessors[self.fusion_method](real_case, ego_id)
                real_batch_data = train_utils.to_device(self.dataset.collate_batch_test([real_batch]), self.device)
                _, _, real_feature = self.attack_intermediate_forward(real_batch_data, attacker_index)

            if original_case is not None:
                original_batch = self.preprocessors[self.fusion_method](original_case, ego_id)
                original_batch_data = train_utils.to_device(self.dataset.collate_batch_test([original_batch]), self.device)
                _, _, original_feature = self.attack_intermediate_forward(original_batch_data, attacker_index)
                base_perturbation = ((optimize_feature[attacker_index] - original_feature[attacker_index])[:, center[1]-feature_size:center[1]+feature_size, center[0]-feature_size:center[0]+feature_size]).detach()
            else:
                base_perturbation = torch.zeros(feature_dim, 2 * feature_size, 2 * feature_size).to(self.device).detach()

            if real_original_case is not None:
                real_original_batch = self.preprocessors[self.fusion_method](real_original_case, ego_id)
                real_original_batch_data = train_utils.to_device(self.dataset.collate_batch_test([real_original_batch]), self.device)
                _, _, real_original_feature = self.attack_intermediate_forward(real_original_batch_data, attacker_index)
                real_base_perturbation = ((real_feature[attacker_index] - real_original_feature[attacker_index])[:, real_center[1]-feature_size:real_center[1]+feature_size, real_center[0]-feature_size:real_center[0]+feature_size]).detach()
            else:
                real_base_perturbation = torch.zeros(feature_dim, 2 * feature_size, 2 * feature_size).to(self.device).detach()

        # RC mode skips optimization and returns the direct attack result.
        # This is useful when the caller already provides a stable attack location, such as
        # the direct MVIG prediction used during training. For evaluation, stronger attack
        # execution should switch to a PGD-style mode such as RC+.
        if attack_mode.upper() == "RC":
            logging.info("Using RC attack mode (no PGD optimization)")
            if bbox is not None:
                with torch.no_grad():
                    pred_box_tensor, pred_score_tensor, _ = self.dataset.post_process(optimize_batch_data, optimize_output_dict)
                    if pred_box_tensor is None:
                        pred_bboxes = np.array([])
                        pred_scores = np.array([])
                    else:
                        pred_bboxes = pred_box_tensor.cpu().numpy()
                        pred_bboxes = box_utils.corner_to_center(pred_bboxes, order="lwh")
                        pred_bboxes[:,2] -= 0.5 * pred_bboxes[:,5]
                        pred_scores = pred_score_tensor.cpu().numpy()
                    
                    # Inject the spoof target into the original predictions.
                    if bbox is not None:
                        if pred_bboxes.shape[0] == 0:
                            pred_bboxes = np.array([bbox])
                            pred_scores = np.array([0.9])
                        else:
                            pred_bboxes = np.append(pred_bboxes, [bbox], axis=0)
                            pred_scores = np.append(pred_scores, [0.9])

                return {
                    "pred_bboxes": pred_bboxes,
                    "pred_scores": pred_scores,
                    "attack_mode": "RC"
                }

        # BASIC mode uses a simplified PGD-style optimization loop.
        if attack_mode.upper() == "BASIC":
            logging.info(f"Starting simplified BASIC attack with {max_iteration} iterations, lr={lr}")
            
            # Release cached GPU memory.
            torch.cuda.empty_cache()
            
            # Read the full feature-map size.
            with torch.no_grad():
                _, _, feature_map = self.attack_intermediate_forward(optimize_batch_data, attacker_index)
                C, H, W = feature_map[attacker_index].size()
            
            # Initialize a perturbation over the full feature map.
            if init_perturbation is not None:
                # If an initial perturbation is provided, adjust it to the current shape.
                if init_perturbation.shape[1:] == (H, W):
                    perturbation = torch.from_numpy(init_perturbation).to(self.device)
                else:
                    # If the shape mismatches, create a centered version.
                    perturbation = torch.zeros(feature_dim, H, W).to(self.device)
                    h_start = max(0, H//2 - init_perturbation.shape[1]//2)
                    h_end = min(H, h_start + init_perturbation.shape[1])
                    w_start = max(0, W//2 - init_perturbation.shape[2]//2)
                    w_end = min(W, w_start + init_perturbation.shape[2])
                    
                    # Copy the overlapping region.
                    in_h, in_w = min(h_end-h_start, init_perturbation.shape[1]), min(w_end-w_start, init_perturbation.shape[2])
                    perturbation[:, h_start:h_start+in_h, w_start:w_start+in_w] = torch.from_numpy(
                        init_perturbation[:, :in_h, :in_w]).to(self.device)
            else:
                # Use a small random initialization.
                perturbation = torch.randn(feature_dim, H, W).to(self.device) * 0.005
            
            perturbation.requires_grad = True
            optimizer = torch.optim.Adam([perturbation], lr=lr)
            
            # Track detailed loss curves.
            loss_history = []
            conf_loss_history = []
            background_loss_history = []
            dispersion_loss_history = []
            reg_loss_history = []
            num_boxes_history = []
            
            best_loss = float('inf')
            best_perturbation = None
            best_pred_bboxes = None
            best_pred_scores = None
            best_num_boxes = 0


            # Simplified PGD-style optimization loop.
            for it in range(max_iteration):
                # Copy the batch to avoid mutating the original inputs.
                batch_data = self.detach_all(optimize_batch_data)
                
                # 1. Apply the perturbation in a differentiable way.
                output_dict, clipped_perturbation, _ = self.attack_intermediate_forward(
                    batch_data, attacker_index, 
                    perturbation=perturbation,
                    max_perturb=max_perturb,
                    center=None,  # Explicitly use None instead of forwarding center.
                    feature_size=None  # Explicitly use None instead of forwarding feature_size.
                )
                
                # 2. Build a differentiable loss without breaking the graph via .item().
                # Directly construct the perturbation loss.
                prob = torch.sigmoid(output_dict['ego']['psm'].permute(0, 2, 3, 1)).reshape(-1)
                proposals = self.dataset.post_processor.delta_to_boxes3d(output_dict['ego']['rm'], anchor_box)[0]

                # Define the confidence threshold and high-confidence mask.
                confidence_threshold = 0.3
                high_conf_mask = (prob > confidence_threshold)
                background_mask = ~high_conf_mask

                # Initialize the attack loss.
                adv_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

                # 1. Reduce confidence in high-confidence regions to induce false negatives.
                if torch.any(high_conf_mask):
                    # log(1-p) strongly penalizes highly confident detections.
                    high_conf_loss = -torch.sum(torch.log(1 - prob[high_conf_mask] + 1e-7))
                    adv_loss = adv_loss + high_conf_loss

                # 2. Increase confidence in background regions to induce false positives.
                if torch.any(background_mask):
                    # -log(p) strongly penalizes near-zero confidence.
                    background_loss = -torch.sum(torch.log(prob[background_mask] + 1e-7))
                    adv_loss = adv_loss + 0.5 * background_loss  # Tunable weight.

                # 3. Encourage more dispersed box proposals to reduce overlap.
                # Compute pairwise distances between box centers.
                if proposals.shape[0] > 1:
                    center_points = proposals[:, :2]  # Use xy coordinates only.
                    n = center_points.shape[0]
                    
                    # 1. Limit the number of processed proposals.
                    max_proposals = 100  # Use a practical upper bound.
                    if n > max_proposals:
                        # Keep only the top high-confidence proposals.
                        confidence_values = prob[:n]
                        _, top_indices = torch.topk(confidence_values, min(max_proposals, n))
                        center_points = center_points[top_indices]
                        n = center_points.shape[0]
                    
                    # 2. Compute distances more efficiently.
                    # Compute the ||x_i||^2 vector.
                    x_norm_squared = torch.sum(center_points ** 2, dim=1, keepdim=True)
                    
                    # Compute the x_i · x_j matrix.
                    dot_products = torch.mm(center_points, center_points.t())
                    
                    # Use ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2(x_i · x_j).
                    distances = x_norm_squared + x_norm_squared.t() - 2 * dot_products

                    # Apply the square root with numerical stabilization.
                    distances = torch.sqrt(torch.clamp(distances, min=1e-7))

                    # Set the diagonal to a large value to ignore self-distance.
                    eye_mask = torch.eye(distances.shape[0], device=self.device).bool()
                    distances = distances + eye_mask * 1000.0

                    # Compute the mean nearest-neighbor distance.
                    min_distances = torch.min(distances, dim=1)[0]
                    box_dispersion_loss = -0.1 * torch.mean(min_distances)
                    adv_loss = adv_loss + box_dispersion_loss

                # Add L2 regularization.
                reg_loss = 0.001 * torch.sum(perturbation ** 2)
                total_loss = adv_loss + reg_loss
                
                # Backpropagate.
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                # Clamp the perturbation.
                with torch.no_grad():
                    perturbation.data.clamp_(-max_perturb, max_perturb)
                
                # Evaluate the current result.
                with torch.no_grad():
                    pred_box_tensor, pred_score_tensor, _ = self.dataset.post_process(batch_data, output_dict)
                    
                    if pred_box_tensor is None:
                        pred_bboxes = np.array([])
                        pred_scores = np.array([])
                        num_boxes = 0
                    else:
                        pred_bboxes = pred_box_tensor.cpu().numpy()
                        pred_bboxes = box_utils.corner_to_center(pred_bboxes, order="lwh")
                        pred_bboxes[:,2] -= 0.5 * pred_bboxes[:,5]
                        pred_scores = pred_score_tensor.cpu().numpy()
                        num_boxes = len(pred_bboxes)
                    
                    # Update the best result.
                    if num_boxes > best_num_boxes:
                        best_num_boxes = num_boxes
                        best_loss = total_loss.item()
                        best_perturbation = clipped_perturbation.cpu().detach().numpy()
                        best_pred_bboxes = pred_bboxes
                        best_pred_scores = pred_scores
                        logging.info(f"Found new best result! Generated {num_boxes} detection boxes")
                
                # Release cached memory.
                torch.cuda.empty_cache()
            
            # If no effective result is found, try a random perturbation and evaluate it.
            if best_num_boxes == 0:
                logging.info("No effective perturbation found, trying random perturbation...")
                for attempt in range(5):
                    random_pert = torch.randn(feature_dim, H, W).to(self.device) * max_perturb
                    
                    with torch.no_grad():
                        output_dict, random_clipped_pert, _ = self.attack_intermediate_forward(
                            self.detach_all(optimize_batch_data), attacker_index, 
                            perturbation=random_pert,
                            max_perturb=max_perturb
                        )
                        
                        pred_box_tensor, pred_score_tensor, _ = self.dataset.post_process(
                            self.detach_all(optimize_batch_data), output_dict
                        )
                        
                        if pred_box_tensor is not None:
                            num_boxes = pred_box_tensor.shape[0]
                            if num_boxes > best_num_boxes:
                                best_num_boxes = num_boxes
                                best_perturbation = random_clipped_pert.cpu().detach().numpy()
                                best_pred_bboxes = box_utils.corner_to_center(
                                    pred_box_tensor.cpu().numpy(), order="lwh"
                                )
                                best_pred_bboxes[:,2] -= 0.5 * best_pred_bboxes[:,5]
                                best_pred_scores = pred_score_tensor.cpu().numpy()
                                logging.info(f"Random attempt {attempt+1}: Generated {num_boxes} detection boxes")
     
            # Summarize the attack result.
            logging.info(f"BASIC attack completed. Best result generated {best_num_boxes} detection boxes")
            
            result = {
                "perturbation": best_perturbation,
                "loss": best_loss,
                "pred_bboxes": best_pred_bboxes,
                "pred_scores": best_pred_scores,
                "attack_mode": "BASIC",
                "num_boxes": best_num_boxes,
                "loss_history": loss_history,
                "component_loss": {
                    "conf_loss": conf_loss_history,
                    "background_loss": background_loss_history,
                    "dispersion_loss": dispersion_loss_history,
                    "reg_loss": reg_loss_history,
                    "num_boxes": num_boxes_history
                }
            }
            
            # Visualize final result if we have a best perturbation
            if self.is_visualize and best_perturbation is not None:
                with torch.no_grad():
                    best_pert_tensor = torch.from_numpy(best_perturbation).to(self.device)
                    _, _, final_feature = self.attack_intermediate_forward(
                        self.detach_all(optimize_batch_data), 
                        attacker_index,
                        perturbation=best_pert_tensor,
                        max_perturb=max_perturb,
                        center=None if attack_mode.upper() in ["BASIC", "BAC", "RC+"] else center,
                        feature_size=None if attack_mode.upper() in ["BASIC", "BAC", "RC+"] else feature_size
                    )
                    
                    # Visualize final feature map
                    self.visualize_feature_maps(
                        optimize_feature[attacker_index],
                        final_feature[attacker_index],
                        save_path=f"./visualization/{attack_mode}_final_feature.png",
                        center=center if center is not None else None,
                        feature_size=feature_size if feature_size is not None else None
                    )
            
            return result
        
        # Create a mask for BAC and reuse the same loss design as BASIC.
        if attack_mode.upper() == "BAC":
            logging.info(f"Starting BAC attack with {max_iteration} iterations, lr={lr}")
            
            # Release cached memory.
            torch.cuda.empty_cache()
            
            # Read the full feature-map size.
            with torch.no_grad():
                _, _, feature_map = self.attack_intermediate_forward(optimize_batch_data, attacker_index)
                C, H, W = feature_map[attacker_index].size()
            
            # Initialize a perturbation over the full feature map.
            if init_perturbation is not None:
                # If an initial perturbation is provided, adjust it to the current shape.
                if init_perturbation.shape[1:] == (H, W):
                    perturbation = torch.from_numpy(init_perturbation).to(self.device)
                else:
                    # If the shape mismatches, create a new tensor and center it.
                    perturbation = torch.zeros(feature_dim, H, W).to(self.device)
                    h_start = max(0, H//2 - init_perturbation.shape[1]//2)
                    h_end = min(H, h_start + init_perturbation.shape[1])
                    w_start = max(0, W//2 - init_perturbation.shape[2]//2)
                    w_end = min(W, w_start + init_perturbation.shape[2])
                    
                    # Copy the overlapping region.
                    in_h, in_w = min(h_end-h_start, init_perturbation.shape[1]), min(w_end-w_start, init_perturbation.shape[2])
                    perturbation[:, h_start:h_start+in_h, w_start:w_start+in_w] = torch.from_numpy(
                        init_perturbation[:, :in_h, :in_w]).to(self.device)
            else:
                # Use a small random initialization.
                perturbation = torch.randn(feature_dim, H, W).to(self.device) * 0.005
            
            # Create the BAC mask that restricts the perturbation to a target region.
            mask_size = feature_size
            attack_region_mask = torch.zeros(feature_dim, H, W).to(self.device)
            h_start = max(0, center[1] - mask_size)
            h_end = min(H, center[1] + mask_size)
            w_start = max(0, center[0] - mask_size)
            w_end = min(W, center[0] + mask_size)
            attack_region_mask[:, h_start:h_end, w_start:w_end] = 1.0
            
            # Apply the region mask to the initial perturbation.
            perturbation = perturbation * attack_region_mask
            
            perturbation.requires_grad = True
            optimizer = torch.optim.Adam([perturbation], lr=lr)
            
            # Record detailed loss traces, using the same components as BASIC.
            loss_history = []
            conf_loss_history = []
            background_loss_history = []
            dispersion_loss_history = []
            reg_loss_history = []
            num_boxes_history = []

            best_loss = float('inf')
            best_perturbation = None
            best_pred_bboxes = None
            best_pred_scores = None
            best_num_boxes = 0
            
            # PGD loop for BAC, using the same loss design as BASIC.
            for it in range(max_iteration):
                # Copy batch data to avoid modifying the original input.
                batch_data = self.detach_all(optimize_batch_data)

                # 1. Apply the perturbation in a way that preserves gradient flow.
                output_dict, clipped_perturbation, _ = self.attack_intermediate_forward(
                batch_data, attacker_index, 
                    perturbation=perturbation,
                    max_perturb=max_perturb,
                    center=None,  # Explicitly use None instead of passing center.
                    feature_size=None  # Explicitly use None instead of passing feature_size.
            )
            
                # Ensure the perturbation only affects the masked region.
                perturbation.data = perturbation.data * attack_region_mask
                clipped_perturbation = clipped_perturbation * attack_region_mask
            
                # 2. Use a loss that supports direct backpropagation and avoid `.item()`.
                # Build the perturbation-based loss directly, matching BASIC.
                prob = torch.sigmoid(output_dict['ego']['psm'].permute(0, 2, 3, 1)).reshape(-1)
                proposals = self.dataset.post_processor.delta_to_boxes3d(output_dict['ego']['rm'], anchor_box)[0]

                # Define the confidence threshold and high-confidence mask.
                confidence_threshold = 0.3
                high_conf_mask = (prob > confidence_threshold)
                background_mask = ~high_conf_mask

                # Initialize the loss.
                adv_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

                # 1. Suppress high-confidence regions to encourage false negatives.
                if torch.any(high_conf_mask):
                    # Use `log(1 - p)`, which penalizes highly confident predictions more strongly.
                    high_conf_loss = -torch.sum(torch.log(1 - prob[high_conf_mask] + 1e-7))
                    adv_loss = adv_loss + high_conf_loss

                # 2. Boost low-confidence background regions to encourage false positives.
                if torch.any(background_mask):
                    # Use `-log(p)`, which penalizes near-zero confidence more strongly.
                    background_loss = -torch.sum(torch.log(prob[background_mask] + 1e-7))
                    adv_loss = adv_loss + 0.5 * background_loss  # Adjustable weight.

                # 3. Encourage bounding-box dispersion so neighboring boxes overlap less.
                # Measure pairwise distances between proposal centers.
                if proposals.shape[0] > 1:
                    center_points = proposals[:, :2]  # Use x/y coordinates only.
                    n = center_points.shape[0]
                    
                    # 1. Limit the number of proposals processed.
                    max_proposals = 100  # Use a practical upper bound.
                    if n > max_proposals:
                        # Keep only the top high-confidence proposals.
                        confidence_values = prob[:n]
                        _, top_indices = torch.topk(confidence_values, min(max_proposals, n))
                        center_points = center_points[top_indices]
                        n = center_points.shape[0]
                    
                    # 2. Compute distances with a more efficient vectorized form.
                    # Compute the vector of ||x_i||^2.
                    x_norm_squared = torch.sum(center_points ** 2, dim=1, keepdim=True)
                    
                    # Compute the x_i . x_j matrix.
                    dot_products = torch.mm(center_points, center_points.t())
                    
                    # Use ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2(x_i . x_j).
                    distances = x_norm_squared + x_norm_squared.t() - 2 * dot_products

                    # Take the square root while keeping numerical stability.
                    distances = torch.sqrt(torch.clamp(distances, min=1e-7))

                    # Set the diagonal to a large value to ignore self-distance.
                    eye_mask = torch.eye(distances.shape[0], device=self.device).bool()
                    distances = distances + eye_mask * 1000.0

                    # Compute the mean nearest-neighbor distance directly.
                    min_distances = torch.min(distances, dim=1)[0]
                    box_dispersion_loss = -0.1 * torch.mean(min_distances)
                    adv_loss = adv_loss + box_dispersion_loss
                    
                    # Record the dispersion loss.
                    dispersion_loss_value = box_dispersion_loss.item()
                else:
                    dispersion_loss_value = 0.0

                # Add L2 regularization.
                reg_loss = 0.001 * torch.sum(perturbation ** 2)
                total_loss = adv_loss + reg_loss
                
                # Record each loss component.
                current_loss = total_loss.item()
                loss_history.append(current_loss)
                conf_loss_history.append(high_conf_loss.item() if torch.any(high_conf_mask) else 0.0)
                background_loss_history.append(background_loss.item() if torch.any(background_mask) else 0.0)
                dispersion_loss_history.append(dispersion_loss_value)
                reg_loss_history.append(reg_loss.item())
                
                # Backpropagate.
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                # Re-apply the mask so the perturbation stays inside the region.
                perturbation.data = perturbation.data * attack_region_mask
                
                # Clamp the perturbation.
                with torch.no_grad():
                    perturbation.data.clamp_(-max_perturb, max_perturb)
                
                # Evaluate the result.
                with torch.no_grad():
                    pred_box_tensor, pred_score_tensor, _ = self.dataset.post_process(batch_data, output_dict)
                    
                    if pred_box_tensor is None:
                        pred_bboxes = np.array([])
                        pred_scores = np.array([])
                        num_boxes = 0
                    else:
                        pred_bboxes = pred_box_tensor.cpu().numpy()
                        pred_bboxes = box_utils.corner_to_center(pred_bboxes, order="lwh")
                        pred_bboxes[:,2] -= 0.5 * pred_bboxes[:,5]
                        pred_scores = pred_score_tensor.cpu().numpy()
                        num_boxes = len(pred_bboxes)
                    
                    # Record the number of detected boxes.
                    num_boxes_history.append(num_boxes)
                    
                    # Update the best result.
                    if num_boxes > best_num_boxes:
                        best_num_boxes = num_boxes
                        best_loss = total_loss.item()
                        best_perturbation = clipped_perturbation.cpu().detach().numpy()
                        best_pred_bboxes = pred_bboxes
                        best_pred_scores = pred_scores
                        logging.info(f"Found new best result! Generated {num_boxes} detection boxes")
                
                # Log the detailed loss breakdown.
                logging.info(f"BAC attack iteration {it+1}/{max_iteration}, Loss: {current_loss:.6f}, Boxes: {num_boxes}")
                logging.info(f"  Details - high conf: {conf_loss_history[-1]:.4f}, "
                            f"background: {background_loss_history[-1]:.4f}, "
                            f"dispersion: {dispersion_loss_history[-1]:.4f}, "
                            f"reg: {reg_loss_history[-1]:.4f}")
                
                # Release cached memory.
                torch.cuda.empty_cache()
            
            # If optimization fails, try masked random perturbations as a fallback.
            if best_num_boxes == 0:
                logging.info("No effective perturbation found, trying random perturbation...")
                for attempt in range(5):
                    random_pert = torch.randn(feature_dim, H, W).to(self.device) * max_perturb
                    # Apply the perturbation only inside the masked region.
                    random_pert = random_pert * attack_region_mask
                    
                    with torch.no_grad():
                        output_dict, random_clipped_pert, _ = self.attack_intermediate_forward(
                            self.detach_all(optimize_batch_data), attacker_index, 
                            perturbation=random_pert,
                            max_perturb=max_perturb
                        )
                        
                        pred_box_tensor, pred_score_tensor, _ = self.dataset.post_process(
                            self.detach_all(optimize_batch_data), output_dict
                        )
                        
                        if pred_box_tensor is not None:
                            num_boxes = pred_box_tensor.shape[0]
                            if num_boxes > best_num_boxes:
                                best_num_boxes = num_boxes
                                best_perturbation = random_clipped_pert.cpu().detach().numpy()
                                best_pred_bboxes = box_utils.corner_to_center(
                                    pred_box_tensor.cpu().numpy(), order="lwh"
                                )
                                best_pred_bboxes[:,2] -= 0.5 * best_pred_bboxes[:,5]
                                best_pred_scores = pred_score_tensor.cpu().numpy()
                                logging.info(f"Random attempt {attempt+1}: Generated {num_boxes} detection boxes")
                
            
         
            # Summarize the attack result.
            logging.info(f"BAC attack completed. Best result generated {best_num_boxes} detection boxes")
            
            result = {
                "perturbation": best_perturbation,
                "loss": best_loss,
                "pred_bboxes": best_pred_bboxes,
                "pred_scores": best_pred_scores,
                "attack_mode": "BAC",
                "num_boxes": best_num_boxes,
                "loss_history": loss_history,
                "component_loss": {
                    "conf_loss": conf_loss_history,
                    "background_loss": background_loss_history,
                    "dispersion_loss": dispersion_loss_history,
                    "reg_loss": reg_loss_history,
                    "num_boxes": num_boxes_history
                }
            }

            # Visualize final result if we have a best perturbation
            if self.is_visualize and best_perturbation is not None:
                with torch.no_grad():
                    best_pert_tensor = torch.from_numpy(best_perturbation).to(self.device)
                    _, _, final_feature = self.attack_intermediate_forward(
                        self.detach_all(optimize_batch_data), 
                        attacker_index,
                        perturbation=best_pert_tensor,
                        max_perturb=max_perturb,
                        center=None if attack_mode.upper() in ["BASIC", "BAC", "RC+"] else center,
                        feature_size=None if attack_mode.upper() in ["BASIC", "BAC", "RC+"] else feature_size
                    )
                    
                    # Visualize final feature map
                    self.visualize_feature_maps(
                        optimize_feature[attacker_index],
                        final_feature[attacker_index],
                        save_path=f"./visualization/{attack_mode}_final_feature.png",
                        center=center if center is not None else None,
                        feature_size=feature_size if feature_size is not None else None
                    )
            
            return result
        
        # Create a mask for RC+ and reuse the BAC loss, but add a random location offset.
        if attack_mode.upper() == "RC+":
            logging.info(f"Starting RC+ attack with {max_iteration} iterations, lr={lr}")
            
            # Release cached memory.
            torch.cuda.empty_cache()
            
            # Read the full feature-map size.
            with torch.no_grad():
                _, _, feature_map = self.attack_intermediate_forward(optimize_batch_data, attacker_index)
                C, H, W = feature_map[attacker_index].size()
            
            # Initialize a perturbation over the full feature map.
            if init_perturbation is not None:
                # If an initial perturbation is provided, adjust it to the current shape.
                if init_perturbation.shape[1:] == (H, W):
                    perturbation = torch.from_numpy(init_perturbation).to(self.device)
                else:
                    # If the shape mismatches, create a new tensor and center it.
                    perturbation = torch.zeros(feature_dim, H, W).to(self.device)
                    h_start = max(0, H//2 - init_perturbation.shape[1]//2)
                    h_end = min(H, h_start + init_perturbation.shape[1])
                    w_start = max(0, W//2 - init_perturbation.shape[2]//2)
                    w_end = min(W, w_start + init_perturbation.shape[2])
                    
                    # Copy the overlapping region.
                    in_h, in_w = min(h_end-h_start, init_perturbation.shape[1]), min(w_end-w_start, init_perturbation.shape[2])
                    perturbation[:, h_start:h_start+in_h, w_start:w_start+in_w] = torch.from_numpy(
                        init_perturbation[:, :in_h, :in_w]).to(self.device)
            else:
                # Use a small random initialization.
                perturbation = torch.randn(feature_dim, H, W).to(self.device) * 0.005
            
            # Create the RC+ mask within a constrained region, shrink it to 80%, and shift it randomly.
            # Compute the reduced mask size.
            mask_size_reduced = int(feature_size * 0.8)
            
            # Random offsets, capped at 20% of the original size.
            max_offset = int(feature_size * 0.2)
            h_offset = random.randint(-max_offset, max_offset)
            w_offset = random.randint(-max_offset, max_offset)
            
            # Create the mask with the random offset applied.
            attack_region_mask = torch.zeros(feature_dim, H, W).to(self.device)
            h_center = center[1] + h_offset
            w_center = center[0] + w_offset
            
            # Keep the mask inside the feature-map bounds.
            h_start = max(0, h_center - mask_size_reduced)
            h_end = min(H, h_center + mask_size_reduced)
            w_start = max(0, w_center - mask_size_reduced)
            w_end = min(W, w_center + mask_size_reduced)
            
            # Fill the masked region.
            attack_region_mask[:, h_start:h_end, w_start:w_end] = 1.0
            
            # Record mask metadata for logging and visualization.
            mask_info = {
                "original_center": center,
                "shifted_center": [w_center, h_center],
                "original_size": feature_size,
                "reduced_size": mask_size_reduced,
                "h_offset": h_offset,
                "w_offset": w_offset
            }
            
            logging.info(f"RC+ mask: original center at {center}, shifted to [{w_center}, {h_center}], "
                         f"size reduced from {feature_size} to {mask_size_reduced}")
            
            # Apply the mask to the initial perturbation.
            perturbation = perturbation * attack_region_mask
            
            perturbation.requires_grad = True
            optimizer = torch.optim.Adam([perturbation], lr=lr)
            
            # Record detailed loss traces, matching BAC.
            loss_history = []
            conf_loss_history = []
            background_loss_history = []
            dispersion_loss_history = []
            reg_loss_history = []
            num_boxes_history = []

            best_loss = float('inf')
            best_perturbation = None
            best_pred_bboxes = None
            best_pred_scores = None
            best_num_boxes = 0
            
            # PGD loop for RC+, using the same loss design as BAC.
            for it in range(max_iteration):
                # Copy batch data to avoid modifying the original input.
                batch_data = self.detach_all(optimize_batch_data)

                # 1. Apply the perturbation in a way that preserves gradient flow.
                output_dict, clipped_perturbation, _ = self.attack_intermediate_forward(
                    batch_data, attacker_index, 
                    perturbation=perturbation,
                    max_perturb=max_perturb,
                    center=None,  # Explicitly use None instead of passing center.
                    feature_size=None  # Explicitly use None instead of passing feature_size.
                )
            
                # Ensure the perturbation only affects the masked region.
                perturbation.data = perturbation.data * attack_region_mask
                clipped_perturbation = clipped_perturbation * attack_region_mask
            
                # 2. Use a loss that supports direct backpropagation and avoid `.item()`.
                # Build the perturbation-based loss directly, matching BAC.
                prob = torch.sigmoid(output_dict['ego']['psm'].permute(0, 2, 3, 1)).reshape(-1)
                proposals = self.dataset.post_processor.delta_to_boxes3d(output_dict['ego']['rm'], anchor_box)[0]

                # Define the confidence threshold and high-confidence mask.
                confidence_threshold = 0.3
                high_conf_mask = (prob > confidence_threshold)
                background_mask = ~high_conf_mask

                # Initialize the loss.
                adv_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

                # 1. Suppress high-confidence regions to encourage false negatives.
                if torch.any(high_conf_mask):
                    # Use `log(1 - p)`, which penalizes highly confident predictions more strongly.
                    high_conf_loss = -torch.sum(torch.log(1 - prob[high_conf_mask] + 1e-7))
                    adv_loss = adv_loss + high_conf_loss

                # 2. Boost low-confidence background regions to encourage false positives.
                if torch.any(background_mask):
                    # Use `-log(p)`, which penalizes near-zero confidence more strongly.
                    background_loss = -torch.sum(torch.log(prob[background_mask] + 1e-7))
                    adv_loss = adv_loss + 0.5 * background_loss  # Adjustable weight.

                # 3. Encourage bounding-box dispersion so neighboring boxes overlap less.
                # Measure pairwise distances between proposal centers.
                if proposals.shape[0] > 1:
                    center_points = proposals[:, :2]  # Use x/y coordinates only.
                    n = center_points.shape[0]
                    
                    # 1. Limit the number of proposals processed.
                    max_proposals = 100  # Use a practical upper bound.
                    if n > max_proposals:
                        # Keep only the top high-confidence proposals.
                        confidence_values = prob[:n]
                        _, top_indices = torch.topk(confidence_values, min(max_proposals, n))
                        center_points = center_points[top_indices]
                        n = center_points.shape[0]
                    
                    # 2. Compute distances with a more efficient vectorized form.
                    # Compute the vector of ||x_i||^2.
                    x_norm_squared = torch.sum(center_points ** 2, dim=1, keepdim=True)
                    
                    # Compute the x_i . x_j matrix.
                    dot_products = torch.mm(center_points, center_points.t())
                    
                    # Use ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2(x_i . x_j).
                    distances = x_norm_squared + x_norm_squared.t() - 2 * dot_products

                    # Take the square root while keeping numerical stability.
                    distances = torch.sqrt(torch.clamp(distances, min=1e-7))

                    # Set the diagonal to a large value to ignore self-distance.
                    eye_mask = torch.eye(distances.shape[0], device=self.device).bool()
                    distances = distances + eye_mask * 1000.0

                    # Compute the mean nearest-neighbor distance directly.
                    min_distances = torch.min(distances, dim=1)[0]
                    box_dispersion_loss = -0.1 * torch.mean(min_distances)
                    adv_loss = adv_loss + box_dispersion_loss
                    
                    # Record the dispersion loss.
                    dispersion_loss_value = box_dispersion_loss.item()
                else:
                    dispersion_loss_value = 0.0

                # Add L2 regularization.
                reg_loss = 0.001 * torch.sum(perturbation ** 2)
                total_loss = adv_loss + reg_loss
                
                # Record each loss component.
                current_loss = total_loss.item()
                loss_history.append(current_loss)
                conf_loss_history.append(high_conf_loss.item() if torch.any(high_conf_mask) else 0.0)
                background_loss_history.append(background_loss.item() if torch.any(background_mask) else 0.0)
                dispersion_loss_history.append(dispersion_loss_value)
                reg_loss_history.append(reg_loss.item())
                
                # Backpropagate.
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                # Re-apply the mask so the perturbation stays inside the region.
                perturbation.data = perturbation.data * attack_region_mask
                
                # Clamp the perturbation.
                with torch.no_grad():
                    perturbation.data.clamp_(-max_perturb, max_perturb)
                
                # Evaluate the result.
                with torch.no_grad():
                    pred_box_tensor, pred_score_tensor, _ = self.dataset.post_process(batch_data, output_dict)
                    
                    if pred_box_tensor is None:
                        pred_bboxes = np.array([])
                        pred_scores = np.array([])
                        num_boxes = 0
                    else:
                        pred_bboxes = pred_box_tensor.cpu().numpy()
                        pred_bboxes = box_utils.corner_to_center(pred_bboxes, order="lwh")
                        pred_bboxes[:,2] -= 0.5 * pred_bboxes[:,5]
                        pred_scores = pred_score_tensor.cpu().numpy()
                        num_boxes = len(pred_bboxes)
                    
                    # Record the number of detected boxes.
                    num_boxes_history.append(num_boxes)
                    
                    # Update the best result.
                    if num_boxes > best_num_boxes:
                        best_num_boxes = num_boxes
                        best_loss = total_loss.item()
                        best_perturbation = clipped_perturbation.cpu().detach().numpy()
                        best_pred_bboxes = pred_bboxes
                        best_pred_scores = pred_scores
                        logging.info(f"Found new best result! Generated {num_boxes} detection boxes")
                
                # Log the detailed loss breakdown.
                logging.info(f"RC+ attack iteration {it+1}/{max_iteration}, Loss: {current_loss:.6f}, Boxes: {num_boxes}")
                logging.info(f"  Details - high conf: {conf_loss_history[-1]:.4f}, "
                            f"background: {background_loss_history[-1]:.4f}, "
                            f"dispersion: {dispersion_loss_history[-1]:.4f}, "
                            f"reg: {reg_loss_history[-1]:.4f}")
                
                # Release cached memory.
                torch.cuda.empty_cache()
            
            # If optimization fails, try masked random perturbations as a fallback.
            if best_num_boxes == 0:
                logging.info("No effective perturbation found, trying random perturbation...")
                for attempt in range(5):
                    random_pert = torch.randn(feature_dim, H, W).to(self.device) * max_perturb
                    # Apply the perturbation only inside the masked region.
                    random_pert = random_pert * attack_region_mask
                    
                    with torch.no_grad():
                        output_dict, random_clipped_pert, _ = self.attack_intermediate_forward(
                            self.detach_all(optimize_batch_data), attacker_index, 
                            perturbation=random_pert,
                            max_perturb=max_perturb
                        )
                        
                        pred_box_tensor, pred_score_tensor, _ = self.dataset.post_process(
                            self.detach_all(optimize_batch_data), output_dict
                        )
                        
                        if pred_box_tensor is not None:
                            num_boxes = pred_box_tensor.shape[0]
                            if num_boxes > best_num_boxes:
                                best_num_boxes = num_boxes
                                best_perturbation = random_clipped_pert.cpu().detach().numpy()
                                best_pred_bboxes = box_utils.corner_to_center(
                                    pred_box_tensor.cpu().numpy(), order="lwh"
                                )
                                best_pred_bboxes[:,2] -= 0.5 * best_pred_bboxes[:,5]
                                best_pred_scores = pred_score_tensor.cpu().numpy()
                                logging.info(f"Random attempt {attempt+1}: Generated {num_boxes} detection boxes")
            
            # Summarize the attack result.
            logging.info(f"RC+ attack completed. Best result generated {best_num_boxes} detection boxes")
            
            result = {
                "perturbation": best_perturbation,
                "loss": best_loss,
                "pred_bboxes": best_pred_bboxes,
                "pred_scores": best_pred_scores,
                "attack_mode": "RC+",
                "num_boxes": best_num_boxes,
                "loss_history": loss_history,
                "component_loss": {
                    "conf_loss": conf_loss_history,
                    "background_loss": background_loss_history,
                    "dispersion_loss": dispersion_loss_history,
                    "reg_loss": reg_loss_history,
                    "num_boxes": num_boxes_history
                },
                "mask_info": mask_info  # Keep mask metadata for later analysis.
            }

            # Visualize final result if we have a best perturbation
            if self.is_visualize and best_perturbation is not None:
                with torch.no_grad():
                    best_pert_tensor = torch.from_numpy(best_perturbation).to(self.device)
                    _, _, final_feature = self.attack_intermediate_forward(
                        self.detach_all(optimize_batch_data), 
                        attacker_index,
                        perturbation=best_pert_tensor,
                        max_perturb=max_perturb,
                        center=None,  # RC+ uses the full feature map.
                        feature_size=None
                    )
                    
                    # Visualize final feature map
                    self.visualize_feature_maps(
                        optimize_feature[attacker_index],
                        final_feature[attacker_index],
                        save_path=f"./visualization/{attack_mode}_final_feature.png",
                        center=[w_center, h_center],  # Use the shifted center.
                        feature_size=mask_size_reduced,  # Use the reduced mask size.
                        show_mask=True,  # Display the mask region.
                    )
            
            return result
        
        
        with torch.no_grad():
            initial_output_dict, _, _ = self.attack_intermediate_forward(
                self.detach_all(optimize_batch_data), attacker_index, 
                perturbation=None,
                max_perturb=max_perturb,
                center=None,
                feature_size=None
            )
            initial_pred_box_tensor, _, _ = self.dataset.post_process(
                self.detach_all(optimize_batch_data), initial_output_dict
            )
            initial_box_count = 0 if initial_pred_box_tensor is None else initial_pred_box_tensor.shape[0]
            logging.info(f"Initial state has {initial_box_count} detection boxes")

            # Initialize the attack-success flag.
            attack_success = False
            
            # PGD loop for RC+.
            for it in range(max_iteration):
                # Copy batch data to avoid modifying the original input.
                batch_data = self.detach_all(optimize_batch_data)
                
                # Apply the perturbation.
                output_dict, clipped_perturbation, _ = self.attack_intermediate_forward(
                    batch_data, attacker_index, 
                    perturbation=perturbation,
                    max_perturb=max_perturb,
                    center=None,  # Explicitly use None to operate on the full feature map.
                    feature_size=None
                )
                
                # Ensure the perturbation only affects the masked region.
                perturbation.data = perturbation.data * attack_region_mask
                clipped_perturbation = clipped_perturbation * attack_region_mask
                
                # Read classification probabilities and object proposals.
                prob = torch.sigmoid(output_dict['ego']['psm'].permute(0, 2, 3, 1)).reshape(-1)
                proposals = self.dataset.post_processor.delta_to_boxes3d(output_dict['ego']['rm'], anchor_box)[0]
                
                # Reuse the BAC-style loss framework.
                # Define the confidence threshold and high-confidence mask.
                confidence_threshold = 0.3
                high_conf_mask = (prob > confidence_threshold)
                background_mask = ~high_conf_mask

                # Initialize the loss.
                adv_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

                # 1. Suppress high-confidence regions to encourage false negatives.
                if torch.any(high_conf_mask):
                    high_conf_loss = -torch.sum(torch.log(1 - prob[high_conf_mask] + 1e-7))
                    adv_loss = adv_loss + high_conf_loss
                    conf_loss_value = high_conf_loss.item()
                else:
                    conf_loss_value = 0.0

                # 2. Boost low-confidence regions to encourage false positives.
                if torch.any(background_mask):
                    background_loss = -torch.sum(torch.log(prob[background_mask] + 1e-7))
                    adv_loss = adv_loss + 0.5 * background_loss
                    background_loss_value = background_loss.item()
                else:
                    background_loss_value = 0.0

                # 3. Encourage bounding-box dispersion so neighboring boxes overlap less.
                dispersion_loss_value = 0.0
                if proposals.shape[0] > 1:
                    center_points = proposals[:, :2]  # Use x/y coordinates only.
                    n = center_points.shape[0]
                    
                    # Limit the number of proposals processed.
                    max_proposals = 100  # Use a practical upper bound.
                    if n > max_proposals:
                        confidence_values = prob[:n]
                        _, top_indices = torch.topk(confidence_values, min(max_proposals, n))
                        center_points = center_points[top_indices]
                        n = center_points.shape[0]
                    
                    # Compute the center-distance matrix.
                    x_norm_squared = torch.sum(center_points ** 2, dim=1, keepdim=True)
                    dot_products = torch.mm(center_points, center_points.t())
                    distances = x_norm_squared + x_norm_squared.t() - 2 * dot_products
                    distances = torch.sqrt(torch.clamp(distances, min=1e-7))
                    
                    # Set the diagonal to a large value.
                    eye_mask = torch.eye(distances.shape[0], device=self.device).bool()
                    distances = distances + eye_mask * 1000.0
                    
                    # Compute nearest-neighbor distances.
                    min_distances = torch.min(distances, dim=1)[0]
                    
                    # RC+-specific logic: focus only on boxes inside the mask region.
                    if proposals.shape[0] > 0:
                        proposal_centers = proposals[:, :2]  # x/y coordinates.
                        in_mask_region = (proposal_centers[:, 0] >= w_start) & (proposal_centers[:, 0] < w_end) & \
                                            (proposal_centers[:, 1] >= h_start) & (proposal_centers[:, 1] < h_end)
                        
                        # Compute dispersion only for boxes inside the mask.
                        mask_region_indices = torch.where(in_mask_region)[0]
                        if len(mask_region_indices) > 1:
                            mask_distances = min_distances[mask_region_indices]
                            box_dispersion_loss = -0.1 * torch.mean(mask_distances)
                            adv_loss = adv_loss + box_dispersion_loss
                            dispersion_loss_value = box_dispersion_loss.item()
                        
                        # RC+-specific logic: encourage boxes similar to the target bbox.
                        if bbox is not None:
                            bbox_tensor_expanded = bbox_tensor.unsqueeze(0).expand(proposals.shape[0], -1)
                            iou = self.iou_torch(proposals[:,[0,1,2,5,4,3,6]], bbox_tensor_expanded)
                            high_iou_mask = (iou > 0.3)
                            
                            # Increase the confidence of boxes that match the target bbox.
                            if torch.any(high_iou_mask):
                                target_enhancement = -2.0 * torch.sum(torch.log(prob[high_iou_mask] + 1e-7))
                                adv_loss = adv_loss + target_enhancement

                # 4. Regularization loss.
                reg_loss = 0.001 * torch.sum(perturbation ** 2)
                reg_loss_value = reg_loss.item()

                # Total loss.
                total_loss = adv_loss + reg_loss

                # Record each loss component.
                current_loss = total_loss.item()
                loss_history.append(current_loss)
                conf_loss_history.append(conf_loss_value)
                background_loss_history.append(background_loss_value)
                dispersion_loss_history.append(dispersion_loss_value)
                reg_loss_history.append(reg_loss_value)
                
                # Backpropagate.
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                # Re-apply the mask so the perturbation stays inside the region.
                perturbation.data = perturbation.data * attack_region_mask
                
                # Clamp the perturbation.
                with torch.no_grad():
                    perturbation.data.clamp_(-max_perturb, max_perturb)
                
                # Evaluate the result.
                with torch.no_grad():
                    pred_box_tensor, pred_score_tensor, _ = self.dataset.post_process(batch_data, output_dict)

                if pred_box_tensor is None:
                    pred_bboxes = np.array([])
                    pred_scores = np.array([])
                    num_boxes = 0
                else:
                    pred_bboxes = pred_box_tensor.cpu().numpy()
                    pred_bboxes = box_utils.corner_to_center(pred_bboxes, order="lwh")
                    pred_bboxes[:,2] -= 0.5 * pred_bboxes[:,5]
                    pred_scores = pred_score_tensor.cpu().numpy()
                    num_boxes = len(pred_bboxes)

                    #     # For RC+, check whether any generated box resembles the target bbox.
                    #     if num_boxes > 0:
                    #         # Check the similarity between each generated box and the target bbox.
                    #         found_target = False
                    #         for i in range(num_boxes):
                    #             # Pass a single bounding box directly instead of an array.
                    #             box_iou = iou2d(pred_bboxes[i], bbox)
                    #             if box_iou > 0.3:  # Reuse the same threshold as above.
                    #                 found_target = True
                    #                 logging.info(f"Found target-like bbox with IoU: {box_iou:.4f}, score: {pred_scores[i]:.4f}")
                    #                 break

                    # if found_target and (best_num_boxes == 0 or current_loss < best_loss):
                    #     best_num_boxes = num_boxes
                    #     best_loss = current_loss
                    #     best_perturbation = clipped_perturbation.cpu().detach().numpy()
                    #     best_pred_bboxes = pred_bboxes
                    #     best_pred_scores = pred_scores
                    #     logging.info(f"Found new best result with target box at iteration {it+1}!")
                        
                # Simplified success rule: any increase in detected boxes counts as success.
                if num_boxes > initial_box_count:
                    logging.info(f"Attack successful at iteration {it+1}! Increased box count from {initial_box_count} to {num_boxes}")
                    best_num_boxes = num_boxes
                    best_loss = current_loss
                    best_perturbation = clipped_perturbation.cpu().detach().numpy()
                    best_pred_bboxes = pred_bboxes
                    best_pred_scores = pred_scores
                    attack_success = True
                    
                    # Optionally check whether a box appears near the target location.
                    if bbox is not None and num_boxes > 0:
                        for i in range(num_boxes):
                            try:
                                box_iou = iou2d(pred_bboxes[i], bbox)
                                if box_iou > 0.3:
                                    logging.info(f"Found box near target location with IoU: {box_iou:.4f}, score: {pred_scores[i]:.4f}")
                                    break
                            except:
                                pass
                    
                    # Stop early once the attack succeeds.
                    logging.info("Attack successful, terminating early.")
                    break
                
                # Keep the same best-result update logic even without early stopping.
                if num_boxes > best_num_boxes:
                    best_num_boxes = num_boxes
                    best_loss = current_loss
                    best_perturbation = clipped_perturbation.cpu().detach().numpy()
                    best_pred_bboxes = pred_bboxes
                    best_pred_scores = pred_scores
            
                # Record the number of detected boxes.
                num_boxes_history.append(num_boxes)
                
                # Log the detailed loss breakdown and current status.
                logging.info(f"RC+ attack iteration {it+1}/{max_iteration}, Loss: {current_loss:.6f}, Boxes: {num_boxes}")
                logging.info(f"  Details - target loss: {conf_loss_history[-1]:.4f}, "
                            f"outside loss: {background_loss_history[-1]:.4f}, "
                            f"reg loss: {reg_loss_history[-1]:.4f}")
                
                # If a target-like box is found, log extra details.
                if 'found_target' in locals() and found_target:
                    logging.info(f"  Found target-like bbox in iteration {it+1}")
                
                # Release cached memory.
                torch.cuda.empty_cache()
            
            # Summarize the attack outcome.
            if attack_success:
                logging.info(f"RC+ attack succeeded after {it+1}/{max_iteration} iterations.")
                logging.info(f"Increased detection boxes from {initial_box_count} to {best_num_boxes}.")
            else:
                logging.info(f"RC+ attack completed all {max_iteration} iterations without early termination.")
                if best_num_boxes > initial_box_count:
                    logging.info(f"Attack was still successful, increased boxes from {initial_box_count} to {best_num_boxes}.")
                else:
                    logging.info(f"Attack was not successful in increasing box count. Initial: {initial_box_count}, Final: {best_num_boxes}")

        result = {
                    "perturbation": best_perturbation,
                    "loss": best_loss if 'best_loss' in locals() else float('inf'),
                    "pred_bboxes": best_pred_bboxes,
                    "pred_scores": best_pred_scores,
                    "attack_mode": "RC+",
                    "num_boxes": len(best_pred_bboxes),
                    "loss_history": loss_history if 'loss_history' in locals() else []
                }
        
        if 'conf_loss_history' in locals():
            result["component_loss"] = {
                "conf_loss": conf_loss_history,
                "background_loss": background_loss_history,
                "reg_loss": reg_loss_history,
                "num_boxes": num_boxes_history
            }
        
        # Visualize final result if we have a best perturbation
        if self.is_visualize and best_perturbation is not None:
            with torch.no_grad():
                best_pert_tensor = torch.from_numpy(best_perturbation).to(self.device)
                _, _, final_feature = self.attack_intermediate_forward(
                    self.detach_all(optimize_batch_data), 
                    attacker_index,
                    perturbation=best_pert_tensor,
                    max_perturb=max_perturb,
                    center=None if attack_mode.upper() in ["BASIC", "BAC", "RC+"] else center,
                    feature_size=None if attack_mode.upper() in ["BASIC", "BAC", "RC+"] else feature_size
                )
                
                # Visualize final feature map
                self.visualize_feature_maps(
                    optimize_feature[attacker_index],
                    final_feature[attacker_index],
                    save_path=f"./visualization/{attack_mode}_final_feature.png",
                    center=center if center is not None else None,
                    feature_size=feature_size if feature_size is not None else None,
                    show_mask=True
                )
        
        return result

    def retrieve_base_data(self, multi_vehicle_case, ego_id):
        data = OrderedDict()
        ego_pose = multi_vehicle_case[ego_id]["lidar_pose"]
        for vehicle_id, vehicle_data in multi_vehicle_case.items():
            data[vehicle_id] = OrderedDict()
            data[vehicle_id]['ego'] = (vehicle_id == ego_id)
            data[vehicle_id]["cav_id"] = vehicle_id
            data[vehicle_id]['time_delay'] = 0
            if "params" in vehicle_data:
                data[vehicle_id]['params'] = vehicle_data["params"]
                data[vehicle_id]['params']["lidar_pose"] = vehicle_data["lidar_pose"]
                data[vehicle_id]['params']["transformation_matrix"] = np.dot(np.linalg.inv(pose_to_transformation(ego_pose)), pose_to_transformation(vehicle_data["lidar_pose"]))
            else:
                data[vehicle_id]['params'] = {
                    "lidar_pose": vehicle_data["lidar_pose"],
                    "vehicles": {},
                }
            if self.model_name in ["pointpillar"]:
                data[vehicle_id]['lidar_np'] = vehicle_data["lidar"].astype(np.float32)
                data[vehicle_id]['lidar_np'][:,3] = 1
            else:
                data[vehicle_id]['lidar_np'] = vehicle_data["lidar"][:,:4].astype(np.float32)
        return data

    def early_preprocess(self, multi_vehicle_case, ego_id):
        base_data_dict = self.retrieve_base_data(multi_vehicle_case, ego_id)

        processed_data_dict = OrderedDict()
        processed_data_dict['ego'] = {}

        ego_lidar_pose = base_data_dict[ego_id]["params"]['lidar_pose']

        projected_lidar_stack = []
        object_stack = []
        object_id_stack = []

        # loop over all CAVs to process information
        for cav_id, selected_cav_base in base_data_dict.items():
            # check if the cav is within the communication range with ego
            distance = \
                math.sqrt((selected_cav_base['params']['lidar_pose'][0] -
                           ego_lidar_pose[0]) ** 2 + (
                                  selected_cav_base['params'][
                                      'lidar_pose'][1] - ego_lidar_pose[
                                      1]) ** 2)
            # if distance > opencood.data_utils.datasets.COM_RANGE:
            #     continue

            selected_cav_processed = self.dataset.get_item_single_car(
                selected_cav_base,
                ego_lidar_pose)

            # all these lidar and object coordinates are projected to ego
            # already.
            projected_lidar_stack.append(
                selected_cav_processed['projected_lidar'])
            object_stack.append(selected_cav_processed['object_bbx_center'])
            object_id_stack += selected_cav_processed['object_ids']

        # exclude all repetitive objects
        unique_indices = \
            [object_id_stack.index(x) for x in set(object_id_stack)]
        object_stack = np.vstack(object_stack)
        object_stack = object_stack[unique_indices]

        # make sure bounding boxes across all frames have the same number
        object_bbx_center = \
            np.zeros((self.dataset.params['postprocess']['max_num'], 7))
        mask = np.zeros(self.dataset.params['postprocess']['max_num'])
        object_bbx_center[:object_stack.shape[0], :] = object_stack
        mask[:object_stack.shape[0]] = 1

        # convert list to numpy array, (N, 4)
        projected_lidar_stack = np.vstack(projected_lidar_stack)

        # we do lidar filtering in the stacked lidar
        projected_lidar_stack = mask_points_by_range(projected_lidar_stack,
                                                     self.dataset.params['preprocess'][
                                                         'cav_lidar_range'])
        # augmentation may remove some of the bbx out of range
        object_bbx_center_valid = object_bbx_center[mask == 1]
        object_bbx_center_valid = \
            box_utils.mask_boxes_outside_range_numpy(object_bbx_center_valid,
                                                     self.dataset.params['preprocess'][
                                                         'cav_lidar_range'],
                                                     self.dataset.params['postprocess'][
                                                         'order']
                                                     )
        # Two versions of OpenCOOD!
        if isinstance(object_bbx_center_valid, tuple):
            object_bbx_center_valid = object_bbx_center_valid[0]

        mask[object_bbx_center_valid.shape[0]:] = 0
        object_bbx_center[:object_bbx_center_valid.shape[0]] = \
            object_bbx_center_valid
        object_bbx_center[object_bbx_center_valid.shape[0]:] = 0

        # pre-process the lidar to voxel/bev/downsampled lidar
        lidar_dict = self.dataset.pre_processor.preprocess(projected_lidar_stack)

        # generate the anchor boxes
        anchor_box = self.dataset.post_processor.generate_anchor_box()

        # generate targets label
        label_dict = \
            self.dataset.post_processor.generate_label(
                gt_box_center=object_bbx_center,
                anchors=anchor_box,
                mask=mask)

        processed_data_dict['ego'].update(
            {'object_bbx_center': object_bbx_center,
             'object_bbx_mask': mask,
             'object_ids': [object_id_stack[i] for i in unique_indices],
             'anchor_box': anchor_box,
             'processed_lidar': lidar_dict,
             'label_dict': label_dict})

        return processed_data_dict

    def intermediate_preprocess(self, multi_vehicle_case, ego_id):
        base_data_dict = self.retrieve_base_data(multi_vehicle_case, ego_id)

        processed_data_dict = OrderedDict()
        processed_data_dict['ego'] = {}

        ego_id = -1
        ego_lidar_pose = []

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                break

        assert ego_id != -1
        assert len(ego_lidar_pose) > 0

        pairwise_t_matrix = \
            self.dataset.get_pairwise_transformation(base_data_dict,
                                             self.dataset.max_cav)

        processed_features = []
        object_stack = []
        object_id_stack = []

        # loop over all CAVs to process information
        for cav_id, selected_cav_base in base_data_dict.items():
            # check if the cav is within the communication range with ego
            distance = \
                math.sqrt((selected_cav_base['params']['lidar_pose'][0] -
                           ego_lidar_pose[0]) ** 2 + (
                                  selected_cav_base['params'][
                                      'lidar_pose'][1] - ego_lidar_pose[
                                      1]) ** 2)
            # if distance > opencood.data_utils.datasets.COM_RANGE:
            #     continue

            selected_cav_processed = self.dataset.get_item_single_car(
                selected_cav_base,
                ego_lidar_pose)

            object_stack.append(selected_cav_processed['object_bbx_center'])
            object_id_stack += selected_cav_processed['object_ids']
            processed_features.append(
                selected_cav_processed['processed_features'])

        # exclude all repetitive objects
        unique_indices = \
            [object_id_stack.index(x) for x in set(object_id_stack)]
        object_stack = np.vstack(object_stack)
        object_stack = object_stack[unique_indices]

        # make sure bounding boxes across all frames have the same number
        object_bbx_center = \
            np.zeros((self.dataset.params['postprocess']['max_num'], 7))
        mask = np.zeros(self.dataset.params['postprocess']['max_num'])
        object_bbx_center[:object_stack.shape[0], :] = object_stack
        mask[:object_stack.shape[0]] = 1

        # merge preprocessed features from different cavs into the same dict
        cav_num = len(processed_features)
        merged_feature_dict = self.dataset.merge_features_to_dict(processed_features)

        # generate the anchor boxes
        anchor_box = self.dataset.post_processor.generate_anchor_box()

        # generate targets label
        label_dict = \
            self.dataset.post_processor.generate_label(
                gt_box_center=object_bbx_center,
                anchors=anchor_box,
                mask=mask)

        processed_data_dict['ego'].update(
            {'object_bbx_center': object_bbx_center,
             'object_bbx_mask': mask,
             'object_ids': [object_id_stack[i] for i in unique_indices],
             'anchor_box': anchor_box,
             'processed_lidar': merged_feature_dict,
             'label_dict': label_dict,
             'cav_num': cav_num,
             'pairwise_t_matrix': pairwise_t_matrix,
             'velocity': [0 for i in range(len(multi_vehicle_case))],
             'time_delay': [0 for i in range(len(multi_vehicle_case))],
             'infra': [0 for i in range(len(multi_vehicle_case))],
             'spatial_correction_matrix': [np.eye(4) for i in range(len(multi_vehicle_case))],
             "pairwise_t_matrix": pairwise_t_matrix})

        return processed_data_dict

    def late_preprocess(self, multi_vehicle_case, ego_id):
        base_data_dict = self.retrieve_base_data(multi_vehicle_case, ego_id)
        reformat_data_dict = self.dataset.get_item_test(base_data_dict)

        return reformat_data_dict

    def points_to_voxel_torch(self, pcd):
        # https://github.com/DerrickXuNu/OpenCOOD/blob/main/opencood/data_utils/pre_processor/voxel_preprocessor.py
        # full_mean = False
        # block_filtering = False
        data_dict = {}
        lidar_range = self.dataset.pre_processor.params["cav_lidar_range"]
        voxel_size = self.dataset.pre_processor.params["args"]["voxel_size"]
        max_points_per_voxel = self.dataset.pre_processor.params["args"]["max_points_per_voxel"]

        voxel_coords = torch.floor((pcd[:, :3] - 
                torch.tensor(lidar_range[:3]).to(self.device)
            ) / torch.tensor(voxel_size).to(self.device)).int()

        voxel_coords = voxel_coords[:, [2, 1, 0]]
        voxel_coords, inv_ind, voxel_counts = torch.unique(voxel_coords, dim=0,
                                                           return_inverse=True,
                                                           return_counts=True)
        
        voxel_features = torch.zeros((len(voxel_coords), max_points_per_voxel, 4), dtype=torch.float32).to(self.device)

        for i in range(len(voxel_coords)):
            pts = pcd[inv_ind == i]
            if voxel_counts[i] > max_points_per_voxel:
                pts = pts[:max_points_per_voxel, :]
                voxel_counts[i] = max_points_per_voxel

            voxel_features[i, :pts.shape[0], :] = pts

        data_dict['voxel_features'] = voxel_features
        data_dict['voxel_coords'] = voxel_coords
        data_dict['voxel_num_points'] = voxel_counts

        return data_dict

    def point_to_voxel_index(self, point):
        lidar_range = self.dataset.pre_processor.params["cav_lidar_range"]
        voxel_size = self.dataset.pre_processor.params["args"]["voxel_size"]
        voxel_index = (np.floor(point[:3] - lidar_range[:3]) / voxel_size).astype(np.int32)
        return voxel_index

    def iou_torch(self, bboxes_a, bboxes_b):
        corners2d_a = torch.unsqueeze(box_utils.boxes_to_corners2d(bboxes_a, order="lwh")[:,:,:2], 0)
        corners2d_b = torch.unsqueeze(box_utils.boxes_to_corners2d(bboxes_b, order="lwh")[:,:,:2], 0)
        area_a = bboxes_a[:, 3] * bboxes_a[:, 4]
        area_b = bboxes_b[:, 3] * bboxes_b[:, 4]
        area_inter, _ = oriented_box_intersection_2d(corners2d_a, corners2d_b)
        area_inter = area_inter.squeeze()
        height_inter = torch.clip(
            torch.min(bboxes_a[:, 2] + 0.5 * bboxes_a[:, 5], bboxes_b[:, 2] + 0.5 * bboxes_b[:, 5]) - \
            torch.max(bboxes_a[:, 2] - 0.5 * bboxes_a[:, 5], bboxes_b[:, 2] - 0.5 * bboxes_b[:, 5]),
            min=0, max=5)
        iou = area_inter * height_inter / (area_a * bboxes_a[:, 5] + area_b * bboxes_b[:, 5] - area_inter * height_inter)
        return iou

    def pose_to_transformation_torch(self, pose, dim=2):
        x, y, z, roll, yaw, pitch = pose[0], pose[1], pose[2], torch.deg2rad(pose[3]), torch.deg2rad(pose[4]), torch.deg2rad(pose[5])
        if dim == 2:
            T = torch.zeros((3, 3)).to(torch.float32).to(self.device)
            T[0, 0] = torch.cos(yaw)
            T[0, 1] = 0 - torch.sin(yaw)
            T[0, 2] = x
            T[1, 0] = torch.sin(yaw)
            T[1, 1] = torch.cos(yaw)
            T[1, 2] = y
            T[2, 2] = 1
        elif dim == 3:
            T = torch.tensor([[torch.cos(yaw)*torch.cos(pitch), 
                        torch.cos(yaw)*torch.sin(pitch)*torch.sin(roll)-torch.sin(yaw)*torch.cos(roll), 
                        torch.cos(yaw)*torch.sin(pitch)*torch.cos(roll)+torch.sin(yaw)*torch.sin(roll),
                        x],
                        [torch.sin(yaw)*torch.cos(pitch), 
                        torch.sin(yaw)*torch.sin(pitch)*torch.sin(roll)+torch.cos(yaw)*torch.cos(roll), 
                        torch.sin(yaw)*torch.sin(pitch)*torch.cos(roll)-torch.cos(yaw)*torch.sin(roll),
                        y],
                        [-torch.sin(pitch), 
                        torch.cos(pitch)*torch.sin(roll), 
                        torch.cos(pitch)*torch.cos(roll),
                        z],
                        [0, 0, 0, 1]]).to(self.device)
        return T

    def attacker_to_origin_transformation(self, T, attacker_pose, origin_pose, dim=2):
        attacker_T = self.pose_to_transformation_torch(attacker_pose, dim=dim)
        origin_T = self.pose_to_transformation_torch(origin_pose, dim=dim)
        return torch.matmul(torch.matmul(torch.matmul(torch.matmul(torch.inverse(origin_T), attacker_T), T), torch.inverse(attacker_T)), origin_T)

    def detach_all(self, x):
        if isinstance(x, dict):
            y = {}
            for key, value in x.items():
                y[key] = self.detach_all(value)
        elif isinstance(x, list):
            y = []
            for value in x:
                y.append(self.detach_all(value))
        elif isinstance(x, torch.Tensor):
            y = x.detach()
        else:
            y = x
        return y

    def visualize_feature_maps(self, feature_map, perturbed_feature_map=None, save_path=None, center=None, feature_size=None, show_mask=False):
        """
        Visualize feature maps before and after perturbation
        
        Args:
            feature_map: Original feature map tensor (C, H, W)
            perturbed_feature_map: Perturbed feature map tensor (C, H, W), optional
            save_path: Path to save the visualization, if None will show the plot
            center: Center point of interest [x, y, z], optional
            feature_size: Size of the region of interest around center, optional
            show_mask: Whether to show the mask rectangle on the feature map, default False
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        import os
        
        # Convert tensors to numpy if needed
        if isinstance(feature_map, torch.Tensor):
            feature_map = feature_map.detach().cpu().numpy()
        
        if perturbed_feature_map is not None and isinstance(perturbed_feature_map, torch.Tensor):
            perturbed_feature_map = perturbed_feature_map.detach().cpu().numpy()
        
        # Get feature dimensions
        C, H, W = feature_map.shape
        
        # Create a figure with square aspect ratio
        if perturbed_feature_map is not None:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            fig.suptitle('Feature Map Visualization: Original vs Perturbed', fontsize=14)
        else:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            axes = [ax]  # Make it a list for consistent indexing
        
        # Compute channel-wise mean for visualization
        mean_feature = np.mean(feature_map, axis=0)
        
        # Create normalization for comparison
        norm_mean = Normalize(vmin=np.min(mean_feature), vmax=np.max(mean_feature))
        
        # Plot original feature map
        im1 = axes[0].imshow(mean_feature, cmap='viridis', norm=norm_mean, aspect='equal')
        axes[0].set_title('Original Feature Map (Mean)')
        plt.colorbar(im1, ax=axes[0])
        
        # If center and feature_size are provided and show_mask is True, draw a rectangle around the region of interest
        if center is not None and feature_size is not None and show_mask:
            rect = plt.Rectangle((center[0]-feature_size, center[1]-feature_size), 
                                2*feature_size, 2*feature_size, 
                                linewidth=1,  # Changed from 2 to 1 for thinner line
                                edgecolor='r', facecolor='none')
            axes[0].add_patch(rect)
        
        # If perturbed feature map is provided, plot it as well
        fig_diff = None
        if perturbed_feature_map is not None:
            # Check if shapes match
            if perturbed_feature_map.shape != feature_map.shape:
                logging.warning(f"Shape mismatch: original {feature_map.shape} vs perturbed {perturbed_feature_map.shape}")
                # Try to handle the case where shapes don't match
                if len(perturbed_feature_map.shape) == 3:
                    # Resize if possible
                    if perturbed_feature_map.shape[1:] != feature_map.shape[1:]:
                        logging.warning("Cannot create difference map due to spatial dimension mismatch")
                        mean_perturbed = np.mean(perturbed_feature_map, axis=0)
                    else:
                        # Only channel dimension differs
                        mean_perturbed = np.mean(perturbed_feature_map, axis=0)
                else:
                    logging.warning("Cannot visualize perturbed feature map with incompatible dimensions")
                    mean_perturbed = None
            else:
                mean_perturbed = np.mean(perturbed_feature_map, axis=0)
            
            if mean_perturbed is not None:
                im2 = axes[1].imshow(mean_perturbed, cmap='viridis', norm=norm_mean, aspect='equal')
                axes[1].set_title('Perturbed Feature Map (Mean)')
                plt.colorbar(im2, ax=axes[1])
                
                # Draw rectangle on perturbed feature map too if needed
                if center is not None and feature_size is not None and show_mask:
                    rect = plt.Rectangle((center[0]-feature_size, center[1]-feature_size), 
                                        2*feature_size, 2*feature_size, 
                                        linewidth=1,  # Changed from 2 to 1
                                        edgecolor='r', facecolor='none')
                    axes[1].add_patch(rect)
                
                # Create a new figure for the difference map
                fig_diff = plt.figure(figsize=(6, 6))
                ax_diff = fig_diff.add_subplot(111)
                
                # Calculate difference
                diff = mean_perturbed - mean_feature
                
                # Use a diverging colormap for difference (red-blue)
                im_diff = ax_diff.imshow(diff, cmap='coolwarm', aspect='equal')
                ax_diff.set_title('Difference Map (Perturbed - Original)')
                plt.colorbar(im_diff, ax=ax_diff)
                
                # Draw rectangle on difference map if needed
                if center is not None and feature_size is not None and show_mask:
                    rect = plt.Rectangle((center[0]-feature_size, center[1]-feature_size), 
                                        2*feature_size, 2*feature_size, 
                                        linewidth=1,  # Changed from 2 to 1
                                        edgecolor='r', facecolor='none')
                    ax_diff.add_patch(rect)
        
        # Adjust layout and save or show
        plt.tight_layout()
        
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save the figure
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
            # Save difference map if available
            if fig_diff is not None:
                diff_path = save_path.replace('.png', '_diff.png')
                fig_diff.savefig(diff_path, dpi=300, bbox_inches='tight')
                plt.close(fig_diff)
            
            plt.close(fig)
        else:
            plt.show()
            if fig_diff is not None:
                plt.show()  # Show the difference map
        
        return fig
