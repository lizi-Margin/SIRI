import os
import cv2
import mss
import time
import torch
import random
import threading
import subprocess
import pytesseract
import numpy as np
import supervision as sv
import torchvision.transforms as transforms
from PIL import Image
from typing import Union, List


from siri.global_config import GlobalConfig as cfg
from siri.global_config import GloablStatus
from siri.vision.preprocess import preprocess, postprocess, pre_transform, pre_transform_crop, to_int, pre_transform_pad, pre_transform_crop_left_right
from siri.utils.logger import lprint
from siri.utils.sleeper import Sleeper


class ObsMaker:
    def __init__(self):
        pass

    def __call__(self, *args, **kwds):
        return self.make_obs(*args, **kwds)

    def make_obs(self, detections):
        if isinstance(detections, sv.Detections):
            if detections.xyxy.any():
                boxes_array, classes_tensor = self.convert_sv_to_tensor(detections)
                if classes_tensor.numel():
                    target = self.find_best_target(boxes_array, classes_tensor)
                
                    clss = [0., 1., 
                            0 , 1  ]
                    # if cfg.hideout_targets:
                    #     clss.extend([5.0, 6.0])
                    # if not cfg.disable_headshot:
                    #     clss.append(7.0)
                    # if cfg.third_person:
                    #     clss.append(10.0)
                    
                    if target.cls in clss:           
                        x, y = target.x, target.y
                        # x, y = predict_xy(x, y)
                        obs = {
                            'xy': (x, y,),
                            'cls': target.cls
                        }
                        return obs

            return None
        elif isinstance(detections, tuple):
            boxes_array, classes_tensor = detections
            assert isinstance(boxes_array, torch.Tensor)
            assert isinstance(classes_tensor, torch.Tensor)
            target = self.find_best_target(boxes_array, classes_tensor)
            if target:
                obs = {
                    'xy': (target.x, target.y,),
                    'wh': (target.w, target.h,),
                    'cls': target.cls
                }
                return obs
            else:
                return None
        else:
            assert False, 'WTF?'
    
    def find_best_target(self, boxes_array, classes_tensor):
        return self.find_nearest_target_to(
            GloablStatus.in_window_center_xy(),
            boxes_array,
            classes_tensor
        )
    
    def find_nearest_target_to(self, xy:tuple, boxes_array, classes_tensor,):
        center = torch.tensor([xy[0], xy[1]], device=cfg.device)
        distances_sq = torch.sum((boxes_array[:, :2] - center) ** 2, dim=1)
        # weights = torch.ones_like(distances_sq)

        # head_mask = classes_tensor == 7
        # if head_mask.any():
        #     nearest_idx = torch.argmin(distances_sq[head_mask])
        #     nearest_idx = torch.nonzero(head_mask)[nearest_idx].item()
        # else:
        nearest_idx = torch.argmin(distances_sq)

        target_data = boxes_array[nearest_idx, :4].cpu().numpy()
        target_class = classes_tensor[nearest_idx].item()
        """
        names:
            0: player
            1: bot
            2: weapon
            3: outline
            4: dead_body
            5: hideout_target_human
            6: hideout_target_balls
            7: head
            8: smoke
            9: fire
            10: third_person
        """
        return Target(*target_data, target_class)

    @staticmethod
    def convert_sv_to_tensor(frame: sv.Detections):
        assert frame.xyxy.any()
        xyxy = frame.xyxy
        xywh = torch.tensor(np.array(
            [(xyxy[:, 0] + xyxy[:, 2]) / 2,  
             (xyxy[:, 1] + xyxy[:, 3]) / 2,  
             xyxy[:, 2] - xyxy[:, 0],        
             xyxy[:, 3] - xyxy[:, 1]]
        ), dtype=torch.float32).to(cfg.device).T
        
        classes_tensor = torch.from_numpy(np.array(frame.class_id, dtype=np.float32)).to(cfg.device)
        return xywh, classes_tensor


class Target:
    def __init__(self, x, y, w, h, cls):
        self.x = x
        self.y = y if cls == 7 else (y - cfg.body_y_offset * h)
        self.w = w
        self.h = h
        self.cls = cls

class FakeDeepPredictor:
    def predict(self, frame_or_batch: Union[np.ndarray, List[np.ndarray]]):
        deep_obs = {'f': 0, 'l': 0, 'r': 0}
        deep_frame_shape = frame_or_batch[0].shape[:2]
        return deep_obs, (np.ones(deep_frame_shape, dtype=np.float32) * 100)
class DeepPredictor:
    def __init__(self):
        # 加载 MiDaS 模型
        # model_type = "MiDaS_small"
        # local_model_path = './midas_v21_small_256.pt'
        # self.transform = midas_transforms.small_transform
        # self.model_sz = (256, 256,)
        # assert os.path.exists(local_model_path)
        # self.model = torch.hub.load('./MiDaS', model_type, source='local', pretrained=False)
        # self.model.load_state_dict(torch.load(local_model_path))

        from siri.vision.midas import swin_256, dpt_hybrid_384, midas_small_256, swin_l_384
        # self.model, self.transform, self.model_sz = swin_256()
        # self.model, self.transform, self.model_sz = dpt_hybrid_384()
        self.model, self.transform, self.model_sz = midas_small_256()
        # self.model, self.transform, self.model_sz = swin_l_384()


        self.device = cfg.device
        self.model.to(self.device)
        self.model.eval()
        

    def _predict(self, frame_or_batch: Union[np.ndarray, List[np.ndarray]]):
        if isinstance(frame_or_batch, np.ndarray):
            batch = [frame_or_batch]
        elif isinstance(frame_or_batch, list):
            assert isinstance(frame_or_batch[0], np.ndarray)
            batch = frame_or_batch
        else:
            assert False

        assert len(batch[0].shape) == 3

        # 仅使用 pre_transform 进行大小变换
        # batch = pre_transform(batch, self.model_sz)
        batch = pre_transform_crop_left_right(batch, 0.3)
        batch = pre_transform_pad(batch, self.model_sz, white_bg=True)
        # batch = pre_transform_crop(batch)

        depth_maps = []
        for frame in batch:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = self.transform(frame).to(self.device)

            with torch.no_grad():
                prediction = self.model(input_tensor)
                # prediction = torch.nn.functional.interpolate(
                #     prediction.unsqueeze(1),
                #     size=frame.shape[:2],
                #     mode='bicubic',
                #     align_corners=False
                # )
                prediction = prediction.squeeze().cpu().numpy()
                # .astype(np.uint8)
            depth_maps.append(prediction)

        return depth_maps
    
    def predict(self, frame_or_batch: Union[np.ndarray, List[np.ndarray]]):
        deep_frame = self._predict(frame_or_batch)[0]
        return self.process_depth_frame(deep_frame)
        

    # def process_depth_frame_minmax(self, deep_frame: np.ndarray):
    #     """
    #     使用 Min-Max 归一化处理 MiDaS 深度图，决定智能体的移动方向。
    #     :param deep_frame: MiDaS 输出的深度图 (numpy float 数组，正方形)
    #     :return: 'left', 'right' 或 'forward'
    #     """
    #     # 标准化深度值到 [0, 1]
    #     deep_frame -= np.min(deep_frame)
    #     deep_frame /= np.max(deep_frame)
        
    #     return self._process_depth_frame_common(deep_frame)

    def process_depth_frame(self, deep_frame: np.ndarray):
        """
        使用 Z-score 标准化处理 MiDaS 深度图，决定智能体的移动方向。
        :param deep_frame: MiDaS 输出的深度图 (numpy float 数组，正方形)
        :return: 深度信息字典和带标记的深度图
        """
        min_ = np.min(deep_frame)
        max_ = np.max(deep_frame)
        mean = np.mean(deep_frame)
        std = np.std(deep_frame)
        deep_frame_z = (deep_frame.copy() - mean) / (std + 1e-6)  # 避免除零
        
        dp_h, dp_w = deep_frame.shape
        center_x = dp_w // 2
        center_y = dp_h // 2
        
        # 采样区域参数
        num_regions = 7  # 检测框
        margin_x = 0  # 两侧的空隙
        shift_y = dp_h // 15  # 可调高度
        region_width = (dp_w - 2 * margin_x) // num_regions  # 每个区域的宽度
        region_height = dp_h // 12  # 采样区域的高度
        circula_coef = 5
        
        depth_values = []
        deep_obs = {'f': 0, 'l': 0, 'r': 0}
        
        for i in range(num_regions):
            x_start = margin_x + i * region_width
            x_end = x_start + region_width
            y_start = center_y - region_height // 2 + shift_y + int(abs(i + 1 - (num_regions+1)/2) * circula_coef)
            y_end = center_y + region_height // 2 + shift_y + int(abs(i + 1 - (num_regions+1)/2) * circula_coef)
            
            # 计算区域深度
            region_depth_z = deep_frame_z[y_start:y_end, x_start:x_end]
            region_depth = deep_frame[y_start:y_end, x_start:x_end]
            avg_depth = np.mean(region_depth)
            depth_values.append(avg_depth)
            
            # 画检测框
            deep_frame = cv2.rectangle(deep_frame, (x_start, y_start), (x_end, y_end), 0.5, thickness=3)
            deep_frame = cv2.putText(deep_frame, str(int(avg_depth)), 
                                    (x_start, y_start - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, float(max_), thickness=1)
        
        # 计算深度信息
        min_depth = min(depth_values)
        min_index = depth_values.index(min_depth)


        x_start = margin_x + min_index * region_width
        x_end = x_start + region_width
        y_start = center_y - region_height // 2 + shift_y + int(abs(min_index + 1 - (num_regions+1)/2) * circula_coef)
        y_end = center_y + region_height // 2 + shift_y + int(abs(min_index + 1 - (num_regions+1)/2) * circula_coef)
        deep_frame = cv2.rectangle(deep_frame, (x_start, y_start), (x_end, y_end), float(max_), thickness=-1)

        depth_values = np.array(depth_values)  # 转换为 numpy 数组
        # # 先去掉最高值和最低值
        # if len(depth_values) > 2:  # 确保至少有 3 个数值，否则无法去掉
        #     depth_values = np.delete(depth_values, [np.argmax(depth_values), np.argmin(depth_values)])
        # else:
        #     depth_values = depth_values  # 如果元素过少，直接使用原数据
        depth_std = np.std(depth_values)
        
        if std > 100:
            min_num = min_index + 1
            if min_num < (num_regions + 1)/2:
                deep_obs['l'] = 1
            elif min_num > (num_regions+1)/2:
                deep_obs['r'] = 1
            else:
                deep_obs['f'] = 1
        else:
            deep_obs['r'] = 1
        
        return deep_obs, deep_frame


class Detector(threading.Thread):
    def __init__(self, model, obs_hook=None, sv_source_hook=None):
        super().__init__()
        from ultralytics import YOLO
        assert isinstance(model, YOLO)
        self.model = model
        # self.deep_model = DeepPredictor()
        self.deep_model = FakeDeepPredictor()
        self.tracker = sv.ByteTrack()
        self.make_obs = ObsMaker()
        # self.feature_detector = HealthBarFeatureDetector(
        #     # enemy_template_path="enemy_bar.png",
        #     # friendly_template_path="friendly_bar.png"
        #     enemy_template_path="enemy_bar_raw.png",
        #     friendly_template_path="friendly_bar_raw.png"
        # )


        # self.lower_enemy = np.clip(np.array([38, 35, 149]) - 30, 0, 255)
        self.lower_enemy = np.clip(np.array([98, 93, 238]) - 30, 0, 255)
        self.upper_enemy = np.clip(np.array([98, 93, 238]) + 30, 0, 255)
        self.lower_friendly = np.array([0, 0, 240])  
        self.upper_friendly = np.array([180, 30, 255])  


        self.scope_bt_x, self.scope_bt_y = (94, 94,)
        self.scope_bt_r = 161//2
        self.scope_bt_color_lb = np.clip(np.array([38, 35, 149]) - 20, 0, 255)
        self.scope_bt_color_ub = np.clip(np.array([78, 73, 188]) + 20, 0, 255)
        self.scope_bt_color_mean = np.mean([self.scope_bt_color_lb, self.scope_bt_color_ub], axis=0)

        self.obs_hook = obs_hook
        self.sv_source_hook = sv_source_hook



    
    def run(self):
        lprint(self, "start")
        self.start_session()
        lprint(self, "finish")
    
    def start_session(self):
        raise NotImplementedError

    def _predict(self, frame_or_batch: Union[np.ndarray, List[np.ndarray]]):
        if isinstance(frame_or_batch, np.ndarray):
            batch = [frame_or_batch]
        elif isinstance(frame_or_batch, list):
            assert isinstance(frame_or_batch[0], np.ndarray)
            batch = frame_or_batch
        else:
            assert False

        assert len(batch[0].shape) == 3
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        # frame = pad(frame, to_sz_wh=cfg.sz_wh)
        # frame = cv2.resize(frame, cfg.sz_wh)
        if cfg.manual_preprocess: 
            batch = preprocess(batch)

        results = self.model.predict(
            batch,
            cfg=f"{cfg.root_dir}/siri/vision/game.yaml",
            imgsz=tuple(reversed(cfg.sz_wh)),
            stream=True,
            conf=cfg.conf_threshold,
            iou=0.5,
            device=cfg.device,
            half=cfg.half,
            max_det=20,
            agnostic_nms=False,
            augment=False,
            vid_stride=False,
            visualize=False,
            verbose=False,
            show_boxes=False,
            show_labels=False,
            show_conf=False,
            save=False,
            show=False,
            # batch=1
        )

        return results

    def detect_health_bar(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        x_l, y_u, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
        x_center, y_center = int((x1 + x2)/2), int((y1 + y2)/2)
        # y_top = max(min(y_u - 40, y_u - int(h * 0.3)), 0)
        # x_roi_l = max(x_l - 40, 0)
        # x_roi_r = min(x_l + w + 40, frame.shape[1])
        # y_bottom = max(y_u + int(h * 0.05), 1)
        y_top = max(y_u - 60, 0)
        x_roi_l = max(x_center - 50, 0)
        x_roi_r = min(x_center + 50, frame.shape[1])
        y_bottom = max(y_u, 1)

        if y_top > y_bottom or x_roi_l > x_roi_r:
            print(f"AssertionError: {y_top} {y_bottom} {x_roi_l} {x_roi_r}")
            return True, frame

        roi = frame[y_top:y_bottom, x_roi_l:x_roi_r]
        if cfg.yolo_plt: frame = cv2.rectangle(frame, (x_roi_l, y_top), (x_roi_r, y_bottom), (255, 20, 20), thickness=3)


        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        

        mask_enemy = cv2.inRange(hsv, self.lower_enemy, self.upper_enemy)
        # # 分离 RGB 通道
        # b, g, r = cv2.split(frame)

        # # 检查 RGB 值是否在 128 - 155 范围内
        # in_range = (r >= 128) & (r <= 155) & (g >= 128) & (g <= 155) & (b >= 128) & (b <= 155)

        # # 检查 RGB 通道值之间的差值是否不大于 40
        # diff_condition = (np.abs(r - g) <= 40) & (np.abs(r - b) <= 40) & (np.abs(g - b) <= 40)

        # # 合并两个条件
        # final_condition = in_range & diff_condition

        # # 生成掩码
        # mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        # mask[final_condition] = 255
        mask_friendly = cv2.inRange(hsv, self.lower_friendly, self.upper_friendly)

        enemy_pixels = cv2.countNonZero(mask_enemy)
        friendly_pixels = cv2.countNonZero(mask_friendly)

        # print(friendly_pixels, enemy_pixels)
        is_enm = True
        if friendly_pixels > enemy_pixels:
            if friendly_pixels > 100:
                is_enm = False


        if cfg.yolo_plt:
            mask_visualization = np.ones_like(frame, dtype=np.uint8) * 255
            selected_mask = mask_enemy if is_enm else mask_friendly
            mask_visualization[y_top:y_bottom, x_roi_l:x_roi_r] = cv2.merge([selected_mask, selected_mask, selected_mask])
            frame = cv2.bitwise_and(frame, mask_visualization)


        # label = "Enemy" if is_enm else "Friendly"
        wh_str = f"w:{w} h:{h}"
        color = (20, 20, 255) if is_enm else (20, 255, 20)

        
        if cfg.yolo_plt:
            frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=3)
            # frame = cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, thickness=3)
            frame = cv2.putText(frame, wh_str, (int(x2), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=1)

        return is_enm, frame
    
    def get_ammo_text(self, gray):
        # get text
        text = pytesseract.image_to_string(gray, config='--psm 7 --oem 1 -c tessedit_char_whitelist=0123456789/')
        # text = pytesseract.image_to_string(gray, config='--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789/')
        text = ''.join(c for c in text if c.isdigit() or c == '/').strip()
        if '/' in text:
            tmp = text.split('/')
            in_mag_ammo = int(tmp[0]) if tmp[0] else 0
            all_ammo = int(tmp[1]) if tmp[1] else 0
        else:
            # text to int
            in_mag_ammo = int(text) if text else 0
            all_ammo = in_mag_ammo
        return in_mag_ammo, all_ammo

    def detect_ammo(self, frame):
        h, w, _ = frame.shape
        left_trim = 70
        ammo_feild = frame[h-67:h-49, w//2-left_trim: w//2-22]
        gray = cv2.cvtColor(ammo_feild, cv2.COLOR_BGR2GRAY)

        gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
        # _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # _, gray = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
    
        if hasattr(self, "last_ammo"):
            now_time = time.time()
            if now_time - self.last_ammo_time > 3.:
                self.last_ammo_time = now_time
                self.last_ammo = self.get_ammo_text(gray)
        else:
            self.last_ammo_time = time.time()
            self.last_ammo = self.get_ammo_text(gray)
        in_mag_ammo, all_ammo = self.last_ammo
        text = f"{in_mag_ammo}/{all_ammo}"

        # print text to frame
        if text:
            cv2.putText(frame, text, (w//2-left_trim, h-67), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.rectangle(frame, (w//2-left_trim, h-67), (w//2-22, h-49), (0, 255, 0), 1)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.imwrite('gray.png', gray)
        frame[h-67:h-49, w//2-left_trim: w//2-22] = gray
        return (in_mag_ammo, all_ammo,), frame

 

    
    def circle_detect_(self, frame, x, y, r, lb, ub, thresh, visual_color=None):
        roi = frame[max(0, y-r):y+r, max(0, x-r):x+r]
        assert not roi.size == 0
        mask = cv2.inRange(roi, lb, ub)
        pixels = cv2.countNonZero(mask)
        in_ = False
        if pixels > thresh:  # about 15000 at usual
            in_ = True
        if visual_color is not None: frame = cv2.circle(frame, (x, y,), r, visual_color, thickness=7)
        return in_, frame
        

    # def predict_and_plot(self, frame):
    #     results = self._predict(frame)
    #     for result in results:
    #         result_frame = result.plot()
    #         cv2.imshow("tmp", result_frame)

    #         while True:
    #             time.sleep(0.1)
    #             if cv2.waitKey(1) & 0xFF == ord('q'):
    #                 break

    #     cv2.destroyAllWindows()

    
    def predict_and_make_obs(self, frame):
        frame_original = frame.copy()
        results = self._predict(frame)
        # frame = frame.copy()
        # print(results)
        if isinstance(results, list):
            if len(results) > 0:
                result = results[0]
            else:
                lprint(self, "no result were returned by the model")
                return
        else:
            try:
                result = next(results)
            except StopIteration:
                lprint(self, "no result were returned by the model")
                return
    
        deep_obs, deep_frame = self.deep_model.predict(frame)

        
        sv_detections = sv.Detections.from_ultralytics(result)
        sv_detections = self.tracker.update_with_detections(sv_detections)

        enemy_boxes = []
        classes_tensor = []
        annotated = False
        for i, (x1, y1, x2, y2) in enumerate(sv_detections.xyxy):
            if sv_detections.class_id[i] == 0:
                is_enm, frame = self.detect_health_bar(frame, (x1, y1, x2, y2,))
                if is_enm:
                    enemy_boxes.append(to_int([(x1 + x2)/2, (y1 + y2)/2, x2-x1, y2-y1]))  # x_center, y_center, w, h
                    classes_tensor.append(sv_detections.class_id[i])
                annotated = True
            elif sv_detections.class_id[i] == 7:
                pass
        

        
        in_scope, frame = self.circle_detect_(frame, self.scope_bt_x, self.scope_bt_y, self.scope_bt_r, self.scope_bt_color_lb, self.scope_bt_color_ub, 10000, visual_color=self.scope_bt_color_mean)
        ammo, frame = self.detect_ammo(frame)
        

        obs = {
            'in_scope': in_scope,
            'ammo': ammo,
            'frame': frame_original.copy(),
            'deep_frame': deep_frame.copy()
        }
        obs.update(deep_obs)
        if len(enemy_boxes) > 0:
            boxes_tensor = torch.tensor(enemy_boxes, dtype=torch.float32, device=cfg.device)
            classes_tensor = torch.tensor(classes_tensor, dtype=torch.float32, device=cfg.device)
            obs.update(self.make_obs((boxes_tensor, classes_tensor,)))

        if self.obs_hook is not None:
            self.obs_hook(obs)
            

        if self.sv_source_hook is not None:
            self.sv_source_hook({'frame': frame.copy(),
                                #  'deep_frame': deep_frame.copy(),
                                 'deep_frame': None,
                                 'detections': None if annotated else sv_detections})





class ScrGrabber():
    @staticmethod
    def get_scrcpy_window_geometry(window_keyword='Phone'):
        result = subprocess.run(
            ['wmctrl', '-lG'],
            stdout=subprocess.PIPE,
            text=True
        )
        lines = result.stdout.splitlines()
        for line in lines:
            if window_keyword in line:
                parts = line.split()
                x, y = int(parts[2]), int(parts[3])
                scr_width, scr_height = int(parts[4]), int(parts[5])
                return x, y, scr_width, scr_height
        return None

    @staticmethod
    def sync_monitor(geometry):
        left, top, scr_width, scr_height = geometry
        GloablStatus.monitor = {
            "top": top,
            "left": left,
            "width": scr_width,
            "height": scr_height,
        }
        lprint(ScrGrabber, f"sync_monitor: {GloablStatus.monitor}")
    
    def sync_monitor_every_n_step(self, n=100):
        if hasattr(self, "sync_monitor_counter"):
            self.sync_monitor_counter += 1
        else:
            self.sync_monitor_counter = 0
        if self.sync_monitor_counter % n == 0:
            geometry = self.get_scrcpy_window_geometry()
            if not geometry:
                lprint(self, "scrcpy window not found")
            else:
                self.sync_monitor(geometry)
            self.sync_monitor_counter = 0

    def start_session(self, func, *args, **kwargs):
        geometry = self.get_scrcpy_window_geometry()
        if not geometry:
            lprint(self, "scrcpy window not found")
            lprint(self, "start_session failed")
            return

        assert GloablStatus.monitor is None
        self.sync_monitor(geometry)
        

        try:
            with mss.mss() as sct:
                while True:
                    sleeper = Sleeper(tick=cfg.tick, user=self)
                    screenshot = sct.grab(GloablStatus.monitor)
                    frame = np.array(screenshot)
                    if frame.shape[-1] == 4:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    assert frame.shape[-1] == 3, 'not a BGR format'
                    frame = cv2.resize(frame, cfg.sz_wh)

                    func(frame, *args, **kwargs)

                    self.sync_monitor_every_n_step(n=100)
                    sleeper.sleep()
        except KeyboardInterrupt:
            lprint(self, "Sig INT catched, stopping session.")
        finally:
            # cv2.destroyAllWindows()
            GloablStatus.monitor = None
            GloablStatus.stop_event.set()
    
    def save_frame(self, frame: np.ndarray):
        assert isinstance(frame, np.ndarray)

        save_dir = f"{cfg.root_dir}/{self.__class__.__name__}"
        os.makedirs(save_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(save_dir, f"frame_{timestamp}.png")
        cv2.imwrite(filename, frame)
    
    def start_capture_session(self):
        self.start_session(func=self.save_frame)


class ScrDetector(Detector, ScrGrabber):
    def start_session(self):
        ScrGrabber.start_session(self, func=self.predict_and_make_obs)

