import sys
import cv2
import random
import threading, time
import numpy as np
import supervision as sv
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QCoreApplication
from matplotlib import pyplot as plt

from siri.utils.sleeper import Sleeper
from siri.utils.logger import lprint, print_obj
from siri.global_config import GloablStatus
from siri.global_config import GlobalConfig as cfg
from siri.utils.img_window import ImageWindow
from siri.vision.preprocess import to_int, resize_image_to_width


def is_zero(list_like):
    for x in list_like:
        if x != 0:
            return False
    return True


class Visualizer(threading.Thread):
    def __init__(self):
        super().__init__()
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        # self.draw_mutex = threading.Semaphore(value=0)

        self.sv_source_queue: list[dict] = []
        self.obs_act_data: list[dict] = []

        self.plt = cfg.plt
        
        self.video_writer = None
        self.video_writer_fps = 1/cfg.tick
        
        self.last_sv_source = None
        self.last_obs_act = None
        self.last_annotated_frame = None
    
    def run(self):
        lprint(self, "start")
        title = f"{self.__class__.__name__}"
        try:
            while not GloablStatus.stop_event.is_set():
                # self.draw_mutex.acquire()

                sleeper = Sleeper(user=self)            
                if len(self.sv_source_queue) == 0:
                    sv_source = self.last_sv_source
                else:
                    sv_source = self.sv_source_queue.pop(0)
                    self.last_sv_source = sv_source
                
                if len(self.obs_act_data) == 0:
                    obs_act = self.last_obs_act
                else:
                    obs_act = self.obs_act_data.pop(0)
                    if obs_act is not None: self.last_obs_act = obs_act
                    else: obs_act = self.last_obs_act

                if sv_source is None:
                    if self.last_annotated_frame is not None:
                        annotated_frame = self.last_annotated_frame
                    else:
                        sleeper.sleep()
                        continue
                else:
                    annotated_frame = self.plot(sv_source, obs_act)
                    self.last_annotated_frame = annotated_frame

                if len(self.sv_source_queue) == 0 or len(self.obs_act_data) == 0:
                    if max(len(self.sv_source_queue), len(self.obs_act_data)) > 3:
                        self.sv_source_queue = []
                        self.obs_act_data = []

                if (self.video_writer is None) and ((sv_source is not None) and (obs_act is not None)):
                    save_video = f"{self.__class__.__name__}-{time.strftime("%Y%m%d-%H%M%S")}.mp4"
                    h, w, _ = annotated_frame.shape
                    self.frame_size = (w, h)
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self.video_writer = cv2.VideoWriter(save_video, fourcc, self.video_writer_fps, self.frame_size)
                    
                if self.video_writer is not None:
                    self.video_writer.write(annotated_frame)


                if self.plt == 'plt':
                    if not hasattr(self, 'img'):
                        self.fig, ax = plt.subplots()
                        ax.set_title(title)
                        self.img = ax.imshow(annotated_frame)
                        ax.axis('off')
                        plt.ion()
                        plt.show()
                    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    self.img.set_data(annotated_frame)
                    self.fig.canvas.draw()
                    self.fig.canvas.flush_events()
                elif self.plt == 'cv2':
                    cv2.imshow(title, annotated_frame)
                    cv2.waitKey(1)
                elif self.plt == 'qt':
                    if not hasattr(self, 'img'):
                        # BUG, WARNING: QApplication was not created in the main() thread.
                        self.app = QApplication(sys.argv)
                        self.img = ImageWindow(title=title)
                        self.img.show()
                    self.img.update_image(annotated_frame)
                    self.app.processEvents()
                else:
                    raise NotImplementedError()
                sleeper.sleep()
        except KeyboardInterrupt:
            lprint(self, "Sig INT catched, stopping session.")
        finally:
            if self.video_writer is not None:
                self.video_writer.release()
            self.sv_source_queue = []
            self.obs_act_data = []
            if hasattr(self, 'img'):
                # self.app.quit()
                # self.app.exit(0)
                app = QCoreApplication.instance()
                if app is not None:
                    app.quit()

        lprint(self, "finish")

    def draw_sv_source(self, sv_source: dict):
        lprint(self, "draw_sv_source called", debug=True)
        assert isinstance(sv_source, dict)
        self.sv_source_queue.append(sv_source)
    
    def draw_obs_act(self, data: dict):
        lprint(self, "draw_obs_act called", debug=True)
        # assert isinstance(data, dict) and data is not None
        # last_action_data = self.action_data; 
        self.obs_act_data.append(data)
        # if last_action_data is None:
        #     self.draw_mutex.release()
    
    def plot(self, sv_source, obs_act):
        assert isinstance(sv_source, dict)
        frame = sv_source['frame']
        deep_frame = sv_source['deep_frame']
        assert frame is not None

        MAX_WIDTH = 1200

        
        # draw supervision boxes
        if sv_source['detections'] is not None:
            # labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
            frame = self.box_annotator.annotate(
                scene=frame.copy(), detections=sv_source['detections'])
            frame = self.label_annotator.annotate(
                scene=frame, detections=sv_source['detections'])

        if obs_act is not None:
            assert isinstance(obs_act, dict)
            obs = obs_act['obs']; act = obs_act['act']
            target_xy = obs['xy']
            
            # draw target point
            if cfg.yolo_plt:
                target_xy = to_int(target_xy)
                tgt_dot_color = (30, 180, 20,) if not is_zero(target_xy) else (0, 0, 0)
                frame = cv2.circle(
                    img=frame,
                    center=target_xy,
                    radius=8,
                    color=tgt_dot_color,  # color BGR
                    thickness=-1
                )


            ## draw act
            # mouse move
            if not is_zero(act['mv_xy']):
                c_x, c_y = GloablStatus.in_window_center_xy()
                mv_x, mv_y = act['mv_xy']
                cv2.arrowedLine(
                    frame,
                    to_int((c_x, c_y,)),
                    to_int((c_x + mv_x, c_y + mv_y,)),
                    color=(50, 50, 200),
                    thickness=4
                )
            
            ## keys
            # wasd
            KEY_COLOR_UP = (200, 200, 200)  # 按键松开时的颜色
            KEY_COLOR_DOWN = (100, 100, 100)  # 按键按下时的颜色
            DOT_COLOR_UP = (200, 200, 200)  # 圆点松开时的颜色
            DOT_COLOR_DOWN = (100, 100, 255)  # 圆点按下时的颜色
            KEY_SIZE = 30  # 按键方块的大小
            DOT_SIZE = 24  # 圆点的半径
            start_x = 10
            start_y = GloablStatus.monitor['height'] - KEY_SIZE - 10
            keys = [('w', (start_x + KEY_SIZE, start_y - KEY_SIZE)),
                    ('a', (start_x, start_y)),
                    ('s', (start_x + KEY_SIZE, start_y)),
                    ('d', (start_x + 2 * KEY_SIZE, start_y))]
            for key, (x, y) in keys:
                color = KEY_COLOR_DOWN if act[key] else KEY_COLOR_UP
                cv2.rectangle(frame, (x, y), (x + KEY_SIZE, y + KEY_SIZE), color, -1)
                cv2.putText(frame, key.upper(), (x + 10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            dot_x = start_x + 3 * KEY_SIZE + 20
            fire_y = start_y
            scope_y = start_y - 2 * DOT_SIZE

            # 开火圆点
            fire_color = DOT_COLOR_DOWN if act['fire'] else DOT_COLOR_UP
            cv2.circle(frame, (dot_x, fire_y), DOT_SIZE, fire_color, -1)
            cv2.putText(frame, 'Fire', (dot_x - 20, fire_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # 开镜圆点
            scope_color = DOT_COLOR_DOWN if act['scope'] else DOT_COLOR_UP
            cv2.circle(frame, (dot_x, scope_y), DOT_SIZE, scope_color, -1)
            cv2.putText(frame, 'Scope', (dot_x - 20, scope_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


            # 4, 5, 6, g
            left_keys = ['4', '5', '6', 'g', 'reload', 'jump', 'crouch']
            left_start_x = 10
            left_start_y = start_y - len(left_keys) * KEY_SIZE
            for i, key in enumerate(left_keys):
                x = left_start_x
                y = left_start_y + i * (KEY_SIZE + 0)  # 间隔
                color = KEY_COLOR_DOWN if ((key in act) and act[key]) else KEY_COLOR_UP
                cv2.rectangle(frame, (x, y), (x + KEY_SIZE, y + KEY_SIZE), color, -1)
                cv2.putText(frame, key.upper(), (x + 10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # policy input
            if obs['policy_input'] is not None:
                policy_input = obs['policy_input']
                if isinstance(policy_input, np.ndarray):
                    policy_input = (policy_input,)
                else:
                    assert isinstance(policy_input, tuple)
                    assert len(policy_input) > 0
                    assert isinstance(policy_input[0], np.ndarray)
                policy_input_block = None
                for input_frame in policy_input:
                    if policy_input_block is None:
                        policy_input_block = input_frame
                        policy_input_block_width = policy_input_block.shape[1]
                    else:
                        # 上下合并
                        input_frame_with_bg = self.embed_image_in_black_bg_width(input_frame, policy_input_block_width)
                        policy_input_block = np.vstack((policy_input_block, input_frame_with_bg))
                
                frame_height, frame_width = frame.shape[:2]
                policy_input_block_height, policy_input_block_width = policy_input_block.shape[:2]

                if frame_height > policy_input_block_height:
                    policy_input_block = self.embed_image_in_black_bg(policy_input_block, frame_height)
                else:
                    frame = self.embed_image_in_black_bg(frame, policy_input_block_height)

                # 左右拼接两个图像
                combined_frame = np.hstack((frame, policy_input_block))

                
                if combined_frame.shape[1] > MAX_WIDTH:
                    combined_frame = resize_image_to_width(combined_frame, MAX_WIDTH)
                frame = combined_frame
                

        if deep_frame is not None:

            if len(deep_frame.shape) == 2:
                # 归一化到 [0, 255]
                depth_min = deep_frame.min()
                depth_max = deep_frame.max()
                if depth_max > depth_min:
                    deep_frame = 255 * (deep_frame - depth_min) / (depth_max - depth_min)
                deep_frame = np.uint8(deep_frame)
                # deep_frame = cv2.cvtColor(deep_frame, cv2.COLOR_GRAY2BGR)
                deep_frame = cv2.applyColorMap(deep_frame, cv2.COLORMAP_INFERNO)

            # 获取两个图像的高度和宽度
            frame_height, frame_width = frame.shape[:2]
            deep_frame_height, deep_frame_width = deep_frame.shape[:2]

            # 确保两个图像高度一致
            if frame_height > deep_frame_height:
                deep_frame = self.embed_image_in_black_bg(deep_frame, frame_height)
            else:
                frame = self.embed_image_in_black_bg(frame, deep_frame_height)

            # 左右拼接两个图像
            combined_frame = np.hstack((frame, deep_frame))

            if combined_frame.shape[1] > MAX_WIDTH:
                combined_frame = resize_image_to_width(combined_frame, MAX_WIDTH)
            frame = combined_frame
        return frame

    def embed_image_in_black_bg(self, small_img, target_height):
        """
        将小图像嵌入到黑色背景中，使其高度与目标高度一致
        :param small_img: 小图像
        :param target_height: 目标高度
        :return: 嵌入后的图像
        """
        small_height, small_width = small_img.shape[:2]
        # 创建一个黑色背景
        black_bg = np.zeros((target_height, small_width, 3), dtype=np.uint8)
        # 计算嵌入位置
        start_y = (target_height - small_height) // 2

        # 检查小图像是否为单通道
        if len(small_img.shape) == 2:
            # 将单通道图像转换为三通道图像
            small_img = cv2.cvtColor(small_img, cv2.COLOR_GRAY2BGR)

        # 将小图像嵌入到黑色背景中
        black_bg[start_y:start_y + small_height, :] = small_img
        return black_bg
    
    def embed_image_in_black_bg_width(self, small_img, target_width):
        small_height, small_width = small_img.shape[:2]
        black_bg = np.zeros((small_height, target_width, 3), dtype=np.uint8)
        start_x = (target_width - small_width) // 2

        # 检查小图像是否为单通道
        if len(small_img.shape) == 2:
            # 将单通道图像转换为三通道图像
            small_img = cv2.cvtColor(small_img, cv2.COLOR_GRAY2BGR)

        black_bg[:, start_x:start_x + small_width] = small_img
        return black_bg