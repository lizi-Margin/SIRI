import time, numpy as np, cv2, gymnasium.spaces as spaces, copy, torch, random
import multiprocessing as mp
from imitation.utils import safe_load_traj_pool, safe_dump_traj_pool, print_dict, get_container_from_traj_pool, print_nan, check_nan
from UTIL.colorful import *
def get_data(traj_pool, NetActor):
    x_discretizer = NetActor.x_discretizer
    y_discretizer = NetActor.y_discretizer
    wasd_discretizer = NetActor.wasd_discretizer
    req_dict_name = ['key', 'mouse', 'FRAME_raw']
    for traj in traj_pool:
        for name in req_dict_name:
            check_nan(getattr(traj, name))
            # print(len(getattr(traj, name)))

    container = get_container_from_traj_pool(traj_pool, req_dict_name, req_dict_name)
    print_dict(container)

    container['FRAME_center'] = [NetActor.get_center(frame.copy()) for frame in container['FRAME_raw']]
    # frame = container['FRAME_center'][0]
    # print(frame.shape)
    # cv2.imshow('x',frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # frame_center = preprocess(container['FRAME_center'], train=True)
    frame_center = container['FRAME_center']
    act_wasd = container['key'][:, :4]
    index_jump = container['key'][:, 4]
    index_crouch = container['key'][:, 5]
    index_reload = container['key'][:, 14]
    index_r = container['key'][:, 17]
    index_l = container['key'][:, 16]


    # if old_dataset_legacy:
    #     act_mouse_x = container['mouse'][:, 0]   /2 # !!!
    #     act_mouse_y = container['mouse'][:, 1]   /2 # !!!
    # else:
    act_mouse_x = container['mouse'][:, 0]
    act_mouse_y = container['mouse'][:, 1]
    # print(np.max(act_mouse, axis=0))
    # print(np.min(act_mouse, axis=0))
    # print(np.mean(act_mouse, axis=0))
    # print(np.median(act_mouse, axis=0))
    # print(np.std(act_mouse, axis=0))


    index_mouse_x = x_discretizer.discretize(act_mouse_x)
    index_mouse_y = y_discretizer.discretize(act_mouse_y)
    index_wasd = wasd_discretizer.action_to_index(act_wasd)

    # index_xy_norm = np.concatenate([(index_mouse_x/12)[:, np.newaxis], (index_mouse_y/4)[:, np.newaxis]], axis=1)
    # print(index_xy_norm.shape, act_wasd.shape)
    # action_for_reward = np.concatenate([act_wasd[:, np.newaxis], index_xy_norm], axis=1)
    # actLogProbs = np.ones((len(action_for_reward),))

    # if isinstance(frame_center, (tuple, list,)):
    #     for i in range(len(frame_center)):
    #         frame_center[i] = frame_center[i].to('cpu')
    # else:
    #     frame_center = frame_center.to('cpu')

    data = {
        # 'obs': preprocess(frame_center), 
        'obs': frame_center, 
        # 'action_for_reward': action_for_reward,
        # 'actLogProbs': actLogProbs,

        'wasd': index_wasd,
        'x': index_mouse_x,
        'y': index_mouse_y,
        'jump': index_jump,
        'crouch': index_crouch,
        'reload': index_reload,
        'r': index_r,
        'l': index_l
    }

    return data

def data_loader_process(traj_dir, n_traj, queue, NetActor):
    print蓝("[data_loader_process] started")
    load = safe_load_traj_pool(traj_dir=traj_dir)
    while True:
        qsz = queue.qsize()
        while qsz >= 1:
            # print蓝(f"[data_loader_process] waiting, queue.qsize()={qsz}")
            time.sleep(1)
            qsz = queue.qsize()
        print蓝(f"[data_loader_process] start loading: {traj_dir}")
        pool = load(n_samples=n_traj)
        datas = []
        for traj in pool:
            datas.append(get_data([traj], NetActor))
        print蓝(f"[data_loader_process] load completed")
        queue.put_nowait((datas, traj_dir,)) 
        del pool

import os
from typing import Generator, Dict, Optional, List

class VideoDatasetLoader:
    """
    视频数据集加载器，自动扫描文件夹下的所有MP4文件
    提供按固定时间间隔获取帧的生成器接口
    每个视频的最后一帧会标记 done=True
    """
    
    def __init__(self, 
                 root_dir: str, 
                 interval_sec: float = 0.1,
                 extensions: List[str] = None,
                 shuffle: bool = False):
        """
        :param root_dir: 包含视频文件的根目录
        :param interval_sec: 采样间隔时间(秒)
        :param extensions: 支持的视频扩展名(默认['.mp4', '.MP4'])
        :param shuffle: 是否随机打乱视频顺序
        """
        self.root_dir = root_dir
        self.interval_sec = interval_sec
        self.extensions = extensions or ['.mp4', '.MP4']
        self.shuffle = shuffle
        
        # 扫描视频文件
        self.video_files = self._scan_video_files()
        if not self.video_files:
            raise FileNotFoundError(f"No video files found in {root_dir} with extensions {self.extensions}")
        
        # 当前状态
        self.current_video_index = -1  # 初始化为-1，表示尚未开始
    
    def _scan_video_files(self) -> List[str]:
        """扫描目录下的所有视频文件"""
        video_files = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if any(file.endswith(ext) for ext in self.extensions):
                    video_files.append(os.path.join(root, file))
        
        if self.shuffle:
            import random
            random.shuffle(video_files)
        
        return video_files
    
    def __len__(self) -> int:
        """返回视频文件总数"""
        return len(self.video_files)
    
    def __iter__(self) -> Generator[Dict, None, None]:
        """
        迭代器接口，返回生成器
        每次yield返回包含帧信息的字典:
        {
            'frame': np.ndarray,    # 帧图像
            'video_path': str,      # 当前视频路径
            'frame_count': int,     # 在当前视频中的帧计数
            'timestamp': float,     # 当前时间戳(秒)
            'video_index': int,     # 当前视频索引
            'done': bool           # 是否是当前视频的最后一帧
        }
        """
        for video_idx, video_path in enumerate(self.video_files):
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Warning: Could not open video {video_path}")
                continue
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                print(f"Warning: Invalid FPS in {video_path}")
                cap.release()
                continue
                
            frame_interval = max(1, int(round(fps * self.interval_sec)))
            frame_count = 0
            next_frame_to_capture = 0
            last_frame = None
            last_frame_info = None
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    # 视频结束，返回最后一帧（如果有）
                    if last_frame is not None:
                        last_frame_info['done'] = True
                        yield last_frame_info
                    break
                
                current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                
                if frame_count >= next_frame_to_capture:
                    frame_info = {
                        'frame': frame.copy(),
                        'video_path': video_path,
                        'frame_count': frame_count,
                        'timestamp': current_time,
                        'video_index': video_idx,
                        'done': False
                    }
                    
                    # 保存最后一帧信息（可能不是采样帧）
                    last_frame = frame
                    last_frame_info = frame_info.copy()
                    
                    yield frame_info
                    next_frame_to_capture += frame_interval
                
                frame_count += 1
            
            cap.release()
            
            # 如果整个视频都没有采样到任何帧（极短视频）
            if last_frame is None and frame_count > 0:
                # 强制返回第一帧作为最后一帧
                cap = cv2.VideoCapture(video_path)
                ret, frame = cap.read()
                if ret:
                    yield {
                        'frame': frame,
                        'video_path': video_path,
                        'frame_count': 0,
                        'timestamp': 0,
                        'video_index': video_idx,
                        'done': True
                    }
                cap.release()

# 使用示例
if __name__ == "__main__":
    # 初始化加载器
    loader = VideoDatasetLoader(
        root_dir="path/to/your/videos",
        interval_sec=0.1,  # 每0.1秒采样一帧
        shuffle=False
    )
    
    print(f"Found {len(loader)} video files")
    
    # 遍历所有视频的所有采样帧
    for frame_data in loader:
        frame = frame_data['frame']
        done_marker = " [LAST FRAME]" if frame_data['done'] else ""
        print(f"Video {frame_data['video_index']+1}/{len(loader)} "
              f"| Frame {frame_data['frame_count']} "
              f"| Time: {frame_data['timestamp']:.2f}s{done_marker}")
        
        # 显示帧 (按Q退出)
        cv2.imshow('Video Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()