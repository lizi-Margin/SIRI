import os,time
import sys
import json
import pprint
import pickle
import numpy as np
from random import sample
from typing import Union, List, Tuple, Dict
from UTIL.colorful import *
class cfg:
    logdir = './HMP_IL/'

def print_dict(data):
    summary = {
        key: f" {type(value)}, shape={value.shape}, dtype={value.dtype}" if isinstance(value, np.ndarray) 
                                                                         else type(value) 
                                                                         for key, value in data.items()
    }
    pprint.pp(summary)


def print_list(data):
    assert isinstance(data, list)
    print("list len: ", len(data))
    # item_len = []
    # for item in data: item_len.append(len(item))
    # print(item_len)
    print("[", end="")
    for index, item in enumerate(data): 
        print(f" {index}: ", end="")
        print_dict(vars(item))
    print("]")


def is_basic_type(obj):
    if (
        isinstance(obj, int) or isinstance(obj, bool) or isinstance(obj, str) 
        or isinstance(obj, list) or isinstance(obj, tuple) or isinstance(obj, dict)
    ):
        return True
    else:
        return False

def safe_dump(obj, path):
    if not os.path.exists(path): os.makedirs(path)
    cls_name = obj.__class__.__name__
    serializable_data = {}
    numpy_arrays = {}
    for attr, value in obj.__dict__.items():
        if isinstance(value, np.ndarray):
            numpy_arrays[attr] = value
        elif is_basic_type(value):
            serializable_data[attr] = value
        else:
            assert False, 'not implemented yet'
    npy_filenames = {}
    for key, array in numpy_arrays.items():
        npy_filename = f"{cls_name}_{key}.npy"
        np.save(f"{path}/{npy_filename}", array, allow_pickle=True)
        npy_filenames[key] = npy_filename
    serializable_data['npy_filenames'] = npy_filenames
    with open(f"{path}/{cls_name}.json", 'w') as f:
        json.dump(serializable_data, f)


def safe_load(obj, path):
    if not os.path.exists(path):
        print亮黄(f"warning: {path} not found, skip loading")
        return obj

    cls_name = obj.__class__.__name__
    serializable_data = {}
    numpy_arrays = {}
    with open(f"{path}/{cls_name}.json", 'r') as f:
        serializable_data = json.load(f)
    assert isinstance(serializable_data, dict)

    npy_filenames = serializable_data.pop('npy_filenames')
    for key, npy_filename in npy_filenames.items():
        numpy_arrays[key] = np.load(f"{path}/{npy_filename}", allow_pickle=True)

    for attr, value in serializable_data.items():
        setattr(obj, attr, value)
    for attr, value in numpy_arrays.items():
        setattr(obj, attr, value)
    
    return obj


def safe_dump_traj_pool(traj_pool, pool_name):
    default_traj_dir = f"{cfg.logdir}/traj_pool_safe/"
    if os.path.islink(default_traj_dir):
        os.unlink(default_traj_dir)
    traj_dir = f"{cfg.logdir}/{time.strftime("%Y%m%d-%H:%M:%S")}/"
    
    for index, traj in enumerate(traj_pool):
        traj_name = f"traj-{pool_name}-{index}.d"
        safe_dump(obj=traj, path=f"{traj_dir}/{traj_name}")
    
        print亮黄(f"traj saved in file: {traj_dir}/{traj_name}")
    os.symlink(os.path.abspath(traj_dir), os.path.abspath(default_traj_dir))


class safe_load_traj_pool:
    def __init__(self, max_len=None):
        self.traj_dir = f"{cfg.logdir}/traj_pool_safe/"
        self.traj_names = os.listdir(self.traj_dir)
        if max_len is not None:
            assert max_len > 0
            if max_len < len(self.traj_names):
                self.traj_names = self.traj_names[:max_len]
    
    def __call__(self, pool_name='', n_samples=200):
        from .inputs import trajectory
        traj_pool = []
        if len(self.traj_names) > n_samples:
            n_samples = max(n_samples, 0)
            self.traj_names = sample(self.traj_names, n_samples)
        
        for i, traj_name in enumerate(self.traj_names):
            if traj_name.startswith(f"traj-{pool_name}"):
                print(traj_name)

                traj = safe_load(
                    obj=trajectory(traj_limit=int(cfg.ScenarioConfig.MaxEpisodeStep), env_id=i),
                    path=f"{self.traj_dir}/{traj_name}"
                )
                traj_pool.append(traj)

                # print亮黄(f"traj loaded from file: {traj_dir}/{traj_name}")
        
        print(f"safe loaded {len(traj_pool)} trajs")
        return traj_pool


def load_all_traj_pool(dir=None, idx:list=None):
    """
        Load all trj_pool in traj_pool_dir for behavior cloning.
    """
    MAX_TRAJ_POOL_NUM = 10
    if idx is None: idx = [ _ for _ in range(MAX_TRAJ_POOL_NUM) ]
    traj_pool_dir = f"{cfg.logdir}/traj_pool/" if dir is None else dir
    if not os.path.exists(traj_pool_dir): raise FileNotFoundError("traj_pool_dir not found")

    new_traj_pool = []
    traj_pool_file_path = lambda cnt: f"{traj_pool_dir}/traj_pool-{cnt}.pkl"
    for id in idx:
        if os.path.exists(traj_pool_file_path(id)):
            with open(traj_pool_file_path(id), "rb") as pkl_file:
                from .inputs import trajectory
                import ALGORITHM.ppo_ma_with_mask_predict_test_bc_daggr
                sys.modules["ALGORITHM.ppo_ma_with_mask_test_bc"] = ALGORITHM.ppo_ma_with_mask_predict_test_bc_daggr
                traj_pool_slice = pickle.load(pkl_file)
                assert isinstance(traj_pool_slice, list)
                # print(f"traj_pool_file loaded : {traj_pool_file_path(id)}")
            new_traj_pool += traj_pool_slice 
        else:
            print亮黄(f"warning: {traj_pool_file_path(idx)} not found, skip loading")

    return new_traj_pool if len(new_traj_pool) > 0 else None


def load_traj_pool(cnt, dir=None):
    """
        Load one trj_pool for behavior cloning.
    """ 
    traj_pool_dir = f"{cfg.logdir}/traj_pool/" if dir is None else dir
    traj_pool_file_path = lambda cnt_: f"{traj_pool_dir}/traj_pool-{cnt_}.pkl"
    if not os.path.exists(traj_pool_file_path(cnt)):
        # print亮黄(f"warning: {traj_pool_file_path(cnt)} not found, skip loading")
        return None

    with open(traj_pool_file_path(cnt), "rb") as pkl_file:
        from .inputs import trajectory
        traj_pool_slice = pickle.load(pkl_file)
        assert isinstance(traj_pool_slice, list[trajectory])

    # print亮黄(f"traj_pool_file loaded : {traj_pool_file_path(cnt)}")

    return traj_pool_slice


def load_container(cnt, dir=None):
    """
        Load one container for behavior cloning.
    """ 
    container_dir = f"{cfg.logdir}/traj_container/" if dir is None else dir
    container_file_path = lambda cnt_: f"{container_dir}/container-{cnt_}.pkl"
    if not os.path.exists(container_file_path(cnt)):
        print(f"warning: {container_file_path(cnt)} not found, skip loading")
        return None

    with open(container_file_path(cnt), "rb") as pkl_file:
        container = pickle.load(pkl_file)
        assert isinstance(container, dict)

    print(f"traj_pool_file loaded : {container_file_path(cnt)}")

    return container


def save_container(container, cnt, dir=None):
    """
        Try saving container for behavior cloning.
    """
    container_file_name = f"container-{cnt}.pkl"
    container_dir = f"{cfg.logdir}/traj_container/"
    
    if not os.path.exists(container_dir): os.makedirs(container_dir)
    with open(f"{container_dir}/{container_file_name}", "wb") as pkl_file:
        pickle.dump(container, pkl_file)
    print(f"container saved in file: {container_dir}/{container_file_name}")



def get_container_from_traj_pool(traj_pool, req_dict_rename, req_dict=None):
    container = {}
    if req_dict is None: req_dict = ['avail_act', 'obs', 'action', 'actionLogProb', 'return', 'reward', 'value']
    assert len(req_dict_rename) == len(req_dict)

    # replace 'obs' to 'obs > xxxx'
    for key_index, key in enumerate(req_dict):
        key_name =  req_dict[key_index]
        key_rename = req_dict_rename[key_index]
        if not hasattr(traj_pool[0], key_name):
            real_key_list = [real_key for real_key in traj_pool[0].__dict__ if (key_name+'>' in real_key)]
            assert len(real_key_list) > 0, ('check variable provided!', key, key_index)
            for real_key in real_key_list:
                mainkey, subkey = real_key.split('>')
                req_dict.append(real_key)
                req_dict_rename.append(key_rename+'>'+subkey)
    big_batch_size = -1  # vector should have same length, check it!
    
    # load traj into a 'container'
    for key_index, key in enumerate(req_dict):
        key_name =  req_dict[key_index]
        key_rename = req_dict_rename[key_index]
        if not hasattr(traj_pool[0], key_name): continue
        set_item = np.concatenate([getattr(traj, key_name) for traj in traj_pool], axis=0)
        if not (big_batch_size==set_item.shape[0] or (big_batch_size<0)):
            print('error')
        assert big_batch_size==set_item.shape[0] or (big_batch_size<0), (key,key_index)
        big_batch_size = set_item.shape[0]
        container[key_rename] = set_item    # 指针赋值

    return container