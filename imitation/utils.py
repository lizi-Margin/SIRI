import os,time,cv2,copy
import sys
import json
import pprint
import numpy as np
from random import sample
from typing import Union, List, Tuple, Dict
from UTIL.colorful import *
from siri.utils.logger import lprint, lprint_
class cfg:
    logdir = './HMP_IL/'

def print_dict(data):
    summary = {
        key: f" {type(value)}, shape={value.shape}, dtype={value.dtype}" if isinstance(value, np.ndarray) 
                                                                         else type(value) 
                                                                         for key, value in data.items()
    }
    pprint.pp(summary)

def iterable_eq(a,b):
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True

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


def check_nan(traj_np):
    is_not_nan = np.zeros((len(traj_np),), dtype=int)
    for i in range(len(traj_np)):
        is_not_nan[i] = -np.any(np.isnan(traj_np[i]), axis=None).astype(int) + 1
    
    not_nan_ratio = (np.sum(is_not_nan, axis=0)/len(is_not_nan))
    assert not_nan_ratio == 1., f"not_nan_ratio={not_nan_ratio}"


def print_nan(traj_np):
    is_not_nan = np.zeros((len(traj_np),), dtype=int)
    for i in range(len(traj_np)):
        is_not_nan[i] = -np.any(np.isnan(traj_np[i]), axis=None).astype(int) + 1

    print(f"not NaN percent: {round(np.sum(is_not_nan, axis=0)/len(is_not_nan) * 100, 2)}%")


def is_basic_type(obj):
    if (
        isinstance(obj, int) or isinstance(obj, bool) or isinstance(obj, str) 
        or isinstance(obj, list) or isinstance(obj, tuple) or isinstance(obj, dict)
        or (obj is None)
    ):
        return True
    else:
        return False


def save_and_compress_FRAMEs(FRAMEs: np.ndarray, path, FRAMEs_name):
    FRAMES_dir_name = f"{FRAMEs_name}.d"
    FRAMEs_dir = os.path.join(path, FRAMES_dir_name)
    if not os.path.exists(FRAMEs_dir):
        os.makedirs(FRAMEs_dir)

    image_data = []
    for i, img_np in enumerate(FRAMEs):
        file_name = os.path.join(FRAMES_dir_name, f"{FRAMEs_name}_{i}.png")
        if np.any(np.isnan(img_np), axis=None) or np.all(img_np == 0):
            is_nan = True
        else:
            is_nan = False
            img_path = os.path.join(path, file_name)
            cv2.imwrite(img_path, img_np)
        print(f"\r[save_and_compress_FRAMEs] saving {file_name}, is_nan={is_nan}", end='')
        image_data.append({
            'file_name': file_name,
            'shape': img_np.shape,
            'is_nan': is_nan
        })
    print(end='\n')

    json_path = os.path.join(path, f"{FRAMEs_name}.json")
    with open(json_path, 'w') as f:
        json.dump(image_data, f)


def load_compressed_FRAMEs(path, FRAMEs_name):
    with open(os.path.join(path, f"{FRAMEs_name}.json"), 'r') as f:
        meta_data = json.load(f)

    images = []
    for img_info in meta_data:
        file_name = img_info['file_name']
        expected_shape = img_info['shape']
        if ('is_nan' in img_info) and img_info['is_nan'] == True:
            is_nan = True
            img_np = np.zeros(expected_shape, dtype=np.uint8)
            # img_np[...] = np.nan
        else:
            is_nan = False
            img_path = os.path.join(path, file_name)
            img_np = cv2.imread(img_path).astype(np.uint8)
            loaded_shape = img_np.shape

            if not iterable_eq(expected_shape, loaded_shape):
                raise ValueError(f"Image {file_name} has inconsistent shape. "
                                f"Expected: {expected_shape}, Got: {loaded_shape}")
            
        print(f"\r[load_compressed_FRAMEs] loading {file_name}, is_nan={is_nan}", end='')

        # if len(expected_shape) == 3 and expected_shape[2] == 3:
        #     img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        images.append(img_np)
    print(end='\n')
    
    return np.array(images)


def safe_dump(obj, path):
    if not os.path.exists(path): os.makedirs(path)
    cls_name = obj.__class__.__name__
    serializable_data = {}
    numpy_arrays = {}
    FRAMEs = {}
    for attr, value in obj.__dict__.items():
        if str(attr).startswith("FRAME"): 
            assert isinstance(value, np.ndarray)
            print(f"[safe_dump] {obj.__class__.__name__}.{attr} is FRAME type")
            FRAMEs[attr] = value
        elif isinstance(value, np.ndarray):
            numpy_arrays[attr] = value
        elif is_basic_type(value):
            serializable_data[attr] = value
        else:
            assert False, f'not implemented yet, key={attr} type={type(value)}'

    npy_filenames = {}
    for key, array in numpy_arrays.items():
        npy_filename = f"{cls_name}_{key}.npy"
        np.save(f"{path}/{npy_filename}", array, allow_pickle=True)
        npy_filenames[key] = npy_filename
    serializable_data['npy_filenames'] = npy_filenames

    FRAMEs_filenames = {}
    for key, frame in FRAMEs.items():
        FRAMEs_name = f"{cls_name}_{key}"
        save_and_compress_FRAMEs(frame, path, FRAMEs_name)
        FRAMEs_filenames[key] = FRAMEs_name
    serializable_data['FRAMEs_filenames'] = FRAMEs_filenames

    with open(f"{path}/{cls_name}.json", 'w') as f:
        json.dump(serializable_data, f)

old_dataset_names = [
    'traj-Grabber-tick=0.1-limit=200-pp19',
    'traj-Grabber-tick=0.1-limit=200-nav',
    'traj-Grabber-tick=0.1-limit=200-pure',
    'traj-Grabber-tick=0.1-limit=200-old',
]
def has_old_dataset_name(path :str):
    for name in old_dataset_names:
        if name in path:
            return True
    return False

def safe_load(obj, path):
    if not os.path.exists(path):
        print亮黄(f"warning: {path} not found, skip loading")
        return obj

    cls_name = obj.__class__.__name__
    serializable_data = {}
    numpy_arrays = {}
    FRAMEs = {}
    with open(f"{path}/{cls_name}.json", 'r') as f:
        serializable_data = json.load(f)
    assert isinstance(serializable_data, dict)

    npy_filenames = serializable_data.pop('npy_filenames')
    for key, npy_filename in npy_filenames.items():
        numpy_arrays[key] = np.load(f"{path}/{npy_filename}", allow_pickle=True)

        if has_old_dataset_name(path):
            if key == "mouse":
                print亮黄(f"[safe_load] warning: {path} has old dataset signiture, load key={key} in legacy mode")
                print亮黄(f"[safe_load] key={key}, shape={numpy_arrays[key].shape}, max={np.max(numpy_arrays[key])}, min={np.min(numpy_arrays[key])}, processing")
                numpy_arrays[key] = numpy_arrays[key] / 2
                print亮黄(f"[safe_load] key={key}, shape={numpy_arrays[key].shape}, max={np.max(numpy_arrays[key])}, min={np.min(numpy_arrays[key])}, processed")


    if 'FRAMEs_filenames' in serializable_data:
        FRAMEs_filenames = serializable_data.pop('FRAMEs_filenames')
        for key, FRAMEs_name in FRAMEs_filenames.items():
            FRAMEs[key] = load_compressed_FRAMEs(path, FRAMEs_name)

    for attr, value in {**serializable_data, **numpy_arrays, **FRAMEs}.items():
        setattr(obj, attr, value)
    
    return obj


def safe_dump_traj_pool(traj_pool, pool_name, traj_dir=None):
    default_traj_dir = f"{cfg.logdir}/traj_pool_safe/"
    if traj_dir is None: traj_dir = f"{cfg.logdir}/{time.strftime("%Y%m%d-%H:%M:%S")}/"
    
    for index, traj in enumerate(traj_pool):
        traj_name = f"traj-{pool_name}-{index}.d"
        safe_dump(obj=traj, path=f"{traj_dir}/{traj_name}")
    
        print亮黄(f"traj saved in file: {traj_dir}/{traj_name}")

    # if os.path.islink(default_traj_dir[:-1]):
    #     os.unlink(default_traj_dir[:-1])
    # os.symlink(os.path.abspath(traj_dir), os.path.abspath(default_traj_dir))


class safe_load_traj_pool:
    def __init__(self, max_len=None, traj_dir="traj_pool_safe"):
        if isinstance(traj_dir, str): traj_dir = [traj_dir]
        self.traj_names = []
        for i in range(len(traj_dir)):
            traj_dir[i] = f"{cfg.logdir}/{traj_dir[i]}/"
            for traj_name in os.listdir(traj_dir[i]):
                self.traj_names.append(f"{traj_dir[i]}/{traj_name}")
            
        self.used_traj_names = []
        if max_len is not None:
            assert max_len > 0
            if max_len < len(self.traj_names):
                self.traj_names = self.traj_names[:max_len]
    
    def __call__(self, pool_name='', n_samples=200):
        from .traj import trajectory
        traj_pool = []
        if len(self.traj_names) > n_samples:
            n_samples = max(n_samples, 1)

            if len(self.used_traj_names) > (len(self.traj_names) - n_samples):
                self.used_traj_names = []

            traj_names_to_sample = copy.copy(self.traj_names)
            for traj in self.used_traj_names:
                if traj not in traj_names_to_sample: print亮红(lprint_(self, f"ERROR: {traj} not found in self.traj_names !!!"))
                else: traj_names_to_sample.remove(traj)
                    

            traj_names = sample(traj_names_to_sample, n_samples)

            self.used_traj_names.extend(traj_names)
            plt = ["o"] * len(self.traj_names)
            for traj in self.used_traj_names:
                index = self.traj_names.index(traj)
                plt[index] = "x"
            print("".join(plt))
        else:
            self.used_traj_names = []
            traj_names = self.traj_names
            print("x" * len(self.traj_names))

        
        for i, path_to_traj in enumerate(traj_names):
            traj_name = os.path.basename(path_to_traj)
            # traj_dir = os.path.dirname(path_to_traj)
            if traj_name.startswith(f"traj-{pool_name}"):
                print(path_to_traj)

                traj = safe_load(
                    obj=trajectory(traj_limit='auto loaded', env_id='auto loaded'),
                    path=path_to_traj
                )
                traj_pool.append(traj)

                # print亮黄(f"traj loaded from file: {traj_dir}/{traj_name}")
            else:
                print亮红(lprint_(self, f"ERROR: traj_name invalid: {path_to_traj}"))
        
        print(f"safe loaded {len(traj_pool)} trajs")
        return traj_pool



def get_container_from_traj_pool(traj_pool, req_dict_rename, req_dict=None):
    container = {}
    if req_dict is None: req_dict = ['obs', 'action', 'action_index', 'actionLogProb', 'return', 'reward', 'value']
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



def get_seq_container_from_traj_pool(traj_pool, req_dict_rename, req_dict):
    container = {}
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
        set_item = np.array([getattr(traj, key_name) for traj in traj_pool], axis=0)
        if not (big_batch_size==set_item.shape[0] or (big_batch_size<0)):
            print('error')
        assert big_batch_size==set_item.shape[0] or (big_batch_size<0), (key,key_index)
        big_batch_size = set_item.shape[0]
        container[key_rename] = set_item    # 指针赋值

    return container