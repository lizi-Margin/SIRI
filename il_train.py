import time, numpy as np, cv2, gymnasium.spaces as spaces, copy, torch, random
import multiprocessing as mp
# from typing import Union, List
# from siri.utils.logger import lprint
from imitation.utils import safe_load_traj_pool, safe_dump_traj_pool, print_dict, get_container_from_traj_pool, print_nan, check_nan
# from imitation.discretizer import SimpleDiscretizer, wasd_Discretizer
from UTIL.colorful import *

# from imitation.bc import wasd_xy_Trainer as Trainer
# from imitation.net import DVNet as Net, DVNet as NetActor
# from imitation.net import DVNet2 as Net, DVNet2 as NetActor
# from imitation.net import DVNet3 as Net, DVNet3 as NetActor
# from imitation.net import DVNet4 as Net, DVNet4 as NetActor
# from imitation.net import LSTMNet as Net, NetActor as NetActor

from imitation_bc.bc import FullTrainer as Trainer
# from imitation_bc.net import LSTMB5 as Net, NetActor as NetActor
# from imitation_bc.net import DVNet_SAF as Net, NetActor as NetActor
# from imitation_bc.net import DVNetDual_CA as Net, NetActor as NetActor
from imitation_bc.map_net import DoubleBranchMapNet as Net;NetActor = Net


x_discretizer = NetActor.x_discretizer
y_discretizer = NetActor.y_discretizer
wasd_discretizer = NetActor.wasd_discretizer


CENTER_SZ_WH = NetActor.CENTER_SZ_WH
try:
    policy = Net(CENTER_SZ_WH, wasd_discretizer.n_actions, x_discretizer.n_actions, y_discretizer.n_actions)
except:
    policy = Net()

trainer = Trainer(policy)
# trainer.load_model()
preprocess = NetActor.preprocess
get_center = NetActor.get_center


def get_data(traj_pool):
    req_dict_name = ['key', 'mouse', 'FRAME_raw']
    for traj in traj_pool:
        for name in req_dict_name:
            check_nan(getattr(traj, name))
            # print(len(getattr(traj, name)))

    container = get_container_from_traj_pool(traj_pool, req_dict_name, req_dict_name)
    print_dict(container)

    container['FRAME_center'] = [get_center(frame.copy()) for frame in container['FRAME_raw']]
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

    # if isinstance(frame_center, (tuple, list,)):
    #     for i in range(len(frame_center)):
    #         frame_center[i] = frame_center[i].to('cpu')
    # else:
    #     frame_center = frame_center.to('cpu')

    data = {
        # 'obs': preprocess(frame_center), 
        'obs': frame_center, 

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

def data_loader_process(traj_dir, n_traj, queue):
    print蓝("[data_loader_process] started")
    load = safe_load_traj_pool(traj_dir=traj_dir)
    while True:
        qsz = queue.qsize()
        while qsz >= 1:
            print蓝(f"[data_loader_process] waiting, queue.qsize()={qsz}")
            time.sleep(1)
            qsz = queue.qsize()
        print蓝(f"[data_loader_process] start loading: {traj_dir}")
        pool = load(n_samples=n_traj)
        datas = []
        for traj in pool:
            datas.append(get_data([traj]))
        print蓝(f"[data_loader_process] load completed")
        queue.put_nowait((datas, traj_dir,)) 
        del pool


def train_on(traj_dir, N_LOAD=2000):
    n_traj = 10
    traj_reuse = 1
    # torch.rand(1)
    queue = mp.Queue(maxsize=2)
    loader_process = mp.Process(target=data_loader_process, args=(traj_dir, n_traj, queue), daemon=True)
    loader_process.start()

    for i in range(N_LOAD):
        decoration = "_" * 20
        datas, dir_name = queue.get()  # wait
        print(decoration + f" train N_LOAD={i} starts, traj_dir={dir_name} " + decoration)

        for j in range(n_traj * traj_reuse):
            data = copy.copy(datas[j % n_traj])
            data['obs'] = preprocess(data['obs'])
            print_dict(data)
            try:
                trainer.train_on_data_(data)
            except torch.OutOfMemoryError:
                continue
        
        del datas
        trainer.save_model()
    loader_process.terminate()


if __name__ == '__main__':
    # train_on('traj-Grabber-tick=0.1-limit=200-nav', N_LOAD=2)
    # train_on('traj-Grabber-tick=0.1-limit=200-old', N_LOAD=12)
    # train_on('traj-Grabber-tick=0.1-limit=200-pure')
    # train_on('traj-Grabber-tick=0.1-limit=200-nav')
    # train_on([
    #     'traj-Grabber-tick=0.1-limit=200-nav', 
    #     'traj-Grabber-tick=0.1-limit=200-pp19'
    # ])
    
    
    # train_on([
    #     'traj-Grabber-tick=0.1-limit=200-pp19',
    #     'traj-Grabber-tick=0.1-limit=200-nav',
    #     'traj-Grabber-tick=0.1-limit=200-pure',
    #     'traj-Grabber-tick=0.1-limit=200-old',
    # ], N_LOAD=15)

    # train_on('traj-Grabber-tick=0.1-limit=200-fight-pp19')

    # train_on('traj-Grabber-tick=0.1-limit=200-classic-pp19')

    train_on([
        'traj-Grabber-tick=0.1-limit=200-pp19',
        'traj-Grabber-tick=0.1-limit=200-nav',
        'traj-Grabber-tick=0.1-limit=200-pure',
        'traj-Grabber-tick=0.1-limit=200-old',
        'traj-Grabber-tick=0.1-limit=200-classic-pp19',
    ])

    train_on([
        'traj-Grabber-tick=0.1-limit=200-pure',
    ], N_LOAD=16)