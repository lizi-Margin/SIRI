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

