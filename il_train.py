import time, numpy as np, cv2, gymnasium.spaces as spaces, copy, torch
from typing import Union, List
from siri.utils.logger import lprint
from imitation.utils import safe_load_traj_pool, safe_dump_traj_pool, print_dict, get_container_from_traj_pool, print_nan, check_nan
from imitation.discretizer import SimpleDiscretizer, wasd_Discretizer
from imitation.bc import wasd_xy_Trainer
# from imitation.net import DVNet as Net, DVNet as NetActor
# from imitation.net import DVNet2 as Net, DVNet2 as NetActor
# from imitation.net import DVNet3 as Net, DVNet3 as NetActor
# from imitation.net import DVNet4 as Net, DVNet4 as NetActor
from imitation.net import LSTMNet as Net, NetActor as NetActor
from siri.vision.preprocess import crop_wh

x_discretizer = NetActor.x_discretizer
y_discretizer = NetActor.y_discretizer
wasd_discretizer = NetActor.wasd_discretizer
preprocess = NetActor.preprocess
get_center = NetActor.get_center

CENTER_SZ_WH = NetActor.CENTER_SZ_WH
policy = Net(CENTER_SZ_WH, wasd_discretizer.n_actions, x_discretizer.n_actions, y_discretizer.n_actions)
trainer = wasd_xy_Trainer(policy)
trainer.load_model()

def get_data(traj_pool):
    req_dict_name = ['key', 'mouse', 'FRAME_raw']
    for traj in traj_pool:
        for name in req_dict_name:
            check_nan(getattr(traj, name))
            # print(len(getattr(traj, name)))

    container = get_container_from_traj_pool(traj_pool, req_dict_name, req_dict_name)
    print_dict(container)


    container['FRAME_center'] = np.array([get_center(frame.copy()) for frame in container['FRAME_raw']])
    # frame = container['FRAME_center'][0]
    # print(frame.shape)
    # cv2.imshow('x',frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



    frame_center = preprocess(container['FRAME_center'])
    act_wasd = container['key'][:, :4]
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

    if isinstance(frame_center, (tuple, list,)):
        for i in range(len(frame_center)):
            frame_center[i] = frame_center[i].to('cpu')
    else:
        frame_center = frame_center.to('cpu')

    data = {
        'obs': frame_center, 

        'wasd': index_wasd,
        'x': index_mouse_x,
        'y': index_mouse_y,
    }

    return data


def train_on(traj_dir, N_LOAD=2000):
    n_traj = 40
    traj_reuse = 2
    for i in range(N_LOAD):
        decoration = "_" * 20
        print(decoration + f" load{i} starts, traj_dir={traj_dir} " + decoration)
        load = safe_load_traj_pool(traj_dir=traj_dir)
        # load = safe_load_traj_pool(traj_dir='traj-Grabber-tick=0.1-limit=200-pure')
        pool = load(n_samples=n_traj)
        datas = [get_data([traj]) for traj in pool]
        for j in range(n_traj * traj_reuse):
            data = copy.copy(datas[j%n_traj])
            print_dict(data)
            trainer.train_on_data_(data)
        del datas
        del pool


if __name__ == '__main__':
    # train_on('traj-Grabber-tick=0.1-limit=200-nav', N_LOAD=2)
    # train_on('traj-Grabber-tick=0.1-limit=200-old', N_LOAD=12)
    # train_on('traj-Grabber-tick=0.1-limit=200-pure')
    train_on('traj-Grabber-tick=0.1-limit=200-nav')