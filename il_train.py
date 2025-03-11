import time, numpy as np, cv2, gymnasium.spaces as spaces, copy, torch
from typing import Union, List
from siri.utils.logger import lprint
from imitation.utils import safe_load_traj_pool, safe_dump_traj_pool, print_dict, get_container_from_traj_pool, print_nan, check_nan
from imitation.discretizer import SimpleDiscretizer, wasd_Discretizer
from imitation.bc import wasd_xy_Trainer
from imitation.net import LSTMNet as Net, NetActor
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

def load_and_train():
    req_dict_name = ['key', 'mouse', 'FRAME_raw']


    load = safe_load_traj_pool()
    traj_pool = load(n_samples=1)
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

    data = {
        'frame_center': frame_center, 

        'wasd': index_wasd,
        'x': index_mouse_x,
        'y': index_mouse_y
    }

    trainer.train_on_data_(data)

if __name__ == '__main__':
    N_LOAD = 2000
    for i in range(N_LOAD):
        decoration = "_" * 20
        print(decoration + f" load{i} starts " + decoration)
        load_and_train()
        time.sleep(30)