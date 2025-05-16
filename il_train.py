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

# from imitation_bc.bc import FullTrainer as Trainer
# from imitation_bc.net import LSTMB5 as Net, NetActor as NetActor
# from imitation_bc.net import DVNet_SAF as Net, NetActor as NetActor
# from imitation_bc.net import DVNetDual_CA as Net, NetActor as NetActor
# from imitation_bc.map_net import DoubleBranchMapNet as Net;NetActor = Net

# from imitation_airl.bc import Trainer
# from imitation_airl.AC import DoubleBranchMapAC as Net;NetActor = Net

from imitation_daggr.bc import Trainer
# from imitation_daggr.AC import DoubleBranchMapAC as Net;NetActor = Net
from imitation_daggr.AC import TransformerMapAC as Net;NetActor = Net

# CENTER_SZ_WH = NetActor.CENTER_SZ_WH
# try:
#     policy = Net(CENTER_SZ_WH, wasd_discretizer.n_actions, x_discretizer.n_actions, y_discretizer.n_actions)
# except:
policy = Net()

trainer = Trainer(policy)
trainer.load_model()

from data_loader import data_loader_process

def train_on(traj_dir, N_LOAD=2000):
    n_traj = 35
    traj_reuse = 1
    # torch.rand(1)
    queue = mp.Queue(maxsize=2)
    loader_process = mp.Process(target=data_loader_process, args=(traj_dir, n_traj, queue, NetActor), daemon=True)
    loader_process.start()

    for i in range(N_LOAD):
        decoration = "_" * 20
        datas, dir_name = queue.get()  # wait
        print(decoration + f" train N_LOAD={i} starts, traj_dir={dir_name} " + decoration)

        for j in range(n_traj * traj_reuse):
            data = copy.copy(datas[j % n_traj])
            data['obs'] = NetActor.preprocess(data['obs'])
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

    # train_on([
    #     'traj-Grabber-tick=0.1-limit=200-pp19',
    #     'traj-Grabber-tick=0.1-limit=200-nav',
    #     'traj-Grabber-tick=0.1-limit=200-pure',
    #     'traj-Grabber-tick=0.1-limit=200-old',
    #     'traj-Grabber-tick=0.1-limit=200-classic-pp19',
    # ], N_LOAD=500)

    train_on([
        'traj-Grabber-tick=0.1-limit=200-pure',
    ])