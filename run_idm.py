import time, numpy as np, cv2, gymnasium.spaces as spaces, copy, torch, random
from imitation_daggr.foundation import ReinforceAlgorithmFoundation
from siri.utils.logger import lprint, lprint_
from UTIL.colorful import *
from imitation.utils import safe_dump_traj_pool, iterable_eq
from imitation.traj import trajectory

def norm(x: float, lower_side: float=-1.0, upper_side: float=1.0):
    if (x > upper_side): x = upper_side
    if (x < lower_side): x = lower_side
    return x
class IDM_Labler():
    def __init__(self):
        super().__init__()
        self.traj_pool: list[trajectory] = []
        self.traj_limit = 200
        self.tick = 0.1

    def new_traj(self):
        # Create trajectory storage
        traj = trajectory(traj_limit=self.traj_limit, env_id=0)
        return traj
    @staticmethod
    def get_real_act(wasd, xy):
        limit = 500
        mv_x, mv_y = norm(xy[0], lower_side=-limit, upper_side=limit), norm(xy[1], lower_side=-limit, upper_side=limit)
        act_dict = {
            'w': 0,
            'a': 0,
            's': 0,
            'd': 0,
        }
        if wasd[0] > 0:
            act_dict['w'] = 1
        elif wasd[1] > 0:
            act_dict['a'] = 1
        elif wasd[2] > 0:
            act_dict['s'] = 1
        elif wasd[3] > 0:
            act_dict['d'] = 1
        return act_dict, mv_x, mv_y
    
    def start_idm_session(self, rl_alg, video_dir_path):
        from data_loader import VideoDatasetLoader
        video_loader = VideoDatasetLoader(
            root_dir=video_dir_path,
            interval_sec=self.tick,
            shuffle=False
        )

        try:
            traj = self.new_traj()

            for data in video_loader:
                frame = data['frame']
                frame_count = data['frame_count']
                video_path = ['video_path']
                current_time = data['timestamp']
                video_idx = data['video_index']
                done = data['done']
                print绿('\r'+lprint_(self, f"started {video_path}, traj collected: {len(self.traj_pool)}"), end='')
                m_wasd, m_xy = rl_alg.interact_with_env({
                    'obs': frame,
                    'done': done,
                    'rec': {},
                    'human_active': False
                })
                
                act_dict, mv_x, mv_y = self.get_real_act(m_wasd, m_xy)
                key = np.concatenate([m_wasd, np.zeros(14,)])
                xy = np.array([mv_x, mv_y])
                assert len(key.shape) == 1, len(key.shape)
                video_path, frame_count, current_time, video_idx, done_np = map(
                    np.array, 
                    [data['video_path'], data['frame_count'], data['timestamp'], data['video_index'], data['done']]
                )
                assert key.shape[0] == 18, key.shape[0]
                traj.remember('FRAME_raw', frame.copy())
                traj.remember('key', key.copy())
                traj.remember('mouse', xy.copy())
                traj.remember('done', done_np)
                traj.remember('frame_count', frame_count)
                traj.remember('video_path', video_path)
                traj.remember('timestamp', current_time)
                traj.remember('video_index', video_idx)
                traj.time_shift()
                if traj.time_pointer == self.traj_limit or done:
                    self.traj_pool.append(traj)
                    traj = self.new_traj()
                    if len(self.traj_pool) > 10:
                        self.save_traj()
        except KeyboardInterrupt:
            lprint(self, "Sig INT catched, stopping session.")
        finally:
            if traj.time_pointer > 0:
                self.traj_pool.append(traj)

            # cv2.destroyAllWindows()
            self.save_traj()
            print亮黄(lprint_(self, "terminated"))
    
    def save_traj(self):
        if len(self.traj_pool) > 0:
            for i in range(len(self.traj_pool)): self.traj_pool[i].cut_tail()
            pool_name = f"{self.__class__.__name__}-tick={self.tick}-limit={self.traj_limit}-{time.strftime("%Y%m%d-%H:%M:%S")}"
            safe_dump_traj_pool(self.traj_pool, pool_name, traj_dir=f"IDM/AUTOSAVED/{time.strftime("%Y%m%d-%H:%M:%S")}/")
            self.traj_pool = []

def main():
    alg = ReinforceAlgorithmFoundation()
    alg.trainer.load_model(pt_path='imitation_TRAIN/DAggr/model-DoubleBranchMapNet(aug)-nop-p19-cp19-40000-p-25000.pt')

    grabber = IDM_Labler()
    grabber.start_idm_session(alg, "./web_video/")

if __name__ == "__main__":
    main()
