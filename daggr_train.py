import time, numpy as np, cv2, gymnasium.spaces as spaces, copy, torch, random
from imitation_daggr.foundation import ReinforceAlgorithmFoundation
from imitation.inputs import DAggrGrabber
    

def main():
    rl_alg = ReinforceAlgorithmFoundation()
    rl_alg.load_model()

    grabber = DAggrGrabber()
    grabber.start_rl_session(rl_alg)

if __name__ == "__main__":
    main()
