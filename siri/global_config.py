import os, threading


def get_root_dir():
    # root_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.getcwd() + '/'
    return root_dir


class GlobalConfig:
    debug = False

    root_dir = get_root_dir()

    device = 'cuda:0'
    conf_threshold = 0.33
    half = True
    tick = 0.04
    # tick = 0.05
    # tick = 1
    # sz_wh = (640, 360,)
    sz_wh = (1280, 578)

    manual_preprocess=False

    body_y_offset = 0.1

    plt = 'qt'

    yolo_plt = False

class GloablStatus:
    monitor = None
    stop_event = threading.Event()

    @classmethod
    def in_window_center_xy(cls):
        assert cls.monitor is not None
        return (cls.monitor['width']/2, cls.monitor['height']/2)