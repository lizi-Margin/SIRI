def get_a_logger():
    from VISUALIZE.mcom import mcom, logdir
    mcv = mcom( path='%s/logger/'%logdir,
                    digit=16,
                    rapid_flush=True,
                    draw_mode='Img',
                    tag='[task_runner.py]',
                    resume_mod=False)
    mcv.rec_init(color='b')
    return mcv



class AlgorithmConfig:
    logdir = './imitation_TRAIN/BC/'
    device = 'cuda:0'

    show_preprocessed_preview = True

    num_epoch_per_update = 4
    sample_size_min = 45
    sample_size_max = 45

    # num_epoch_per_update = 10
    # sample_size_min = 20
    # sample_size_max = 80

    # behavior cloning part
    lr = 0.01
    lr_sheduler_min_lr = 0.005
    # lr = 0.005 
    # lr_sheduler_min_lr = 0.0008
    lr_sheduler = True  # whether to use lr_sheduler
    dist_entropy_loss_coef = 1e-4

    mcom = get_a_logger()
