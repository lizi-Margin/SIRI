def get_a_logger():
    from VISUALIZE.mcom import mcom, logdir
    mcv = mcom( path='%s/logger/'%logdir,
                    digit=16,
                    rapid_flush=True,
                    draw_mode='Img',
                    tag='[task_runner.py]',
                    resume_mod=False)
    mcv.rec_init(color='k')
    return mcv




class AlgorithmConfig:
    logdir = './imitation_TRAIN/DAggr/'
    device = 'cuda:0'
    # device = 'cpu'

    train_traj_needed = 10
    prevent_batchsize_oom = False
    gamma_in_reward_forwarding = False
    gamma_in_reward_forwarding_value = 0.99
    gamma = 0.99
    tau = 0.95
    train_traj_needed = 2
    TakeRewardAsUnity = False
    use_normalization = True
    add_prob_loss = False
    n_entity_placeholder = 10
    load_checkpoint = False
    load_specific_checkpoint = ''
    clip_param = 0.2
    ppo_epoch = 16
    n_pieces_batch_division = 1
    value_loss_coef = 0.1
    entropy_coef = 0.05
    max_grad_norm = 0.5
    clip_param = 0.2


    show_preprocessed_preview = False

    num_epoch_per_update = 4
    sample_size_min = 45
    sample_size_max = 45

    # num_epoch_per_update = 10
    # sample_size_min = 20
    # sample_size_max = 80

    # behavior cloning part
    lr = 0.005
    lr_sheduler_min_lr = 0.0005
    # lr = 0.005 
    # lr_sheduler_min_lr = 0.0008
    lr_sheduler = True  # whether to use lr_sheduler
    dist_entropy_loss_coef = 1e-4

    binary_coef = 0.

    mcom = get_a_logger()
