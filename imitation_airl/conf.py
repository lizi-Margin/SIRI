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
    logdir = './imitation_TRAIN/AIRL/'
    device = 'cuda:0'
    mcom = get_a_logger()
    show_preprocessed_preview = True
    datasets = ['traj-Grabber-tick=0.1-limit=200-pure']

    # configuration, open to jsonc modification
    gamma = 0.99
    tau = 0.95
    train_traj_needed = 2
    TakeRewardAsUnity = False
    use_normalization = True
    add_prob_loss = False
    n_entity_placeholder = 10
    load_checkpoint = False
    load_specific_checkpoint = ''

    # PPO part
    clip_param = 0.2
    ppo_epoch = 16
    n_pieces_batch_division = 1
    value_loss_coef = 0.1
    entropy_coef = 0.05
    max_grad_norm = 0.5
    clip_param = 0.2
    lr = 1e-4
    ppo_lr = lr

    # adversarial part
    num_epoch_per_update = 20
    disc_lr = 5e-4
    test_reward_net = True

    # sometimes the episode length gets longer,
    # resulting in more samples and causing GPU OOM,
    # prevent this by fixing the number of samples to initial
    # by randomly sampling and droping
    prevent_batchsize_oom = False
    gamma_in_reward_forwarding = False
    gamma_in_reward_forwarding_value = 0.99

    net_hdim = 24
    
    dual_conc = True

    n_entity_placeholder = 'auto load, do not change'
    n_agent = 'auto load, do not change'
    entity_distinct = 'auto load, do not change'

    ConfigOnTheFly = True


    

    policy_resonance = False

    use_avail_act = True
    
    debug = False


class AlgorithmConfigBC:
    logdir = AlgorithmConfig.logdir
    device = AlgorithmConfig.device

    show_preprocessed_preview = AlgorithmConfig.show_preprocessed_preview

    num_epoch_per_update = 4
    sample_size_min = 45
    sample_size_max = 45

    # num_epoch_per_update = 10
    # sample_size_min = 20
    # sample_size_max = 80

    # behavior cloning part
    lr = 0.01
    lr_sheduler_min_lr = 0.001
    # lr = 0.005 
    # lr_sheduler_min_lr = 0.0008
    lr_sheduler = True  # whether to use lr_sheduler
    dist_entropy_loss_coef = 1e-4

    binary_coef = 0.

    mcom = AlgorithmConfig.mcom

