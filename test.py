from net_utils.utils import load_device, load_model, load_tester, load_dataloader
from net_utils.utils import CheckpointIO
from configs.config_utils import mount_external_config

def run(cfg, mode):
    eval_map = True
    cfg.config['mode'] = 'test'

    if mode == "prepare_data":
        from prepare_data import prepare_data_for_recon as test

    else:
        from test_epoch import test


    '''Begin to run network.'''
    checkpoint = CheckpointIO(cfg)

    '''Mount external config data'''
    cfg = mount_external_config(cfg, eval_map = eval_map)

    '''Load save path'''
    cfg.log_string('Data save path: %s' % (cfg.save_path))

    '''Load device'''
    cfg.log_string('Loading device settings.')
    device = load_device(cfg)


    '''Load net'''
    cfg.log_string('Loading model.')
    net = load_model(cfg, device=device)
    checkpoint.register_modules(net=net)
    cfg.log_string(net)

    '''Load existing checkpoint'''
    if 'bsp_weight' in cfg.config:
        checkpoint.load_bsp_decoder()
    checkpoint.parse_checkpoint()

    '''Load tester'''
    cfg.log_string('Loading tester.')
    tester = load_tester(cfg=cfg, net=net, device=device)

    '''Start to test'''
    cfg.log_string('Start to test.')
    cfg.log_string('Total number of parameters in {0:s}: {1:d}.'.format(cfg.config['method'], sum(p.numel() for p in net.parameters())))

    #prepare_data_for_refine,
    '''Load data'''
    cfg.log_string('Loading dataset.')
    _,test_loader,_ = load_dataloader(cfg, mode='test')
    test(cfg=cfg, tester=tester, test_loader=test_loader)

    cfg.write_config()
    cfg.log_string('Testing finished.')