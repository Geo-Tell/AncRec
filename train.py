# Training script
# author: ynie
# date: Feb, 2020
from models.optimizers import load_optimizer, load_scheduler, load_bnm_scheduler
from net_utils.utils import load_device, load_model, load_trainer, load_dataloader
from net_utils.utils import CheckpointIO
from configs.config_utils import mount_external_config

def run(cfg, mode):
    cfg.config['mode'] = 'train'
    '''Begin to run network.'''
    checkpoint = CheckpointIO(cfg)

    '''Mount external config data'''
    eval_map = True if mode =="train" else False
    cfg = mount_external_config(cfg, eval_map= eval_map)

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

    '''Load optimizer'''
    cfg.log_string('Loading optimizer.')
    optimizer = load_optimizer(config=cfg.config, net=net)
    checkpoint.register_modules(optimizer=optimizer)

    '''Load scheduler'''
    cfg.log_string('Loading optimizer scheduler.')
    # scheduler, warmup_scheduler = load_scheduler(config=cfg.config, optimizer=optimizer)
    scheduler = load_scheduler(config=cfg.config, optimizer=optimizer) #
    checkpoint.register_modules(scheduler=scheduler)

    '''Check existing checkpoint (resume or finetune)'''
    if 'bsp_weight' in cfg.config:
        checkpoint.load_bsp_decoder(load_module =['decoder','generator'])#, load_encoder=True )

    checkpoint.parse_checkpoint()

    '''Load trainer'''
    cfg.log_string('Loading trainer.')
    trainer = load_trainer(cfg=cfg, net=net, optimizer=optimizer, device=device)

    '''Start to train'''
    cfg.log_string('Start to train.')
    cfg.log_string('Total number of parameters in {0:s}: {1:d}.'.format(cfg.config['method'], sum(p.numel() for p in net.parameters())))

    if mode == "train_recon":
        from train_epoch_reconstruction import train
        train(cfg=cfg, trainer=trainer, scheduler=scheduler, checkpoint=checkpoint, validate=True) #,warmup_scheduler=warmup_scheduler

    else: #train detection
        '''BN momentum scheduler'''
        cfg.log_string('Loading BN momentum scheduler.')
        bnm_scheduler = load_bnm_scheduler(cfg=cfg, net=net, start_epoch=scheduler.last_epoch)

        cfg.log_string('Loading dataset.')
        train_dataset,  train_loader, train_sampler = load_dataloader(cfg, mode='train')
        val_dataset,  val_loader,  val_sampler = load_dataloader(cfg, mode='val')

        from train_epoch_detection import train
        train(cfg=cfg, trainer=trainer, scheduler=scheduler, bnm_scheduler=bnm_scheduler, checkpoint=checkpoint,
              train_loader=train_loader, val_loader=val_loader,train_sampler = train_sampler,validate=True)

    cfg.log_string('Training finished.')