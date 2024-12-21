from net_utils.utils import LossRecorder, LogBoard
from time import time
import torch
import os
from models.anchorrec.SingleObjectDatasetWithRGB import AnchorRec_ScanNet_For_Refine, AnchorRec_ScanNet_For_Refine_single, \
    collate_fn,my_worker_init_fn, denormalize_image
import torch.utils.data
from torch.utils.data import DataLoader
from net_utils.ap_helper import APCalculator
from tqdm import tqdm


def train_epoch(cfg, epoch, trainer, dataloader, log_board): #, warmup_scheduler
    mode = 'train'
    batch_size = cfg.config[mode]['batch_size']
    loss_recorder = LossRecorder(batch_size)
    trainer.net.train(mode == 'train')
    trainer.net.module.set_mode()

    for iter, data in enumerate(dataloader):
        loss = trainer.train_step(data)
        loss_recorder.update_loss(loss)
        torch.cuda.empty_cache()
        # with warmup_scheduler.dampening():
        #     pass
        if ((iter + 1) % cfg.config['log']['print_step']) == 0:
            trainer.show_lr()
            cfg.log_string('Process: Phase: %s. Epoch %d: %d/%d. Current loss: %s.' % (
            mode, epoch, iter + 1, len(dataloader), str(loss)))
            log_board.update(loss, cfg.config['log']['print_step'], mode)
            lrs = [trainer.optimizer.param_groups[i]['lr'] for i in range(len(trainer.optimizer.param_groups))]
            for lr_i, lr in enumerate(lrs):
                if lr_i < 2:
                    log_board.update({'lr_%s'%lr_i: lr}, cfg.config['log']['print_step'],mode)


    cfg.log_string('=' * 100)
    for loss_name, loss_value in loss_recorder.loss_recorder.items():
        cfg.log_string('Currently the last %s loss (%s) is: %f' % (mode, loss_name, loss_value.avg))
    cfg.log_string('=' * 100)

    return loss_recorder.loss_recorder

def train(cfg, trainer, scheduler,checkpoint, validate = True): #warmup_scheduler,
    start_epoch = scheduler.last_epoch #int(scheduler.last_epoch / 2)
    total_epochs = cfg.config['train']['epochs']
    eval_map_start_epoch = cfg.config['train']['eval_map_start_epoch']
    min_eval_loss = checkpoint.get('min_loss')
    # min_train_loss = checkpoint.get('min_loss')
    log_board = LogBoard(os.path.join(cfg._save_path,'runs'))

    # train_dataset = AnchorRec_ScanNet_For_Refine(cfg, 'train')
    train_dataset = AnchorRec_ScanNet_For_Refine_single(cfg, 'train')
    train_dataloader = DataLoader(dataset=train_dataset,
                                  num_workers=cfg.config['device']['num_workers'],
                                  batch_size=cfg.config['train']['batch_size'],
                                  collate_fn=collate_fn,
                                  shuffle = True,
                                  worker_init_fn=my_worker_init_fn)

    if validate:
        val_dataset = AnchorRec_ScanNet_For_Refine(cfg, 'test') #'test')
        val_dataloader = DataLoader(dataset = val_dataset,
                                      num_workers = cfg.config['device']['num_workers'],
                                      batch_size = cfg.config['val']['batch_size'],
                                      collate_fn = collate_fn,
                                      shuffle = False,
                                      worker_init_fn=my_worker_init_fn)
        max_eval_mAP = 0

    for epoch in range(start_epoch, total_epochs):
        cfg.log_string('Epoch (%d/%s):' % (epoch + 1, total_epochs))
        start = time()
        cfg.log_string('Switch Phase to train')
        cfg.log_string('-' * 100)

        train_loss_recorder = train_epoch(cfg, epoch + 1, trainer, train_dataloader,log_board) #, warmup_scheduler)
        train_loss = trainer.eval_loss_parser(train_loss_recorder)
        checkpoint.register_modules(epoch=epoch, min_loss=train_loss)

        # bnm_scheduler.step()
        cfg.log_string('Epoch (%d/%s) Time elapsed: (%f).' % (epoch + 1, total_epochs, time() - start))
        if validate:
            cfg.log_string('Switch Phase to val')
            if epoch < eval_map_start_epoch:
                eval_loss_recorder = evaluate_epoch(cfg, trainer, val_dataloader,log_board,epoch)
            else:
                mAP_object_sum, eval_loss_recorder = \
                    eval_epoch_map(cfg, trainer, val_dataloader, log_board)
                if epoch ==eval_map_start_epoch or  mAP_object_sum > max_eval_mAP:
                    checkpoint.save('best')
                    max_eval_mAP =  mAP_object_sum
                    cfg.log_string('Saved the best checkpoint according to mAP.')
                    cfg.log_string('Currently the best object 0.5 IOU mAP is: %f'% max_eval_mAP)

            eval_loss = trainer.eval_loss_parser(eval_loss_recorder)  # 返回平均值
            checkpoint.register_modules(epoch=epoch, min_loss=eval_loss)
            # with warmup_scheduler.dampening():
            # scheduler.step(eval_loss)
            scheduler.step()
            if epoch == 0 or eval_loss < min_eval_loss:
                # checkpoint.save('min_loss')
                min_eval_loss = eval_loss
                cfg.log_string('Saved the best checkpoint according to eval loss.')
                cfg.log_string('=' * 100)
                for loss_name, loss_value in eval_loss_recorder.items():
                    cfg.log_string('Currently the best val loss (%s) is: %f' % (loss_name, loss_value.avg))
                cfg.log_string('=' * 100)
            # bnm_scheduler.step()
            cfg.log_string('Epoch (%d/%s) Time elapsed: (%f).for validation' % (epoch + 1, total_epochs, time() - start))

        checkpoint.save(str(epoch+1))
        cfg.log_string('Saved the last checkpoint.')

def evaluate_epoch(cfg,trainer,dataloader,log_board, epoch):
    mode = 'val'
    batch_size = cfg.config[mode]['batch_size']
    loss_recorder = LossRecorder(batch_size)
    trainer.net.train(False)
    for iter, data in enumerate(dataloader):
        # loss,_ = trainer.eval_step_for_mAP(data)
        loss, _ = trainer.eval_step(data)
        torch.cuda.empty_cache()
        loss_recorder.update_loss(loss)

        if ((iter + 1) % cfg.config['log']['print_step']) == 0:
            cfg.log_string('Process: Phase: %s.  %d/%d. Current loss: %s.' % (mode, iter + 1, len(dataloader), str(loss)))
            log_board.update(loss, cfg.config['log']['print_step'], mode)


    return loss_recorder.loss_recorder


def eval_epoch_map(cfg, trainer, dataloader, log_board):
    cfg.log_string('-' * 100)
    # set mode
    mode = 'val'
    trainer.net.train(mode == 'train')
    batch_size = cfg.config[mode]['batch_size']
    loss_recorder = LossRecorder(batch_size)

    ap_calculator = APCalculator(0.5, cfg.dataset_config.class2type, False)
    cfg.log_string('-'*100)
    for iter, data in tqdm(enumerate(dataloader)):
        loss, est_data = trainer.eval_step_for_mAP(data)
        eval_dict = est_data['eval_dict']
        ap_calculator.step(eval_dict['batch_pred_map_cls'], eval_dict['batch_gt_map_cls'])
        if ((iter + 1) % cfg.config['log']['print_step']) == 0:
            cfg.log_string('Process: Phase: %s.  %d/%d. Current loss: %s.' % (mode, iter + 1, len(dataloader), str(loss)))
            log_board.update(loss, cfg.config['log']['print_step'], mode)
        loss_recorder.update_loss(loss)


    cfg.log_string(('-'*10 + 'iou_thresh: %f' + '-'*10) % (ap_calculator.ap_thresh))
    metrics_dict = ap_calculator.compute_metrics()
    for key in metrics_dict:
        cfg.log_string('eval %s: %f' % (key, metrics_dict[key]))

    return metrics_dict['mAP'], loss_recorder.loss_recorder

