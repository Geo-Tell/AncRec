from net_utils.utils import LossRecorder, LogBoard
from time import time
import torch
from net_utils.ap_helper import APCalculator
from net_utils.quad_ap_helper import QUADAPCalculator
import os


def train_epoch(cfg, epoch, trainer, dataloader, log_board):
    mode = 'train'
    batch_size = cfg.config[mode]['batch_size']
    loss_recorder = LossRecorder(batch_size)
    trainer.net.train(mode == 'train')
    trainer.net.module.set_mode()

    for iter, data in enumerate(dataloader):
        # print("load data %s from dataloader"% cfg.LOCAL_RANK)
        loss = trainer.train_step(data)
        torch.cuda.empty_cache()
        loss_recorder.update_loss(loss)
        # trainer.visualize_step(epoch, mode, iter, data)

        if cfg.LOCAL_RANK == 0 and ((iter + 1) % cfg.config['log']['print_step']) == 0:
            cfg.log_string('Process: Phase: %s. Epoch %d: %d/%d. Current loss: %s.' % (
            mode, epoch, iter + 1, len(dataloader), str(loss)))
            if log_board is not None:
                log_board.update(loss, cfg.config['log']['print_step'], mode)

    # visualize intermediate results.
    if cfg.LOCAL_RANK == 0:
        if epoch> 100 and epoch % cfg.config['log']['vis_step'] == 0 and cfg.config['log']['save_results']:
            print('visualizing iteration: %s'%iter)
            trainer.visualize_step(epoch, mode, iter, data)

        cfg.log_string('=' * 100)
        for loss_name, loss_value in loss_recorder.loss_recorder.items():
            cfg.log_string('Currently the last %s loss (%s) is: %f' % (mode, loss_name, loss_value.avg))
        cfg.log_string('=' * 100)

    return loss_recorder.loss_recorder

def train(cfg, trainer, scheduler, bnm_scheduler, checkpoint, train_loader, val_loader, train_sampler=None, validate = False):
    start_epoch = int(scheduler.last_epoch)
    # print("epoch:%s"%start_epoch)
    total_epochs = cfg.config['train']['epochs']
    eval_map_start_epoch = cfg.config['train']['eval_map_start_epoch']
    min_eval_loss = checkpoint.get('min_loss')
    min_train_loss = checkpoint.get('min_train_loss')
    log_board = LogBoard(os.path.join(cfg._save_path,'runs'))if cfg.LOCAL_RANK == 0 else None

    dataloaders = {'train': train_loader, 'val': val_loader}
    max_eval_mAP = 0
    max_f1_score = 0
    # train_dataset = SingleObject(cfg, 'train')
    for epoch in range(start_epoch, total_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        trainer.show_lr()
        bnm_scheduler.show_momentum()

        cfg.log_string('-' * 100)
        cfg.log_string('Epoch (%d/%s):' % (epoch, total_epochs))
        start = time()
        cfg.log_string('-' * 100)
        cfg.log_string('Switch Phase to train')
        cfg.log_string('-' * 100)

        train_loss_recorder = train_epoch(cfg, epoch, trainer, dataloaders['train'],log_board)
        train_loss = trainer.eval_loss_parser(train_loss_recorder)
        checkpoint.save('last')
        cfg.log_string('Saved the last checkpoint.')
        # eval_map_start_epoch = 0
        # if epoch < eval_map_start_epoch and epoch%50 !=0:
        #     continue
        # if epoch%10 !=0:
        #     continue

        if validate:
            torch.cuda.empty_cache()
            cfg.log_string('-' * 100)
            cfg.log_string('Switch Phase to val')
            cfg.log_string('-' * 100)

            if epoch < eval_map_start_epoch:
                eval_loss_recorder = evaluate_epoch(cfg, trainer, dataloaders['val'],log_board)
            elif cfg.config['val']['phase'] == 'detection':
                mAP_object_sum, eval_loss_recorder = \
                    evaluate_epoch_detection(cfg, trainer,dataloaders['val'],log_board,cfg.config['val']['phase'])
                if epoch ==eval_map_start_epoch or  mAP_object_sum > max_eval_mAP:
                    checkpoint.save('best')
                    max_eval_mAP =  mAP_object_sum
                    cfg.log_string('Saved the best checkpoint according to mAP.')
                    cfg.log_string('Currently the best object 0.5 IOU mAP is: %f'% max_eval_mAP)
            elif cfg.config['val']['phase'] == 'quad_detection':
                quad_f1_score, eval_loss_recorder = \
                    evaluate_epoch_detection(cfg, trainer,dataloaders['val'],log_board, cfg.config['val']['phase'])
                if epoch == eval_map_start_epoch or quad_f1_score > max_f1_score:
                    checkpoint.save('best')
                    max_f1_score = quad_f1_score
                    cfg.log_string('Saved the best checkpoint according to F1 score.')
                    cfg.log_string('Currently the best f1 score is: %f' % (max_f1_score))
                save_quad_ckpt_thresh = cfg.config['val']['save_quad_ckpt_thresh'] if 'save_quad_ckpt_thresh' in cfg.config['val'] else 0.7
                if quad_f1_score >save_quad_ckpt_thresh:
                     checkpoint.save('epoch_%s'%epoch)
                     cfg.log_string('Saved checkpoint because F1 score is larger than 0.67.')
            else:
                mAP_object_sum, quad_f1_score, eval_loss_recorder  = \
                    evaluate_epoch_detection(cfg,trainer, dataloaders['val'],log_board,cfg.config['val']['phase'])
                if cfg.config['val']['save_best_according_to'] == 'quad_detection':
                    if epoch == eval_map_start_epoch or quad_f1_score > max_f1_score:
                        checkpoint.save('best')
                        max_eval_mAP = mAP_object_sum
                        max_eval_mAP_f1_score = quad_f1_score
                        cfg.log_string('Saved the best checkpoint according to F1 score.')
                        cfg.log_string('Currently the best f1 score is: %f , '
                                       'when mAP socre is: %f' % (max_eval_mAP_f1_score, max_eval_mAP))
                else:
                    if epoch == eval_map_start_epoch or mAP_object_sum > max_eval_mAP:
                        checkpoint.save('best')
                        max_eval_mAP = mAP_object_sum
                        max_eval_mAP_f1_score = quad_f1_score
                        cfg.log_string('Saved the best checkpoint according to mAP.')
                        cfg.log_string('Currently the best object 0.5 IOU mAP is: %f , '
                                       'when f1 score is: %f' % (max_eval_mAP, max_eval_mAP_f1_score))

            eval_loss = trainer.eval_loss_parser(eval_loss_recorder)  # 返回平均值
            checkpoint.register_modules(epoch=epoch, min_loss=eval_loss)
            if type(scheduler)== torch.optim.lr_scheduler.ReduceLROnPlateau:
                scheduler.step(eval_loss)
            else:
                scheduler.step()
            if epoch == 0 or eval_loss < min_eval_loss:
                min_eval_loss = eval_loss
                if cfg.LOCAL_RANK == 0:
                    checkpoint.save('min_loss')
                    # checkpoint.save('min_train_loss')
                    cfg.log_string('Saved the best checkpoint according to eval loss.')
                    cfg.log_string('=' * 100)
                    for loss_name, loss_value in eval_loss_recorder.items():
                        cfg.log_string('Currently the best val loss (%s) is: %f' % (loss_name, loss_value.avg))
                    cfg.log_string('=' * 100)

        else:
            checkpoint.register_modules(epoch=epoch, min_loss=train_loss)
            scheduler.step(train_loss)




def evaluate_epoch_detection(cfg, trainer, dataloader, log_board, val_stage='detection'):
    mode = 'val'
    batch_size = cfg.config[mode]['batch_size']
    loss_recorder = LossRecorder(batch_size)
    AP_IOU_THRESHOLDS = cfg.config[mode]['ap_iou_thresholds']


    if val_stage == 'detection':
        ap_calculator_list = [APCalculator(iou_thresh, cfg.dataset_config.class2type)
                              for iou_thresh in AP_IOU_THRESHOLDS]
        ap_calculator_lists = [ap_calculator_list]
    elif val_stage == 'quad_detection' :
        ap_calculator_list = [QUADAPCalculator(iou_thresh, cfg.dataset_config.class2quad) \
                              for iou_thresh in AP_IOU_THRESHOLDS]
        ap_calculator_lists = [ap_calculator_list]
    else:
        object_ap_calculator_list = [APCalculator(iou_thresh, cfg.dataset_config.class2type)
                                     for iou_thresh in AP_IOU_THRESHOLDS]
        quad_ap_calculator_list = [QUADAPCalculator(iou_thresh, cfg.dataset_config.class2quad) \
                                   for iou_thresh in AP_IOU_THRESHOLDS]
        ap_calculator_lists = [object_ap_calculator_list, quad_ap_calculator_list]

    for iter, data in enumerate(dataloader):
        loss, est_data = trainer.eval_step_for_mAP(data)
        eval_dict = est_data['eval_dict']
        for ap_calculator_list in ap_calculator_lists:
            for ap_calculator  in ap_calculator_list:
                if type(ap_calculator).__name__ == 'APCalculator':
                    ap_calculator.step(eval_dict['batch_pred_map_cls'], eval_dict['batch_gt_map_cls'])
                elif type(ap_calculator).__name__ == 'QUADAPCalculator':
                    ap_calculator.step(eval_dict['batch_pred_quad_map_cls'], eval_dict['batch_gt_quad_map_cls'],
                                            eval_dict['batch_pred_quad_corners_list'],
                                            eval_dict['batch_gt_quad_corners_list'],
                                            eval_dict['batch_gt_quad_horizontal_list'], eval_dict['has_quad_ind'])
                else:
                    raise NotImplementedError

        if ((iter + 1) % cfg.config['log']['print_step']) == 0:
            cfg.log_string('Process: Phase: %s.  %d/%d. Current loss: %s.' % (mode, iter + 1, len(dataloader), str(loss)))
            log_board.update(loss, cfg.config['log']['print_step'], mode)
        loss_recorder.update_loss(loss)

    # 默认只使用一个Iou （0.5）
    mAP_object_sum = 0

    for ap_calculator_list in ap_calculator_lists:
        for i, ap_calculator in enumerate(ap_calculator_list):
            cfg.log_string(('-' * 10 + 'iou_thresh: %f' + '-' * 10) % (AP_IOU_THRESHOLDS[i]))
            metrics_dict = ap_calculator.compute_metrics()
            if type(ap_calculator).__name__ == 'QUADAPCalculator':
                if i == 0:
                    f1 = ap_calculator.compute_F1(calculated=True)
                    cfg.log_string(f'F1 scores: {f1}')
            else:
                mAP_object_sum  += metrics_dict['mAP']
            for key in metrics_dict:
                cfg.log_string('eval %s: %f' % (key, metrics_dict[key]))

    if val_stage == 'detection':
        cfg.log_string('eval object 0.5iou mAP: %f' % mAP_object_sum)
        return mAP_object_sum, loss_recorder.loss_recorder #, f1,
    elif val_stage == 'quad_detection':
        cfg.log_string('eval quad f1 score: %f' % f1)
        return f1, loss_recorder.loss_recorder
    elif val_stage == 'joint_detection':
        cfg.log_string('eval object 0.5iou mAP: %f' % mAP_object_sum)
        cfg.log_string('eval quad f1 score: %f' % f1)
        return mAP_object_sum, f1, loss_recorder.loss_recorder

def evaluate_epoch(cfg,trainer,dataloader,log_board):
    mode = 'val'
    batch_size = cfg.config[mode]['batch_size']
    loss_recorder = LossRecorder(batch_size)
    trainer.net.train(mode == 'train')
    for iter, data in enumerate(dataloader):
        loss, _ = trainer.eval_step(data)
        torch.cuda.empty_cache()
        loss_recorder.update_loss(loss)

        if cfg.LOCAL_RANK == 0 and ((iter + 1) % cfg.config['log']['print_step']) == 0:
            cfg.log_string('Process: Phase: %s.  %d/%d. Current loss: %s.' % (mode, iter + 1, len(dataloader), str(loss)))
            log_board.update(loss, cfg.config['log']['print_step'], mode)
    return loss_recorder.loss_recorder
