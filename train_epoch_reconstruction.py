from net_utils.utils import LossRecorder, LogBoard
from time import time
import torch
import os
from models.anchorrec.SingleObject import SingleObject, collate_fn,my_worker_init_fn
import torch.utils.data
from torch.utils.data import DataLoader
from utils.pc_util import write_ply
import trimesh


def train_epoch(cfg, epoch, trainer, dataloader, log_board):
    mode = 'train'
    batch_size = cfg.config[mode]['batch_size']
    loss_recorder = LossRecorder(batch_size)
    trainer.net.train(mode == 'train')
    trainer.net.module.set_mode()
    vis_path = cfg.config['log']['vis_path']
    for iter, data in enumerate(dataloader):
        loss = trainer.train_step(data)
        loss_recorder.update_loss(loss)
        torch.cuda.empty_cache()

        if ((iter + 1) % cfg.config['log']['print_step']) == 0:
            trainer.show_lr()
            cfg.log_string('Process: Phase: %s. Epoch %d: %d/%d. Current loss: %s.' % (
            mode, epoch, iter + 1, len(dataloader), str(loss)))
            log_board.update(loss, cfg.config['log']['print_step'], mode)

        # visualize intermediate results.
        if (iter % cfg.config['log']['vis_step_train']) == 0 and cfg.config['log']['save_results'] == True \
        and epoch % cfg.config['log']['vis_step_epoch'] == 0 and epoch > cfg.config['log']['vis_start_epoch'] == 0:
            est_data = trainer.net.module.generate(data)
            for bid, mesh in enumerate(est_data['meshes']):
                # pred mesh
                mesh.export(os.path.join(vis_path, 'train_epoch_%s_iter_%s_pred_%s.ply' % (epoch, iter, bid)))
                # gt z mesh
                if 'gt_meshes' in est_data:
                    est_data['gt_meshes'][bid].export(
                        os.path.join(vis_path, 'train_epoch_%s_iter_%s_gt_z_%s.ply' % (epoch, iter, bid)))
                # gt mesh
                gt_mesh = trimesh.load(data['mesh_pth'][bid], process=False)
                gt_mesh.export(os.path.join(vis_path, 'train_epoch_%s_iter_%s_gt_%s.ply' % (epoch, iter, bid)))

                # input pts
                write_ply(data['input_points'][bid].detach().cpu().numpy(), os.path.join(vis_path,
                                                                                          'train_epoch_%s_iter_%s_objpts_%s.ply' % (
                                                                                          epoch, iter, bid)))

    cfg.log_string('=' * 100)
    for loss_name, loss_value in loss_recorder.loss_recorder.items():
        cfg.log_string('Currently the last %s loss (%s) is: %f' % (mode, loss_name, loss_value.avg))
    cfg.log_string('=' * 100)

    return loss_recorder.loss_recorder

def train(cfg, trainer, scheduler,checkpoint, validate = False):
    start_epoch = scheduler.last_epoch #int(scheduler.last_epoch / 2)
    total_epochs = cfg.config['train']['epochs']
    # min_eval_loss = checkpoint.get('min_loss')
    # min_train_loss = checkpoint.get('min_loss')
    min_eval_loss = 100
    log_board = LogBoard(os.path.join(cfg._save_path,'runs'))
    train_dataset = SingleObject(cfg, 'train')
    train_dataloader = DataLoader(dataset=train_dataset,
                                  num_workers=cfg.config['device']['num_workers'],
                                  batch_size=cfg.config['train']['batch_size'],
                                  collate_fn=collate_fn,
                                  shuffle = True,
                                  worker_init_fn=my_worker_init_fn)

    if validate:
        val_dataset = SingleObject(cfg, 'val')
        val_dataloader = DataLoader(dataset = val_dataset,
                                      num_workers = cfg.config['device']['num_workers'],
                                      batch_size = cfg.config['val']['batch_size'],
                                      collate_fn = collate_fn,
                                      worker_init_fn=my_worker_init_fn)


    for epoch in range(start_epoch, total_epochs):
        cfg.log_string('-' * 100)
        cfg.log_string('Epoch (%d/%s):' % (epoch + 1, total_epochs))
        # bnm_scheduler.show_momentum()
        start = time()
        cfg.log_string('-' * 100)
        cfg.log_string('Switch Phase to train')
        cfg.log_string('-' * 100)

        train_loss_recorder = train_epoch(cfg, epoch + 1, trainer, train_dataloader,log_board) #, warmup_scheduler)
        train_loss = trainer.eval_loss_parser(train_loss_recorder)

        checkpoint.register_modules(epoch=epoch, min_loss=train_loss)
        # bnm_scheduler.step()
        cfg.log_string('Epoch (%d/%s) Time elapsed: (%f).' % (epoch + 1, total_epochs, time() - start))
        if validate:
            cfg.log_string('-' * 100)
            cfg.log_string('Switch Phase to val')
            cfg.log_string('-' * 100)
            eval_loss_recorder = evaluate_epoch(cfg, trainer, val_dataloader,log_board,epoch)
            eval_loss = trainer.eval_loss_parser(eval_loss_recorder)

            checkpoint.register_modules(epoch=epoch, min_loss=eval_loss)

            scheduler.step() #exponential scheduler
            # scheduler.step(eval_loss)
            if epoch == 0 or eval_loss < min_eval_loss:
                checkpoint.save('min_loss')
                min_eval_loss = eval_loss
                cfg.log_string('Saved the best checkpoint according to eval loss.')
                cfg.log_string('=' * 100)
                for loss_name, loss_value in eval_loss_recorder.items():
                    cfg.log_string('Currently the best val loss (%s) is: %f' % (loss_name, loss_value.avg))
                cfg.log_string('=' * 100)
            # bnm_scheduler.step()
            cfg.log_string('Epoch (%d/%s) Time elapsed: (%f).for validation' % (epoch + 1, total_epochs, time() - start))

        # checkpoint.save(str(epoch))
        checkpoint.save('last')
        cfg.log_string('Saved the latest checkpoint.')
        if epoch == total_epochs - 1:
            checkpoint.save(str(epoch+ 1))
            cfg.log_string('Saved the last checkpoint.')

def evaluate_epoch(cfg,trainer,dataloader,log_board, epoch):
    mode = 'val'
    batch_size = cfg.config[mode]['batch_size']
    loss_recorder = LossRecorder(batch_size)
    trainer.net.train(False)
    vis_path = cfg.config['log']['vis_path']
    for iter, data in enumerate(dataloader):
        # loss,_ = trainer.eval_step_for_mAP(data)
        loss, _ = trainer.eval_step(data)
        torch.cuda.empty_cache()
        loss_recorder.update_loss(loss)

        if ((iter + 1) % cfg.config['log']['print_step']) == 0:
            cfg.log_string('Process: Phase: %s.  %d/%d. Current loss: %s.' % (mode, iter + 1, len(dataloader), str(loss)))
            log_board.update(loss, cfg.config['log']['print_step'], mode)

        vis_iter = 0
        # visualize intermediate results.
        if (iter % cfg.config['log']['vis_step_val']) == 0 and cfg.config['log']['save_results'] == True \
                and epoch > cfg.config['log']['vis_start_epoch'] and epoch % cfg.config['log']['vis_step_epoch'] == 0:
            est_data = trainer.net.module.generate(data)
            for bid, mesh in enumerate(est_data['meshes']):
                # pred mesh
                mesh.export(os.path.join(vis_path, 'val_epoch_%s_iter_%s_pred_%s.ply' % (epoch, iter, bid)))
                if vis_iter==0:
                    # gt z mesh
                    est_data['gt_meshes'][bid].export(os.path.join(vis_path,  'val_epoch_%s_iter_%s_gt_z_%s.ply' % (epoch, iter, bid)))
                    # gt mesh
                    gt_mesh = trimesh.load(data['mesh_pth'][bid], process=False)
                    gt_mesh.export(os.path.join(vis_path, 'val_epoch_%s_iter_%s_gt_%s.ply' % (epoch, iter, bid)))
                    # input pts
                    write_ply(data['object_points'][bid].detach().cpu().numpy(), os.path.join(vis_path,
                                                    'val_epoch_%s_iter_%s_objpts_%s.ply' % (epoch, iter, bid)))
                vis_iter += 1


    return loss_recorder.loss_recorder