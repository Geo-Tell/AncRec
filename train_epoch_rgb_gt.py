from net_utils.utils import LossRecorder, LogBoard
from time import time
import torch
import os
from models.anchorrec.SingleObjectDatasetWithRGB import RGB_GT, collate_fn,my_worker_init_fn, denormalize_image
import torch.utils.data
from torch.utils.data import DataLoader


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
        if ((iter + 1) % cfg.config['log']['print_step']) == 0:
            trainer.show_lr()
            cfg.log_string('Process: Phase: %s. Epoch %d: %d/%d. Current loss: %s.' % (
            mode, epoch, iter + 1, len(dataloader), str(loss)))
            log_board.update(loss, cfg.config['log']['print_step'], mode)

    cfg.log_string('=' * 100)
    for loss_name, loss_value in loss_recorder.loss_recorder.items():
        cfg.log_string('Currently the last %s loss (%s) is: %f' % (mode, loss_name, loss_value.avg))
    cfg.log_string('=' * 100)

    return loss_recorder.loss_recorder

def train(cfg, trainer, scheduler,checkpoint, validate = True): #warmup_scheduler,
    start_epoch = scheduler.last_epoch #int(scheduler.last_epoch / 2)
    total_epochs = cfg.config['train']['epochs']
    # min_train_loss = checkpoint.get('min_train_loss')
    min_eval_loss = checkpoint.get('min_eval_loss')
    log_board = LogBoard(os.path.join(cfg._save_path,'runs'))

    train_dataset = RGB_GT(cfg, 'train')
    train_dataloader = DataLoader(dataset=train_dataset,
                                  num_workers=cfg.config['device']['num_workers'],
                                  batch_size=cfg.config['train']['batch_size'],
                                  collate_fn=collate_fn,
                                  shuffle = True,
                                  worker_init_fn=my_worker_init_fn)
    val_dataset = RGB_GT(cfg, 'val')
    val_dataloader = DataLoader(dataset=val_dataset,
                                  num_workers=cfg.config['device']['num_workers'],
                                  batch_size=cfg.config['train']['batch_size'],
                                  collate_fn=collate_fn,
                                  shuffle = False,
                                  worker_init_fn=my_worker_init_fn)



    for epoch in range(start_epoch, total_epochs):
        cfg.log_string('Epoch (%d/%s):' % (epoch + 1, total_epochs))
        cfg.log_string('Switch Phase to train')
        cfg.log_string('-' * 100)

        train_loss_recorder = train_epoch(cfg, epoch + 1, trainer, train_dataloader,log_board) #, warmup_scheduler)
        train_loss = trainer.eval_loss_parser(train_loss_recorder)

        eval_loss_recorder = evaluate_epoch(cfg, trainer, val_dataloader, log_board)
        eval_loss = trainer.eval_loss_parser(eval_loss_recorder)  # 返回平均值

        checkpoint.register_modules(epoch=epoch)
        scheduler.step()
        # scheduler.step(train_loss)
        if epoch == 0 or eval_loss < min_eval_loss: #train_loss < min_train_loss:
            checkpoint.save('min_train_loss')
            min_eval_loss = train_loss
            checkpoint.register_modules(min_eval_loss = min_eval_loss)
            cfg.log_string('Saved the best checkpoint according to eval loss.')
            cfg.log_string('=' * 100)

            for loss_name, loss_value in eval_loss_recorder.items():
                cfg.log_string('Currently the best val loss (%s) is: %f' % (loss_name, loss_value.avg))
            cfg.log_string('=' * 100)

        checkpoint.save('last')
        cfg.log_string('Saved the last checkpoint.')

def evaluate_epoch(cfg,trainer,dataloader,log_board):
    mode = 'val'
    batch_size = cfg.config[mode]['batch_size']
    loss_recorder = LossRecorder(batch_size)
    trainer.net.train(False)
    for iter, data in enumerate(dataloader):
        loss, _ = trainer.eval_step(data)
        loss_recorder.update_loss(loss)

        if ((iter + 1) % cfg.config['log']['print_step']) == 0:
            cfg.log_string('Process: Phase: %s.  %d/%d. Current loss: %s.' % (mode, iter + 1, len(dataloader), str(loss)))
            log_board.update(loss, cfg.config['log']['print_step'], mode)


    return loss_recorder.loss_recorder
