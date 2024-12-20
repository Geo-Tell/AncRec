# Testing functions.
# author: ynie
# date: April, 2020
from net_utils.utils import LossRecorder
from time import time
from net_utils.ap_helper import APCalculator
from net_utils.quad_ap_helper import QUADAPCalculator
import os
import numpy as np

def test_func(cfg, tester, test_loader):
    '''
    test function.
    :param cfg: configuration file
    :param tester: specific tester for networks
    :param test_loader: dataloader for testing
    :return:
    '''
    mode = cfg.config['mode']
    batch_size = 1
    loss_recorder = LossRecorder(batch_size)
    AP_IOU_THRESHOLDS = cfg.config[mode]['ap_iou_thresholds']

    if 'test_phase' in cfg.config['test']:
        test_phase = cfg.config['test']['test_phase']
    else:
        test_phase = cfg.config[cfg.config['mode']]['phase']

    if type(test_phase)==list and len(test_phase)==1:
        test_phase = test_phase[0]

    if  test_phase == 'detection':
        ap_calculator_list = [APCalculator(iou_thresh, cfg.dataset_config.class2type, False)
                              for iou_thresh in AP_IOU_THRESHOLDS]
        ap_calculator_lists = [ap_calculator_list]
    elif test_phase == 'quad_detection':
        ap_calculator_list = [QUADAPCalculator(iou_thresh, cfg.dataset_config.class2quad) \
                               for iou_thresh in AP_IOU_THRESHOLDS]
        ap_calculator_lists = [ap_calculator_list]
    elif test_phase == 'joint_detection':
        object_ap_calculator_list = [APCalculator(iou_thresh, cfg.dataset_config.class2type, False)
                              for iou_thresh in AP_IOU_THRESHOLDS]
        quad_ap_calculator_list = [QUADAPCalculator(iou_thresh, cfg.dataset_config.class2quad) \
                                  for iou_thresh in AP_IOU_THRESHOLDS]
        ap_calculator_lists = [object_ap_calculator_list,quad_ap_calculator_list]


    cfg.log_string('-'*100)
    for iter, data in enumerate(test_loader):
        loss, est_data = tester.test_step(data)
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
                else: raise NotImplementedError

        # loss_recorder.update_loss(loss)
        #
        if ((iter + 1) % cfg.config['log']['print_step']) == 0:
            cfg.log_string('Process: Phase: %s. Epoch %d: %d/%d. Current loss: %s.' % (
            mode, 0, iter + 1, len(test_loader), str({key: np.mean(item) for key, item in loss.items()})))

        # visualize intermediate results.
        # if cfg.config['log']['save_results']: # and iter < 10: #(iter+1) % cfg.config['generation']['dump_step']== 0 :
        #     if test_phase == 'detection':
        #         tester.visualize_step(data, est_data) #phase, iter, gt_data, our_data, eval_dict
        #     elif test_phase == 'quad_detection':
        #         tester.visualize_quad(mode, iter, data, est_data)
        #     elif test_phase == 'joint_detection':
        #         tester.visualize_step(mode, iter, data, est_data)
        #         tester.visualize_quad(mode, iter, data, est_data)

    # return ap_calculator_lists
    return loss_recorder.loss_recorder, ap_calculator_lists

def test(cfg, tester, test_loader):
    '''
    train epochs for network
    :param cfg: configuration file
    :param tester: specific tester for networks
    :param test_loader: dataloader for testing
    :return:
    '''
    cfg.log_string('-' * 100)
    # set mode
    mode = cfg.config['mode']
    tester.net.train(mode == 'train')
    start = time()
    # ap_calculator_lists = test_func(cfg, tester, test_loader)
    test_loss_recoder, ap_calculator_lists = test_func(cfg, tester, test_loader)
    cfg.log_string('Test time elapsed: (%f).' % (time()-start))
    for key, test_loss in test_loss_recoder.items():
        cfg.log_string('Test loss (%s): %f' % (key, test_loss.avg))

    # Evaluate average precision
    for ap_calculator_list in ap_calculator_lists:
        for i, ap_calculator in enumerate(ap_calculator_list):
            if type(ap_calculator).__name__ == 'QUADAPCalculator':
                if i ==0:
                    f1 = ap_calculator.compute_F1(calculated=True)
                    cfg.log_string(f'F1 scores: {f1}')
            else:
                if ap_calculator.distance_method == 'iou':# 0,1
                    cfg.log_string(('-'*10 + 'iou_thresh: %f' + '-'*10) % (ap_calculator.ap_thresh))
                    metrics_dict = ap_calculator.compute_metrics()
                else:
                    cfg.log_string(('-'*10 + '%s_thresh: %f' + '-'*10) % (ap_calculator.distance_method, ap_calculator.ap_thresh))
                    metrics_dict = ap_calculator.compute_distance_metrics()

                for key in metrics_dict:
                    cfg.log_string('eval %s: %f' % (key, metrics_dict[key]))