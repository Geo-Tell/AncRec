from net_utils.utils import LossRecorder
from time import time
from net_utils.ap_helper import APCalculator
from tqdm import tqdm
from models.anchorrec.SingleObjectDatasetWithRGB import AnchorRec_ScanNet_For_Refine, collate_fn,my_worker_init_fn
from torch.utils.data import DataLoader


def test_func(cfg, tester, test_loader, ap_calculator_list):
    mode = cfg.config['mode']
    batch_size = 1
    loss_recorder = LossRecorder(batch_size)
    cfg.log_string('-'*100)
    for iter, data in tqdm(enumerate(test_loader)):
        loss, est_data = tester.test_step(data)
        eval_dict = est_data['eval_dict']
        for ap_calculator  in ap_calculator_list:
           ap_calculator.step(eval_dict['batch_pred_map_cls'], eval_dict['batch_gt_map_cls'])
           # ap_calculator.step_pred(eval_dict['batch_pred_map_cls']) #, eval_dict['batch_gt_map_cls'])

        if cfg.config['log']['save_results']: # and iter < 10: #(iter+1) % cfg.config['generation']['dump_step']== 0 :
            tester.visualize_step(data, est_data)

        # loss_recorder.update_loss(loss)
        #
        # if ((iter + 1) % cfg.config['log']['print_step']) == 0:
        #     cfg.log_string('Process: Phase: %s. Epoch %d: %d/%d. Current loss: %s.' % (
        #     mode, 0, iter + 1, len(test_loader), str({key: np.mean(item) for key, item in loss.items()})))
        #
        # # visualize intermediate results.
        # if cfg.config['log']['save_results']: # and iter < 10: #(iter+1) % cfg.config['generation']['dump_step']== 0 :
        #     tester.visualize_step(mode, iter, data, est_data, eval_dict) #phase, iter, gt_data, our_data, eval_dict

    # return ap_calculator_lists
    return loss_recorder.loss_recorder, ap_calculator_list

def test(cfg, tester):
    cfg.log_string('-' * 100)
    # set mode
    mode = cfg.config['mode']
    tester.net.train(mode == 'train')
    start = time()
    # ap_calculator_lists = test_func(cfg, tester, test_loader)
    dataset = AnchorRec_ScanNet_For_Refine(cfg, 'test')
    test_loader = DataLoader(dataset = dataset,
                                  num_workers=cfg.config['device']['num_workers'],
                                  batch_size= 1,
                                  collate_fn=collate_fn,
                                  worker_init_fn=my_worker_init_fn)

    AP_IOU_THRESHOLDS = cfg.config[mode]['ap_iou_thresholds']
    ap_calculator_list = [APCalculator(iou_thresh, cfg.dataset_config.class2type, False)
                          for iou_thresh in AP_IOU_THRESHOLDS]

    # for iter, data in tqdm(enumerate(scannet_test_loader)):
    #     parsed_gts = parse_groundtruths(data, cfg.dataset_config)
    #     parsed_gts['scan_name'] = data['scan_name']
    #     batch_gt_map_cls = assembly_gt_map_cls(parsed_gts)
    #     for ap_calculator  in ap_calculator_list:
    #        ap_calculator.step_gt(batch_gt_map_cls) #, eval_dict['batch_gt_map_cls'])
    # for ap_calculator in ap_calculator_list:
    #     ap_calculator.reset2() #scan_cnt = 0

    test_loss_recoder, ap_calculator_list = test_func(cfg, tester, test_loader, ap_calculator_list)


    # Evaluate average precision
    for i, ap_calculator in enumerate(ap_calculator_list):
        cfg.log_string(('-'*10 + 'iou_thresh: %f' + '-'*10) % (ap_calculator.ap_thresh))
        metrics_dict = ap_calculator.compute_metrics()
        for key in metrics_dict:
            cfg.log_string('eval %s: %f' % (key, metrics_dict[key]))