# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import os, json, copy, numpy as np, sys, pprint
from traj_metrics import compute_ADE, compute_FDE, compute_APD, compute_FPD, best_of_K
from xinshuo_miscellaneous import print_log, AverageMeter, remove_list_from_list, remove_unique_item_from_list
from xinshuo_io import fileparts

len2skip = {10: 1, 20: 2, 50: 5}        # number of skip frames for each pred_len

def complete_data_window(gt_window_obj, start_frame, end_frame, skip):
    # complete GT data for an object within a frame window with dummy data for numpy stacking
    # because some frames of data might be missing
    # gt_window_obj:        N x 4, GT data within a frame window for a specific object

    gt_window_obj = copy.copy(gt_window_obj)
    frame_exist = gt_window_obj[:, 0].tolist()
    frame_all = [*range(start_frame, end_frame, skip)]
    frame_miss, _ = remove_list_from_list(frame_all, frame_exist)
    gt_dummy = np.zeros((len(frame_miss), 4), dtype='float32')
    gt_dummy[:, 0] = np.array(frame_miss)
    gt_window_obj = np.concatenate((gt_window_obj, gt_dummy), axis=0)
    gt_window_obj = gt_window_obj[np.argsort(gt_window_obj[:, 0])]

    return gt_window_obj

def check_eval_windows(start_pred, pred_len, split='test'):
    # start_pred:       the frame index starts to predict in this window, e.g., seq of 0-9 -> 10-19 has start frame at 10
    # pred_len:         the number of frames to predict 
    # split:            train, val, test
    # note that this function assumes past and future len are the same

    if split == 'test':
        reserve_interval = 50
        pred_len = 50
    else:
        reserve_interval = 0
    check = (start_pred - pred_len) % (pred_len * 2 + reserve_interval) == 0

    return check

def compute_metrics(gt_json, pred_json, split='test', total_sample=20):
    results_dir = fileparts(pred_json)[0]
    log_file = os.path.join(results_dir, 'eval_traj_log.txt'); log_file = open(log_file, 'w')
    
    # data loading
    print_log('loading GT from %s' % gt_json, log_file)
    with open(gt_json, 'r') as file: gt_data = json.load(file)
    print_log('loading results from %s' % pred_json, log_file)
    with open(pred_json, 'r') as file: pred_data = json.load(file)

    # 4 metrics x 3 settings x 4 classes x N windows
    metrics = ['ADE', 'FDE', 'APD', 'FPD', 'MsR']  # set up metrics
    stats_meter = {}
    for pred_len in [10, 20, 50]:        # 1s, 2s, 5s prediction settings 
        skip = len2skip[pred_len]

        for obj_class in ['Car', 'Ped', 'Cyc', 'Mot']:
            for metric_tmp in metrics: stats_meter['%s_%d_%s' % (metric_tmp, pred_len, obj_class)] = AverageMeter() 
            gt_obj = gt_data[obj_class]
            if str(pred_len) not in pred_data.keys(): continue
            pred_data_len = pred_data[str(pred_len)]        # get prediction at a specific pred len
            if obj_class not in pred_data_len.keys(): continue
            pred_data_obj = pred_data_len[obj_class]        # get prediction for a specific class

            # loop through based on GT data so that missing prediction can be handled
            num_total_obj_valid, num_miss = 0, 0
            for seqname, gt_seq in gt_obj.items():
                gt_seq = np.array(gt_seq)       # N x 4

                # frame-centric evaluation, go through every possible window of frames, e.g., frame 1-20, 2-21 for 10->10
                frames = np.unique(gt_seq[:, 0]).tolist()
                min_frame, max_frame = frames[0], frames[-1]
                num_windows = int(max_frame - min_frame + 1 - pred_len * 2 + skip)     # this additional + 1 is correct 
                for window_index in range(num_windows):
                    start_frame = int(window_index + min_frame + pred_len)
                    end_frame = int(start_frame + pred_len)

                    # check which window to be dropped
                    check_pass = check_eval_windows(start_frame, pred_len, split)
                    if not check_pass: continue

                    ####### filter frame windows, reserver any window with past data >= 1 frame, future data >= 1 frame
                    # filter out frame window where there is no GT inside
                    gt_window = []
                    for frame in range(start_frame, end_frame, skip):
                        gt_window.append(gt_seq[frame == gt_seq[:, 0], :])
                    gt_window = np.concatenate(gt_window, axis=0)           # N x 4, within a particular time window
                    if gt_window.shape[0] == 0: continue                    
                    ID_list = np.unique(gt_window[:, 1]).tolist()           # get GT IDs

                    # filter out frame window where there is no past observations at all
                    past_start = start_frame - pred_len
                    gt_window_past = []
                    for frame in range(past_start, start_frame, skip):
                        gt_window_past.append(gt_seq[frame == gt_seq[:, 0], :])
                    gt_window_past = np.concatenate(gt_window_past, axis=0) # N x 4, within a particular time window
                    if gt_window_past.shape[0] == 0: continue

                    # filter out ID in this frame window when this ID has no past observations
                    for ID_tmp in ID_list:
                        gt_past = copy.copy(gt_window_past[gt_window_past[:, 1] == float(ID_tmp), :])  # (<=pred_len/skip) x 2, might have missing data
                        num_past_frames = gt_past.shape[0]
                        if num_past_frames == 0: ID_list, _ = remove_unique_item_from_list(ID_list, ID_tmp)
                    if len(ID_list) == 0: continue
                    num_total_obj_valid += len(ID_list)

                    # check prediction if have data in this seq and frame, otherwise all objects are missed
                    try:
                        pred_window = pred_data_obj[seqname][str(start_frame)]               # {sample: {ID: {state, prob}}}
                    except:
                        try: 
                            pred_window = pred_data_obj[seqname][start_frame]                # {sample: {ID: {state, prob}}}
                        except:
                            print('No prediction data found for len of %d, %s, seq %s, frame %s' % \
                                (pred_len, obj_class, seqname, start_frame))
                            num_miss += len(ID_list)
                            continue

                    ######## get predictions within this window
                    sample_count = 0
                    ade_list, fde_list = [], []
                    pred_all_sample, gt_all_sample = [], []
                    
                    # sample must be in the outer loop to compute scene-specific min of ADE
                    for sample_index, pred_tmp in pred_window.items():  
                        best_k_pred, best_k_gt = [], []
                        gt_mask = np.zeros((len(ID_list), int(pred_len/skip)))  # initialize mask on missing frames
                        keep_ID_index = []
                        for ID_count in range(len(ID_list)):
                            ID_tmp = ID_list[ID_count]

                            # check if prediction has data for this ID, otherwise this object is missed
                            try:
                                pred_ID = pred_tmp[str(int(ID_tmp))]
                            except:
                                try:
                                    pred_ID = pred_tmp[int(ID_tmp)] 
                                except:
                                    if int(sample_index) == 0: 
                                        num_miss += 1
                                        print('\nID %d missied in prediction data for len of %d, %s, seq %s, frame %s\n' % \
                                            (ID_tmp, pred_len, obj_class, seqname, start_frame))
                                    continue

                            # get prediction states            
                            state, prob = pred_ID['state'], pred_ID['prob']
                            state = np.array(state)                                     # pred_len/skip x 2
                            assert state.shape[0] == pred_len / skip, 'error'
                            
                            # get GT states
                            gt_ID = copy.copy(gt_window[gt_window[:, 1] == float(ID_tmp), :])      # (<=pred_len/skip) x 2, might have missing data
                            assert gt_ID.shape[0] > 0, 'error'      # this is guaranteed, as this ID exists so there is at least one frame of data for this ID
                            frame_exist = gt_ID[:, 0].tolist()                          # get existing frames before completion
                            if gt_ID.shape[0] < pred_len / skip: 
                                gt_ID = complete_data_window(gt_ID, start_frame, end_frame, skip)     # pred_len/skip x 2

                            # get GT mask
                            frame_exist_index = np.array([frame_tmp - start_frame for frame_tmp in frame_exist])
                            frame_exist_index = (frame_exist_index / skip).astype('uint8')
                            gt_mask[ID_count, frame_exist_index] = 1

                            # store both prediction and GT
                            keep_ID_index.append(ID_count)
                            best_k_pred.append(state)
                            best_k_gt.append(gt_ID[:, 2:])      # take only the state (last two columns)

                        # compute ADE/FDE in this sample
                        gt_mask = gt_mask[keep_ID_index, :]
                        best_k_pred = np.stack(best_k_pred, axis=0)     # obj x pred_len/skip x 2
                        best_k_gt = np.stack(best_k_gt, axis=0)         # obj x pred_len/skip x 2
                        ade_tmp = compute_ADE(best_k_pred, best_k_gt, mask=gt_mask)
                        fde_tmp = compute_FDE(best_k_pred, best_k_gt, mask=gt_mask)
                        ade_list.append(ade_tmp); fde_list.append(fde_tmp)

                        # aggregate all samples
                        pred_all_sample.append(best_k_pred); gt_all_sample.append(best_k_gt)

                        # evaluation allows up to the first 20 samples
                        sample_count += 1
                        if sample_count > total_sample: break

                    # check if every sample has the same shape, i.e., has the same number of predictions of IDs
                    try:
                        ade_array = np.stack(ade_list, axis=1)                  # obj x samples
                        fde_array = np.stack(fde_list, axis=1)                  # obj x samples
                        pred_all_sample = np.stack(pred_all_sample, axis=0)     # samples x obj x frames x 2
                        gt_all_sample   = np.stack(gt_all_sample,   axis=0)
                        assert pred_all_sample.shape == gt_all_sample.shape, 'error'
                    except ValueError:
                        assert False, 'each sample has predictions with different set of IDs, please make sure every sample has the same shape of prediction (same set of IDs)'

                    # compute best of K ADE/FDE
                    ade_best, fde_best = best_of_K(ade_array, fde_array)
                    num_obj = ade_array.shape[0]                                # might less than len(ID_list) as some IDs are not contained in predictions
                    stats_meter['ADE_%s_%s' % (pred_len, obj_class)].update(ade_best, n=num_obj)
                    stats_meter['FDE_%s_%s' % (pred_len, obj_class)].update(fde_best, n=num_obj)

                    # compute APD/FPD for diversity
                    apd = compute_APD(pred_all_sample); fpd = compute_FPD(pred_all_sample)
                    stats_meter['APD_%s_%s' % (pred_len, obj_class)].update(apd, n=num_obj)
                    stats_meter['FPD_%s_%s' % (pred_len, obj_class)].update(fpd, n=num_obj)

                    # only copying the ones under the current class when logging
                    stats_meter_show = {}
                    for metric_tmp, values in stats_meter.items():
                        if (str(pred_len) in metric_tmp) and obj_class in metric_tmp: 
                            stats_meter_show[metric_tmp] = values
                    
                    # logging
                    stats_str = ' '.join([f'{x[:3]}: {y.val:6.3f} ({y.avg:6.3f})' for x, y in stats_meter_show.items()])
                    print_str = 'Len: %d, %s, %s, %03d-%03d, %s\r' % \
                        (pred_len, obj_class, seqname, start_frame, end_frame-1, stats_str)
                    print_log(print_str, log_file, display=False)
                    sys.stdout.write(print_str)
                    sys.stdout.flush()

            # compute missing rate metric
            missrate = num_miss * 1.0 / num_total_obj_valid
            stats_meter['MsR_%s_%s' % (pred_len, obj_class)].update(missrate, n=num_total_obj_valid)
            assert missrate == stats_meter['MsR_%s_%s' % (pred_len, obj_class)].avg, 'error'
            if num_miss > 0:
                print('\n\nMissing IDs: %d, total IDs: %d, miss rate: %.3f for len of %d, %s\n' % \
                    (num_miss, num_total_obj_valid, missrate, pred_len, obj_class))

    ################ logging 
    print_log('', log_file)
    metric_all = dict()
    for pred_len in [10, 20, 50]:
        # compute average ADE/FDE/APD/FPD/MsR
        for metric in ['ADE', 'FDE', 'APD', 'FPD', 'MsR']:
            metric_all['%s_%d' % (metric, pred_len)] = list()
            for obj_class in ['Car', 'Ped', 'Cyc', 'Mot']:
                key_tmp = '%s_%d_%s' % (metric, pred_len, obj_class)                
                num_value = stats_meter[key_tmp].count
                if num_value == 0: tmp_value = np.nan
                else:              tmp_value = stats_meter[key_tmp].avg
                metric_all['%s_%d' % (metric, pred_len)].append(tmp_value)
            metric_all['%s_%d' % (metric, pred_len)] = np.array(metric_all['%s_%d' % (metric, pred_len)]).mean()

        # logging header
        print_log('', log_file)
        print_log('-' * 30 + ' STATS for length of %d ' % pred_len + '-' * 30, log_file)
        print_log(' ' * 5, log_file, same_line=True)
        print_log('ADE_%d'%pred_len+' '*2+'FDE_%d'%pred_len+' '*2+'APD_%d'%pred_len+' '*2+'FPD_%d'%pred_len+' '*2+'MsR_%d'%pred_len+' '*2, \
            log_file, same_line=True)
        print_log('', log_file)

        # logging table: row class, column metrics
        cur_class = 'Car'
        cur_index = 0
        for name, meter in stats_meter.items():
            num_value = meter.count
            if num_value == 0: final_value = np.nan
            else:              final_value = meter.avg
            if str(pred_len) not in name: continue

            # check to see if switch from one class to the other
            if cur_class not in name:
                print_log('', log_file)
                cur_class = name[-3:]
                cur_index = 0

            # print all errors for this class
            if cur_index == 0: print_log('%s: ' % name[-3:], log_file, same_line=True)
            print_log('%6.3f  ' % final_value, log_file, same_line=True)
            cur_index += 1

        # print average value
        print_log('', log_file)
        print_log('Ave: ', log_file, same_line=True)
        for metric in ['ADE', 'FDE', 'APD', 'FPD', 'MsR']:
            ave_value = metric_all['%s_%d' % (metric, pred_len)]
            print_log('%6.3f  ' % ave_value, log_file, same_line=True)

        # print end of the logging
        print_log('', log_file)
        print_log('-' * 84, log_file)
    log_file.close()

    # filter out metrics other than len of 20
    output = dict()
    for key, value in metric_all.items():        
        if '20' in key:      # only add length of 20 to output metric for EvalAI leaderboard
            output[key] = value
    pprint.pprint(output)                   # print output metrics

    return output

if __name__ == '__main__':
    test_annotation_file = '../data/traj_val_anno.json'
    user_submission_file = '../data/traj_val.json'
    split = 'val'
    compute_metrics(test_annotation_file, user_submission_file, split)
