# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import copy, numpy as np
from scipy.spatial.distance import pdist

def compute_ADE(pred, gt, mask):
    # compute Average Displacement Error with mask on frames with missing GT data
    # pred: obj x frames x 2
    # gt:   obj x frames x 2
    # mask: obj x frames

    assert pred.shape == gt.shape, 'error'
    assert mask.shape[0] == gt.shape[0], 'error'
    assert mask.shape[1] == gt.shape[1], 'error'

    dist = np.linalg.norm(pred - gt, axis=-1)       # obj x frames
    dist = dist * mask                              # obj x frames
    valid_num = mask.sum(axis=1)                    # obj
    dist = dist.sum(axis=1)                         # obj
    dist = dist / valid_num                         # obj

    return dist

def compute_FDE(pred, gt, mask):
    # compute Average Displacement Error with mask to select the right last timestamp
    # pred: obj x frames x 2
    # gt:   obj x frames x 2
    # mask: obj x frames

    dist = np.linalg.norm(pred - gt, axis=-1)       # obj x frames
    dist_last = []
    for obj_tmp in range(dist.shape[0]):

        # select the correct last frame, which is not masked out
        mask_tmp = mask[obj_tmp]            # frames
        good_index = np.nonzero(mask_tmp)
        assert len(good_index) == 1, 'error'
        good_index = good_index[0].tolist()
        assert len(good_index) > 0, 'error'
        last_index = np.max(good_index)
        dist_last.append(dist[obj_tmp, last_index])

    dist_last = np.array(dist_last)         # obj

    return dist_last

def best_of_K(ADE_list, FDE_list):
    # compute scene-specific best of K ADE/FDE, i.e., best of sum of all objects in one sample
    # note that this is different from taking best of K for each object independently
    # output is averaged over object

    ADE_list = copy.copy(ADE_list)
    FDE_list = copy.copy(FDE_list)
    num_obj = ADE_list.shape[0]

    # find the error that minimizes all objects in the current timestamp
    ADE_sample  = np.sum(ADE_list, axis=0)       # samples
    ADE_min     = np.min(ADE_sample)   
    best_index  = np.argmin(ADE_sample)   

    # compute average
    ADE_ave = ADE_min / num_obj
    FDE_sample  = np.sum(FDE_list, axis=0)       # samples
    FDE_min = FDE_sample[best_index]
    FDE_ave = FDE_min / num_obj

    return ADE_ave, FDE_ave

def compute_APD(pred):
    # compute Average Pairwise Distance between predictions across samples
    # output is averaged over frames, averaged over objects
    # pred:     samples x obj x frames x 2

    # loop through all objects
    num_obj = pred.shape[1]
    apd_obj = list()
    for obj_index in range(num_obj):
        pred_obj = pred[:, obj_index]       # samples x frames x 2

        # compute APD at each frame and then average
        apd_frame = 0
        for i in range(pred_obj.shape[1]):
            dist = pdist(pred_obj[:, i, :])    # samples x 2 -> (samples x samples-1) / 2, all combinations
            apd_frame += dist / pred_obj.shape[1]

        # average over different combinations of samples
        apd_frame = apd_frame.mean()
        apd_obj.append(apd_frame)

    # average over objects
    apd_obj = np.array(apd_obj).mean()

    return apd_obj

def compute_FPD(pred):
    # compute Final Pairwise Distance between predictions across samples
    # output is averaged over objects
    # pred:     samples x obj x frames x 2

    # loop through all objects
    num_obj = pred.shape[1]
    fpd_obj = list()
    for obj_index in range(num_obj):
        pred_obj = pred[:, obj_index]       # samples x frames x 2

        dist = pdist(pred_obj[:, -1, :])
        fpd = dist.mean()
        fpd_obj.append(fpd)

    # average over objects
    fpd_obj = np.array(fpd_obj).mean()

    return fpd_obj