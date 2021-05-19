import argparse, json, os, sys, torch, numpy as np, random
sys.path.append(os.getcwd())
from attrdict import AttrDict
from sgan.data.loader import data_loader
from sgan.models import TrajectoryGenerator
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path
from xinshuo_io import fileparts, mkdir_if_missing
from xinshuo_miscellaneous import get_timestring, AverageMeter, print_log

def prepare_seed(rand_seed):
    np.random.seed(rand_seed)
    random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='val', type=str)
prepare_seed(0)

# for AIODrive
def int2seqname(final_id):
    seq_id = final_id % 1000
    city_id = (final_id - seq_id) / 1000
    if city_id == 10: seqname = 'Town%02dHD_seq%04d' % (city_id, seq_id)
    else:             seqname = 'Town%02d_seq%04d' % (city_id, seq_id)
    return seqname

def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(obs_len=args.obs_len, pred_len=args.pred_len, embedding_dim=args.embedding_dim, \
        encoder_h_dim=args.encoder_h_dim_g, decoder_h_dim=args.decoder_h_dim_g, mlp_dim=args.mlp_dim, num_layers=args.num_layers, \
        noise_dim=args.noise_dim, noise_type=args.noise_type, noise_mix_type=args.noise_mix_type, pooling_type=args.pooling_type, \
        pool_every_timestep=args.pool_every_timestep, dropout=args.dropout, bottleneck_dim=args.bottleneck_dim, \
        neighborhood_size=args.neighborhood_size, grid_size=args.grid_size, batch_norm=args.batch_norm)
    generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.eval()
    return generator

def best_of_K(error, seq_start_end, err_type='ADE'):
    sum_, num_total = 0, 0
    error = torch.stack(error, dim=1)       # objects x samples
    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]           # objects (in a short seq) x samples
        
        # filter out nan values
        invalid_bool = torch.isnan(_error[:, 0])
        valid_index = torch.nonzero(1 - invalid_bool.float()).squeeze(1)
        _error = _error[valid_index, :]     # obj x samples

        # collect number of valid objects
        num_obj = _error.size(0)
        num_total += num_obj

        # find the error that minimizes all objects in the current timestamp
        _error = torch.sum(_error, dim=0)   # sample
        _error = torch.min(_error)          # 1,
        sum_ += _error

        ave_frame = _error / num_obj

    ave = sum_ / num_total
    return ave, num_total

def evaluate(args, loader, generator, num_samples, path):
    # ade_outer, fde_outer = [], []
    ade_all, fde_all = AverageMeter(), AverageMeter()
    total_obj = 0
    pred_len = args.pred_len
    dataset_name = args.dataset_name
    obj_class = dataset_name.split('_')[1][:3]

    save_dir, _, _ = fileparts(path)
    save_dir = os.path.join(save_dir, 'results_%s' % get_timestring()); mkdir_if_missing(save_dir)
    result_file_single = os.path.join(save_dir, 'results.json')
    result_dict = dict()
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, seq_start_end, id_frame) = batch
            # obs_traj          frames x objects x 2
            # pred_traj_gt      frames x objects x 2
            # seq_Start_end     start, end of ped index in each timestamp, used for pooling at every timestamp
            # id_frame          2frames x objects x 3
            # loss_mask         objects x 2frames 

            num_obs = obs_traj.size(0)
            num_objects = obs_traj.size(1)
            id_frame_pred = id_frame[num_obs:]      # frames x obj x 3
            loss_mask_pred = loss_mask[:, num_obs:]         # objects x seq_len

            ade, fde = [], []
            for sample_index in range(num_samples):
                pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end)
                pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])      # frames x objects x 2

                # save results
                for object_index in range(num_objects):
                    id_frame_tmp = id_frame_pred[:, object_index, :]
                    frame = int(id_frame_tmp[0, 0].item())

                    # seqname should be the same across frames
                    seq = np.unique(id_frame_tmp[:, -1].cpu().clone().numpy())
                    assert len(seq) == 1, 'error'
                    seqname = int2seqname(seq[0])         # AIODrive only

                    # seqname should be the same across frames
                    ID = np.unique(id_frame_tmp[:, 1].cpu().clone().numpy())
                    assert len(ID) == 1, 'error'
                    ID = int(ID[0])

                    # saving to individual frames
                    final_results = torch.cat([id_frame_tmp[:, :2], pred_traj_fake[:, object_index, :]], axis=-1).cpu().clone().numpy()
                    save_path = os.path.join(save_dir, seqname, 'frame_%06d' % (frame), 'sample_%03d' % sample_index+'.txt')
                    mkdir_if_missing(save_path)
                    with open(save_path, 'a') as f: np.savetxt(f, final_results, fmt="%.3f")

                    # saving to a single file, result format
                    # {seqname1: {frame1: {sample1: {ID1: {state: N x 2, prob: 1}}}, seqname2, ...}
                    if seqname not in result_dict.keys(): result_dict[seqname] = dict()
                    if frame not in result_dict[seqname].keys(): result_dict[seqname][frame] = dict()
                    if sample_index not in result_dict[seqname][frame].keys(): result_dict[seqname][frame][sample_index] = dict()
                    if ID not in result_dict[seqname][frame][sample_index].keys(): result_dict[seqname][frame][sample_index][ID] = dict()
                    result_dict[seqname][frame][sample_index][ID]['state'] = pred_traj_fake[:, object_index, :].cpu().clone().numpy().tolist()
                    result_dict[seqname][frame][sample_index][ID]['prob']  = 1.0

                # compute ADE
                ade_tmp = displacement_error(pred_traj_fake, pred_traj_gt, mode='raw', mask=loss_mask_pred)      # list of ade for each object in the batch
                ade.append(ade_tmp)             # list of error for all samples

                # select the right last timestamp for FDE computation, i.e., not select the last frame if masked out
                pred_traj_last = []
                gt_traj_last = []
                for obj_tmp in range(num_objects):
                    loss_mask_tmp = loss_mask_pred[obj_tmp]         # seq_len
                    good_index = torch.nonzero(loss_mask_tmp)
                    if torch.nonzero(loss_mask_tmp).size(0) == 0:
                        pred_traj_last.append(torch.zeros(2).cuda() / 0)
                        gt_traj_last.append(torch.zeros(2).cuda() / 0)
                    else:
                        last_index = torch.max(good_index)
                        pred_traj_last.append(pred_traj_fake[last_index, obj_tmp, :])
                        gt_traj_last.append(pred_traj_gt[last_index, obj_tmp, :])
                gt_traj_last   = torch.stack(gt_traj_last, dim=0)       # num_obj x 2
                pred_traj_last = torch.stack(pred_traj_last, dim=0)     # num_obj x 2

                # compute FDE
                fde_tmp = final_displacement_error(pred_traj_last, gt_traj_last, mode='raw')
                fde.append(fde_tmp)         # list of error for all samples

            # select the one sample with the minimum errors, remove nan
            num_invalid = torch.sum(torch.isnan(ade_tmp))
            num_valid = pred_traj_gt.size(1) - num_invalid
            total_obj += num_valid         # only add No.obj if it is valid, not all future frames are padded
            ade_ave, num_obj = best_of_K(ade, seq_start_end, err_type='ADE')
            fde_ave, num_obj = best_of_K(fde, seq_start_end, err_type='FDE')
            ade_all.update(ade_ave, n=num_obj)
            fde_all.update(fde_ave, n=num_obj)

        actual_len = pred_len * args.skip
        final_dict = {actual_len: {obj_class: result_dict}}
        with open(result_file_single, 'w') as outfile:
            json.dump(final_dict, outfile)

        return ade_all.avg, fde_all.avg

def main(args):
    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [os.path.join(args.model_path, file_) for file_ in filenames]
    else:
        paths = [args.model_path]

    for path in paths:
        checkpoint = torch.load(path)
        generator = get_generator(checkpoint)
        _args = AttrDict(checkpoint['args'])
        datapath = get_dset_path(_args.dataset_name, args.dset_type)        # convert model path to dataset path
        _, loader = data_loader(_args, datapath, shuffle=False, phase='testing', split=args.dset_type)
        ade, fde = evaluate(_args, loader, generator, args.num_samples, path)
        print('Dataset: {}, Pred Len: {}, ADE: {:.3f}, FDE: {:.3f}'.format(
            _args.dataset_name, _args.pred_len*_args.skip, ade, fde))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)