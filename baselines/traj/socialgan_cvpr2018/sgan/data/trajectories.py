import numpy as np, sys, math, os, logging, torch
from torch.utils.data import Dataset
from xinshuo_io import load_list_from_folder, fileparts
from xinshuo_miscellaneous import remove_list_from_list, find_unique_common_from_lists

logger = logging.getLogger(__name__)
np.set_printoptions(precision=3, suppress=True)

def seq_collate(data):
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,
     non_linear_ped_list, loss_mask_list, idframe_list) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)
    id_frame = torch.cat(idframe_list, dim=0).permute(2, 0, 1)

    out = [obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped, loss_mask, seq_start_end, id_frame]

    return tuple(out)

def read_file(_path, delim='tab'):
    data = []
    if delim == 'tab':     delim = '\t'
    elif delim == 'space': delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)

def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """

    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold: return 1.0
    else:                          return 0.0

# for AIODrive
def seqname2int(seqname):
    city_id, seq_id = seqname.split('_')
    city_id = int(city_id[4:6])
    seq_id = int(seq_id[3:])
    final_id = city_id * 1000 + seq_id

    return final_id

def check_eval_windows(start_pred, obs_len, pred_len, split='test'):
    # start_pred:       the frame index starts to predict in this window, e.g., seq of 0-9 -> 10-19 has start frame at 10
    # pred_len:         the number of frames to predict 
    # split:            train, val, test

    if split == 'test':
        reserve_interval = 50
        pred_len = 50
        obs_len  = 50
    else:
        reserve_interval = 0
    check = (start_pred - obs_len) % (obs_len + pred_len + reserve_interval) == 0

    return check

class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(self, data_dir, obs_len=8, pred_len=12, skip=1, threshold=0.002, min_ped=0, delim='\t', \
        phase='training', split='test'):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()
        self.data_dir = data_dir
        self.obs_len, self.pred_len = obs_len, pred_len
        self.seq_len = self.obs_len + self.pred_len
        self.delim, self.skip = delim, skip
        all_files, _ = load_list_from_folder(self.data_dir)
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        seq_id_list = []
        for path in all_files:
            print_str = 'load %s\r' % path
            sys.stdout.write(print_str)
            sys.stdout.flush()
            _, seq_name, _ = fileparts(path)
            data = read_file(path, delim)
            
            # as testing files only contains past, so add more windows
            if split == 'test':
                min_frame, max_frame = 0, 999
                num_windows = int(max_frame - min_frame + 1 - skip*(self.seq_len - 1))      
                num_windows += (self.pred_len-1)*skip + 1
            else:
                frames = np.unique(data[:, 0]).tolist()
                min_frame, max_frame = frames[0], frames[-1]
                num_windows = int(max_frame - min_frame + 1 - skip*(self.seq_len - 1))      # include all frames for past and future

            # loop through every windows
            for window_index in range(num_windows):               
                start_frame = int(window_index + min_frame)
                end_frame = int(start_frame + self.seq_len*skip)        # right-open, not including this frame   

                # reduce window during testing, only evaluate every N windows
                if phase == 'testing':
                    check_pass = check_eval_windows(start_frame+self.obs_len*skip, self.obs_len*skip, self.pred_len*skip, split=split)
                    if not check_pass: continue

                # get data in current window
                curr_seq_data = []
                for frame in range(start_frame, end_frame, skip):
                    curr_seq_data.append(data[frame == data[:, 0], :])        
                curr_seq_data = np.concatenate(curr_seq_data, axis=0)

                # initialize data
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                curr_seq_rel   = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))     # objects x 2 x seq_len
                curr_seq       = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq)   , self.seq_len))     # objects x seq_len
                id_frame_list  = np.zeros((len(peds_in_curr_seq), 3, self.seq_len))     # objects x 2 x seq_len
                id_frame_list.fill(-1)
                num_peds_considered = 0
                _non_linear_ped = []

                # loop through every object in this window
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]      # frames x 4 (frames can less than seqlen)
                    pad_front    = int(curr_ped_seq[0, 0] ) - start_frame             
                    pad_end      = int(curr_ped_seq[-1, 0]) - start_frame + skip         
                    assert pad_end % skip == 0, 'error'
                    frame_existing = curr_ped_seq[:, 0].tolist()

                    # pad front and back data to make the trajectory complete
                    if pad_end - pad_front != self.seq_len * skip:
                        
                        # pad end
                        to_be_paded_end = int(self.seq_len - pad_end / skip)
                        pad_end_seq  = np.expand_dims(curr_ped_seq[-1, :], axis=0)
                        pad_end_seq  = np.repeat(pad_end_seq, to_be_paded_end, axis=0)
                        frame_offset = np.zeros((to_be_paded_end, 4), dtype='float32')
                        frame_offset[:, 0] = np.array(range(1, to_be_paded_end+1))
                        pad_end_seq += frame_offset * skip                          # shift first columns for frame
                        curr_ped_seq = np.concatenate((curr_ped_seq, pad_end_seq), axis=0)

                        # pad front
                        to_be_paded_front = int(pad_front / skip)
                        pad_front_seq = np.expand_dims(curr_ped_seq[0, :], axis=0)
                        pad_front_seq = np.repeat(pad_front_seq, to_be_paded_front, axis=0)
                        frame_offset = np.zeros((to_be_paded_front, 4), dtype='float32')
                        frame_offset[:, 0] = np.array(range(-to_be_paded_front, 0))
                        pad_front_seq += frame_offset * skip
                        curr_ped_seq = np.concatenate((pad_front_seq, curr_ped_seq), axis=0)

                        # set pad front and end to correct values
                        pad_front = 0
                        pad_end = self.seq_len * skip

                    # add edge case when the object reappears at a bad frame
                    # in other words, missing intermediate frame
                    if curr_ped_seq.shape[0] != (pad_end - pad_front) / skip:
                        frame_all = list(range(int(curr_ped_seq[0, 0]), int(curr_ped_seq[-1, 0])+skip, skip))     
                        frame_missing, _ = remove_list_from_list(frame_all, curr_ped_seq[:, 0].tolist())

                        # pad all missing frames with zeros
                        pad_seq = np.expand_dims(curr_ped_seq[-1, :], axis=0)
                        pad_seq = np.repeat(pad_seq, len(frame_missing), axis=0)
                        pad_seq.fill(0)
                        pad_seq[:, 0] = np.array(frame_missing)
                        pad_seq[:, 1] = ped_id          # fill ID
                        curr_ped_seq = np.concatenate((curr_ped_seq, pad_seq), axis=0)
                        curr_ped_seq = curr_ped_seq[np.argsort(curr_ped_seq[:, 0])]

                    assert pad_front == 0, 'error'
                    assert pad_end == self.seq_len * skip, 'error'
                    
                    # make sure the seq_len frames are continuous, no jumping frames
                    start_frame_now = int(curr_ped_seq[0, 0])
                    if curr_ped_seq[-1, 0] != start_frame_now + (self.seq_len-1)*skip: 
                        continue

                    # make sure that past data has at least one frame
                    past_frame_list = [*range(start_frame_now, start_frame_now + self.obs_len * skip, skip)]
                    common = find_unique_common_from_lists(past_frame_list, frame_existing, only_com=True)
                    if len(common) == 0: continue

                    # make sure that future GT data has at least one frame
                    if phase != 'testing':
                        gt_frame_list = [*range(start_frame_now + self.obs_len*skip, start_frame_now + self.seq_len*skip, skip)]
                        common = find_unique_common_from_lists(gt_frame_list, frame_existing, only_com=True)
                        if len(common) == 0: continue

                    # only keep the state
                    cache_tmp = np.transpose(curr_ped_seq[:, :2])
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])        # 2 x seq_len

                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered
                    curr_seq[_idx, :, :] = curr_ped_seq     
                    curr_seq_rel[_idx, :, :] = rel_curr_ped_seq

                    # record seqname, frame and ID information
                    id_frame_list[_idx, :2, :] = cache_tmp
                    id_frame_list[_idx, 2, :] = seqname2int(seq_name)
                    
                    # Linear vs Non-Linear Trajectory, only fit for the future part not past part
                    _non_linear_ped.append(poly_fit(curr_ped_seq, pred_len, threshold))     

                    # add mask onto padded dummay data
                    frame_exist_index = np.array([frame_tmp - start_frame_now for frame_tmp in frame_existing])
                    frame_exist_index = (frame_exist_index / skip).astype('uint8')
                    curr_loss_mask[_idx, frame_exist_index] = 1
                    num_peds_considered += 1
                  
                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])
                    seq_id_list.append(id_frame_list[:num_peds_considered])

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)             # objects x 2 x seq_len
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)
        seq_id_list = np.concatenate(seq_id_list, axis=0)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]
        self.seq_id_list = torch.from_numpy(seq_id_list).type(torch.float)

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.seq_id_list[start:end, :]
        ]

        return out