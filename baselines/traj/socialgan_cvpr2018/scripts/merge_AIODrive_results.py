# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# merge results from different lens and classes into a single file for EvalAI submission

import os, json
from xinshuo_miscellaneous import get_timestring

root_dir = '../tmp'

# # test set
dir_dict = dict()
dir_dict['Car_20'] = 'train_20210515_12h41m52s_Car_20/results_20210516_12h37m29s'
dir_dict['Ped_20'] = 'train_20210515_12h42m22s_Ped_20/results_20210516_13h14m40s'
dir_dict['Cyc_20'] = 'train_20210514_15h30m28s_Cyc_20/results_20210516_13h16m37s'
dir_dict['Mot_20'] = 'train_20210515_13h43m45s_Mot_20/results_20210516_13h17m29s'

# val set
# dir_dict = dict()
# dir_dict['Car_10'] = 'train_20210513_22h17m00s_Car_10/results_20210516_14h55m13s'
# dir_dict['Car_20'] = 'train_20210515_12h41m52s_Car_20/results_20210516_14h55m25s'
# dir_dict['Car_50'] = 'train_20210515_12h42m09s_Car_50/results_20210516_14h55m43s'
# dir_dict['Ped_10'] = 'train_20210513_22h21m32s_Ped_10/results_20210516_15h00m19s'
# dir_dict['Ped_20'] = 'train_20210515_12h42m22s_Ped_20/results_20210516_15h00m36s'
# dir_dict['Ped_50'] = 'train_20210515_12h42m30s_Ped_50/results_20210516_15h00m56s'
# dir_dict['Cyc_10'] = 'train_20210513_21h15m39s_Cyc_10/results_20210516_14h56m23s'
# dir_dict['Cyc_20'] = 'train_20210514_15h30m28s_Cyc_20/results_20210516_14h57m01s'
# dir_dict['Cyc_50'] = 'train_20210514_15h39m29s_Cyc_50/results_20210516_14h57m19s'
# dir_dict['Mot_10'] = 'train_20210513_22h17m10s_Mot_10/results_20210516_14h51m26s'
# dir_dict['Mot_20'] = 'train_20210515_13h43m45s_Mot_20/results_20210516_14h51m33s'
# dir_dict['Mot_50'] = 'train_20210515_13h44m02s_Mot_50/results_20210516_14h52m35s'

# data = {10: {}, 20: {}, 50: {}}
# for pred_len in [10, 20, 50]:        # 1s, 2s, 5s prediction settings 
data = {20: {}}
for pred_len in [20]:        # 2s prediction settings 
	for obj_class in ['Car', 'Ped', 'Cyc', 'Mot']:
		key = '%s_%d' % (obj_class, pred_len)
		dir_tmp = dir_dict[key]
		path_tmp = os.path.join(root_dir, dir_tmp, 'results.json')
		print('loading results from %s' % path_tmp)
		with open(path_tmp, 'r') as file: 
			data_tmp = json.load(file)
		
		# copy each dict to the single final dictionary
		data[pred_len][obj_class] = data_tmp[str(pred_len)][obj_class]

print('saving')
save_file = os.path.join(root_dir, 'results_all_%s.json' % get_timestring())
with open(save_file, 'w') as outfile:
	json.dump(data, outfile)