import os
from torch.utils.data import Dataset
from utils.read_and_write import read_json
from configs.path_config import split_path
class ScanNet(Dataset):
    def __init__(self, cfg, mode):
        self.config = cfg.config
        self.dataset_config = cfg.dataset_config
        self.mode = mode
        split_name =  self.config['data'].get('split_name',None)
        print("split name: %s"%split_name)
        if split_name:
            mode = split_name
        split_file = os.path.join(split_path,'fullscan',cfg.config['data']['split'], 'scannetv2_' + mode + '.json')
        self.split = read_json(split_file)

    def __len__(self):
        return len(self.split)