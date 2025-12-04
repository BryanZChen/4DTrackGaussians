from torch.utils.data import Dataset
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov
import torch
from utils.camera_utils import loadCam
from utils.graphics_utils import focal2fov
import os
from pathlib import Path


class FourDGSdataset(Dataset):
    def __init__(
        self,
        dataset,
        args,
        dataset_type,
        flow_dir=None
    ):
        self.dataset = dataset
        self.args = args
        self.dataset_type = dataset_type
        self.flow_dir = flow_dir
        
    def _get_flow_path(self, index):
        """
        Get optical flow path for a given frame index.
        
        Args:
            index: Frame index
        
        Returns:
            Path to flow .npy file or None if not available
        """
        if self.flow_dir is None:
            return None
        
        flow_path = os.path.join(self.flow_dir, f"flow_{index:06d}.npy")
        if os.path.exists(flow_path):
            return flow_path
        else:
            return None
    
    def __getitem__(self, index):
        # breakpoint()

        if self.dataset_type != "PanopticSports":
            try:
                image, w2c, time = self.dataset[index]
                R,T = w2c
                FovX = focal2fov(self.dataset.focal[0], image.shape[2])
                FovY = focal2fov(self.dataset.focal[0], image.shape[1])
                mask=None
            except:
                caminfo = self.dataset[index]
                image = caminfo.image
                R = caminfo.R
                T = caminfo.T
                FovX = caminfo.FovX
                FovY = caminfo.FovY
                time = caminfo.time
    
                mask = caminfo.mask
            
            flow_path = self._get_flow_path(index)
            return Camera(colmap_id=index,R=R,T=T,FoVx=FovX,FoVy=FovY,image=image,gt_alpha_mask=None,
                              image_name=f"{index}",uid=index,data_device=torch.device("cuda"),time=time,
                              mask=mask, flow_path=flow_path)
        else:
            return self.dataset[index]
    def __len__(self):
        
        return len(self.dataset)
