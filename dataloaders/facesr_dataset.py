import os
import cv2
import math
import numpy as np
import random
import torch
import torch.utils.data as data
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image
from dataloaders.codeformer import Codeformer_degradation
from dataloaders.utils.file import load_file_list,list_image_files
from dataloaders.utils.image import center_crop_arr, augment, random_crop_arr
from dataloaders.utils.degradation import (
    random_mixed_kernels, random_add_gaussian_noise, random_add_jpg_compression
)
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor

class CodeformerTxtDataset(data.Dataset):
    
    def __init__(
        self,
        args,
        split,
    ) -> "CodeformerTxtDataset":
        super(CodeformerTxtDataset, self).__init__()

        self.split = split
        self.args = args
        if self.split == 'train':
            self.degradation = Codeformer_degradation(self.args.deg_file_path, device='cpu')
            # degradation configurations
            self.blur_kernel_size = self.degradation.opt['blur_kernel_size']
            self.kernel_list = self.degradation.opt['kernel_list']
            self.kernel_prob = self.degradation.opt['kernel_prob']
            self.blur_sigma = self.degradation.opt['blur_sigma']
            self.downsample_range = self.degradation.opt['downsample_range']
            self.noise_range = self.degradation.opt['noise_range']
            self.jpeg_range = self.degradation.opt['jpeg_range']

            self.out_size = 512
            self.use_hflip = True
            self.gt_list = []
            for idx_dataset in range(len(args.dataset_txt_paths_list)):
                with open(args.dataset_txt_paths_list[idx_dataset], 'r') as f:
                    dataset_list = [line.strip() for line in f.readlines()]
                    for idx_ratio in range(args.dataset_prob_paths_list[idx_dataset]):
                        gt_length = len(self.gt_list)
                        self.gt_list += dataset_list
                        print(f'=====> append {len(self.gt_list) - gt_length} data.')
        elif self.split == 'test':
            self.T = transforms.Lambda(lambda x: x)
            with open(args.dataset_test_txt_paths, 'r') as f:
                self.lq_list = [line.strip() for line in f.readlines()]

    def __getitem__(self, index):
        # load gt image
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.

        if self.split == 'train':
            gt_path = self.gt_list[index]
            success = False
            imgname = gt_path.split('/')[-1]

            for _ in range(3):
                try:
                    pil_img = Image.open(gt_path).convert("RGB")
                    success = True
                    break
                except:
                    time.sleep(1)
            assert success, f"failed to load image {gt_path}"

            if min(*pil_img.size)!= self.out_size:
                pil_img = pil_img.resize((self.out_size,self.out_size),resample=Image.LANCZOS)

            pil_img = np.array(pil_img)
            assert pil_img.shape[:2] == (self.out_size, self.out_size)
                
            img_gt = (pil_img[..., ::-1] / 255.0).astype(np.float32)
            
            # random horizontal flip
            img_gt = augment(img_gt, hflip=self.use_hflip, rotation=True, return_status=False)
            h, w, _ = img_gt.shape

            # ------------------------ generate lq image ------------------------ #
            # blur
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                self.blur_kernel_size,
                self.blur_sigma,
                self.blur_sigma,
                [-math.pi, math.pi],
                noise_range=None
            )
            img_lq = cv2.filter2D(img_gt, -1, kernel)
            # downsample
            scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
            img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
            # noise
            if self.noise_range is not None:
                img_lq = random_add_gaussian_noise(img_lq, self.noise_range)
            # jpeg compression
            if self.jpeg_range is not None:
                img_lq = random_add_jpg_compression(img_lq, self.jpeg_range)
            
            # resize to original size
            img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # BGR to RGB
            target = (img_gt[..., ::-1]).astype(np.float32)
            # BGR to RGB
            source = (img_lq[..., ::-1]).astype(np.float32)

            # np -> tensor
            target = torch.from_numpy(target).permute(2, 0, 1)
            source = torch.from_numpy(source).permute(2, 0, 1)

            #  [0, 1] -> [-1, 1]
            target = target * 2 - 1
            source = source * 2 - 1
            
            example = {}
            example["neg_prompt"] = self.args.neg_prompt
            example["null_prompt"] = ""
            example["output_pixel_values"] = target
            example["conditioning_pixel_values"] = source

            return example

        elif self.split == 'test':
            input_img = Image.open(self.lq_list[index]).convert('RGB')

            # input images scaled to -1, 1
            img_t = self.T(input_img)
            img_t = F.to_tensor(img_t)
            img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
            
            example = {}
            example["neg_prompt"] = self.args.neg_prompt
            example["null_prompt"] = ""
            example["conditioning_pixel_values"] = img_t
            example["base_name"] = os.path.basename(self.lq_list[index])

            return example


    def __len__(self) -> int:
        if self.split == 'train':
            return len(self.gt_list)
        elif self.split == 'test':
            return len(self.lq_list)