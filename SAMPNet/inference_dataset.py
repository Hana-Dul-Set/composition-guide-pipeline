from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
from PIL import Image
import os, json
import torchvision.transforms as transforms
import random
import numpy as np
from config import Config
import cv2
import time

IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]

random.seed(1)
torch.manual_seed(1)
cv2.setNumThreads(0)

# Refer to: Saliency detection: A spectral residual approach
def detect_saliency(img, scale=6, q_value=0.95, target_size=(224,224)):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    W, H = img_gray.shape
    img_resize = cv2.resize(img_gray, (H // scale, W // scale), interpolation=cv2.INTER_AREA)

    myFFT = np.fft.fft2(img_resize)
    myPhase = np.angle(myFFT)
    myLogAmplitude = np.log(np.abs(myFFT) + 0.000001)
    myAvg = cv2.blur(myLogAmplitude, (3, 3))
    mySpectralResidual = myLogAmplitude - myAvg

    m = np.exp(mySpectralResidual) * (np.cos(myPhase) + complex(1j) * np.sin(myPhase))
    saliencyMap = np.abs(np.fft.ifft2(m)) ** 2
    saliencyMap = cv2.GaussianBlur(saliencyMap, (9, 9), 2.5)
    saliencyMap = cv2.resize(saliencyMap, target_size, interpolation=cv2.INTER_LINEAR)
    threshold = np.quantile(saliencyMap.reshape(-1), q_value)
    if threshold > 0:
        saliencyMap[saliencyMap > threshold] = threshold
        saliencyMap = (saliencyMap - saliencyMap.min()) / threshold
    # for debugging
    # import matplotlib.pyplot as plt
    # plt.subplot(1,2,1)
    # plt.imshow(img)
    # plt.axis('off')
    # plt.subplot(1,2,2)
    # plt.imshow(saliencyMap, cmap='gray')
    # plt.axis('off')
    # plt.show()
    return saliencyMap

class InferenceDataset(Dataset):
    def __init__(self, cfg):
        self.image_path = os.path.join(cfg.inf_image_path)

        self.image_list  = os.listdir(self.image_path)
        self.image_size = cfg.image_size
        self.transformer = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_name = self.image_list[index]
        image_file = os.path.join(self.image_path, image_name)
        assert os.path.exists(image_file), image_file + ' not found'
        src = Image.open(image_file).convert('RGB')
        im = self.transformer(src)

        src_im  = np.asarray(src).copy()
        sal_map = detect_saliency(src_im, target_size=(self.image_size, self.image_size))
        sal_map = torch.from_numpy(sal_map.astype(np.float32)).unsqueeze(0)
        return image_name, im, sal_map

    def normbboxes(self, bboxes, w, h):
        bboxes = bboxes.astype(np.float32)
        center_x = (bboxes[:,0] + bboxes[:,2]) / 2. / w
        center_y = (bboxes[:,1] + bboxes[:,3]) / 2. / h
        norm_w = (bboxes[:,2] - bboxes[:,0]) / w
        norm_h = (bboxes[:,3] - bboxes[:,1]) / h
        norm_bboxes = np.column_stack((center_x, center_y, norm_w, norm_h))
        norm_bboxes = np.clip(norm_bboxes, 0, 1)
        assert norm_bboxes.shape == bboxes.shape, '{} vs. {}'.format(bboxes.shape, norm_bboxes.shape)
        # print(w,h,bboxes[0],norm_bboxes[0])
        return norm_bboxes

    def scores2dist(self, scores):
        scores = np.array(scores)
        count = [(scores == i).sum() for i in range(1,6)]
        count = np.array(count)
        assert count.sum() == 5, scores
        distribution = count.astype(np.float) / count.sum()
        distribution = distribution.tolist()
        return distribution

    def collate_fn(self, batch):
        batch = list(zip(*batch))
        if self.split == 'train':
            assert (not self.need_image_path) and (not self.need_mask) \
                   and (not self.need_proposal), 'Multi-scale training not implement'
            self.image_size = random.choice(range(self.min_image_size,
                                                  self.max_image_size+1,
                                                  16))
            batch[0] = [resize(im, self.image_size) for im in batch[0]]
        batch = [torch.stack(data, dim=0) for data in batch]
        return  batch


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image