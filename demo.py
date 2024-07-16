import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder



DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


# def viz(img, flo, order):
#     img = img[0].permute(1,2,0).cpu().numpy()
#     flo = flo[0].permute(1,2,0).cpu().numpy()
    
#     # map flow to rgb image
#     flo = flow_viz.flow_to_image(flo)
#     # img_flo = np.concatenate([img, flo], axis=0)
#     # import matplotlib.pyplot as plt
#     # plt.imshow(flo / 255.0)
#     return flo


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        for i, (imfile1, imfile2) in enumerate(zip(images[:-1], images[1:])):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            print(image1.shape, image2.shape)
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            # viz(image1, flow_up, i)
            flo = flow_up[0].permute(1,2,0).cpu().numpy()
            flo = flow_viz.flow_to_image(flo)
            cv2.imwrite(os.path.join(args.output_path, f'flow_{i}.png'), flo)
            del image1, image2, padder, flow_low, flow_up
            torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--output_path', help="output path for saving the images with optical flow", default='output')    
    args = parser.parse_args()

    demo(args)
