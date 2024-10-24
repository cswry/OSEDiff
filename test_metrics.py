# Image Quality Assessment Script
# Evaluates metrics like PSNR, SSIM, LPIPS, FID, DISTS, etc., for a set of images.

import os
import sys
import glob
import argparse
import logging
from datetime import datetime
import time

import cv2
import numpy as np
import torch

import pyiqa
from basicsr.utils import img2tensor

def get_timestamp():
    """Returns the current timestamp in a specific format."""
    return datetime.now().strftime('%y%m%d-%H%M%S')

def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    """
    Sets up a logger with specified configurations.

    Args:
        logger_name (str): Name of the logger.
        root (str): Root directory for log files.
        phase (str): Phase name (e.g., 'test').
        level (int, optional): Logging level. Defaults to logging.INFO.
        screen (bool, optional): Whether to log to the screen. Defaults to False.
        tofile (bool, optional): Whether to log to a file. Defaults to False.
    """
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        fmt='%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
        datefmt='%y-%m-%d %H:%M:%S'
    )
    logger.setLevel(level)

    if tofile:
        log_file = os.path.join(root, f"{phase}_{get_timestamp()}.log")
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

def dict2str(opt, indent=1):
    """
    Converts a dictionary to a formatted string for logging.

    Args:
        opt (dict): The dictionary to convert.
        indent (int, optional): Indentation level. Defaults to 1.

    Returns:
        str: Formatted string representation of the dictionary.
    """
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent * 2) + f"{k}:[\n"
            msg += dict2str(v, indent + 1)
            msg += ' ' * (indent * 2) + "]\n"
        else:
            msg += ' ' * (indent * 2) + f"{k}: {v}\n"
    return msg

def main():
    parser = argparse.ArgumentParser(description="Image Quality Assessment Script")

    parser.add_argument(
        "--inp_imgs",
        nargs="+",
        required=True,
        help="Path(s) to the input (SR) images directories."
    )

    parser.add_argument(
        "--gt_imgs",
        nargs="+",
        required=True,
        help="Path(s) to the ground truth (GT) images directories."
    )

    parser.add_argument(
        "--log",
        type=str,
        required=True,
        help="Directory path to save the log files."
    )

    parser.add_argument(
        "--log_name",
        type=str,
        default='METRICS',
        help="Base name for the log files."
    )

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Create log directory if it doesn't exist
    os.makedirs(args.log, exist_ok=True)

    # Initialize logger
    # Assuming the first init image path has enough parts
    try:
        args.log_name = args.inp_imgs[0].split('/')[8]
    except IndexError:
        args.log_name = 'METRICS'
    setup_logger('base', args.log, f'test_{args.log_name}', level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info("===== Configuration =====")
    logger.info(dict2str(vars(args)))
    logger.info("==========================\n")

    # Initialize IQA metrics excluding FID
    logger.info("Initializing IQA metrics...")
    iqa_metrics = {
        'PSNR': pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr').to(device),
        'SSIM': pyiqa.create_metric('ssim', test_y_channel=True, color_space='ycbcr').to(device),
        'LPIPS': pyiqa.create_metric('lpips', device=device),
        'DISTS': pyiqa.create_metric('dists', device=device),
        'CLIPIQA': pyiqa.create_metric('clipiqa', device=device),
        'NIQE': pyiqa.create_metric('niqe', device=device),
        'MUSIQ': pyiqa.create_metric('musiq', device=device),
        'MANIQA': pyiqa.create_metric('maniqa-pipal', device=device)
    }

    # Initialize FID separately
    fid_metric = pyiqa.create_metric('fid', device=device)
    logger.info("IQA metrics initialized.\n")

    # Validate input and GT directories
    if len(args.inp_imgs) != len(args.gt_imgs):
        logger.error("The number of input image directories and GT image directories must be the same.")
        sys.exit(1)

    init_imgs_names = []
    for dir_idx, init_dir in enumerate(args.inp_imgs):
        gt_dir = args.gt_imgs[dir_idx]
        img_gt_list = sorted(glob.glob(os.path.join(gt_dir, '*.png')))
        img_sr_list = sorted(glob.glob(os.path.join(init_dir, '*.png')))

        dir_name = os.path.basename(os.path.normpath(init_dir))
        init_imgs_names.append(dir_name)

        logger.info(f"Directory [{dir_name}]: {len(img_gt_list)} GT images vs {len(img_sr_list)} SR images.")
        assert len(img_gt_list) == len(img_sr_list), f"Mismatch in number of images for directory: {dir_name}"

    logger.info("\n===== Starting Evaluation =====\n")

    # Iterate over each directory
    for dir_idx, init_dir in enumerate(args.inp_imgs):
        gt_dir = args.gt_imgs[dir_idx]
        img_gt_list = sorted(glob.glob(os.path.join(gt_dir, '*.png')))
        img_sr_list = sorted(glob.glob(os.path.join(init_dir, '*.png')))
        dir_name = init_imgs_names[dir_idx]

        # Initialize accumulators for average metrics
        metrics_accum = {metric: 0.0 for metric in iqa_metrics.keys()}

        logger.info(f"Testing Directory: [{dir_name}]")

        # Iterate over each image pair
        for img_idx, sr_path in enumerate(img_sr_list):
            gt_path = img_gt_list[img_idx]
            img_name = os.path.basename(sr_path)

            start_time = time.time()

            # Read and preprocess images
            sr_img = cv2.imread(sr_path, cv2.IMREAD_COLOR)
            gt_img = cv2.imread(gt_path, cv2.IMREAD_COLOR)

            if sr_img is None or gt_img is None:
                logger.warning(f"Image read failed for {img_name}. Skipping.")
                continue

            sr_tensor = img2tensor(sr_img, bgr2rgb=True, float32=True).unsqueeze(0).to(device).contiguous() / 255.0
            gt_tensor = img2tensor(gt_img, bgr2rgb=True, float32=True).unsqueeze(0).to(device).contiguous() / 255.0

            # Compute metrics
            with torch.no_grad():
                metrics = {}
                for name, metric in iqa_metrics.items():
                    if name in ['CLIPIQA', 'NIQE', 'MUSIQ', 'MANIQA']:
                        metrics[name] = metric(sr_tensor).item()
                    else:
                        metrics[name] = metric(sr_tensor, gt_tensor).item()

            # Accumulate metrics
            for name in metrics_accum:
                metrics_accum[name] += metrics[name]

            # Calculate runtime
            end_time = time.time()
            runtime = end_time - start_time

            # Log per-image metrics and runtime
            metrics_str = "; ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
            logger.info(f"{dir_name}/{img_name} | {metrics_str} | Runtime: {runtime:.2f} sec")

        # Compute average metrics
        num_images = len(img_sr_list)
        avg_metrics = {k: round(v / num_images, 4) for k, v in metrics_accum.items()}

        # Compute FID for the directory
        fid_start_time = time.time()
        fid_value = fid_metric(gt_dir, init_dir).item()
        fid_end_time = time.time()
        fid_runtime = fid_end_time - fid_start_time

        # Log average metrics for the directory
        avg_metrics_str = "; ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()])
        logger.info(f"\n===== Average Metrics for [{dir_name}] =====\n{avg_metrics_str} | FID: {fid_value:.6f} | FID Runtime: {fid_runtime:.2f} sec\n")

        # Optionally, you can accumulate FID if needed for overall statistics

    logger.info("===== Evaluation Completed =====")

if __name__ == "__main__":
    main()