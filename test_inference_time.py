import os
import sys
import time
import glob
import argparse
import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from osediff import OSEDiff_inference_time
from my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix

from ram.models.ram_lora import ram
from ram import inference_ram as inference

from tqdm import tqdm  # For progress bar


# Define transformations
tensor_transforms = transforms.Compose([
    transforms.ToTensor(),
])

ram_transforms = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def get_validation_prompt(args, lq, model, weight_dtype, device='cuda'):
    validation_prompt = ""
    lq_ram = ram_transforms(lq).to(dtype=weight_dtype, device=device)
    captions = inference(lq_ram, model)
    
    # Assuming captions should be used to form the validation_prompt
    if captions:
        validation_prompt = captions[0]  # Adjust based on how captions are returned
    return validation_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="Model Inference Speed Test")
    parser.add_argument('--pretrained_model_name_or_path', type=str, default=None, help='SD model path')
    parser.add_argument("--osediff_path", type=str, default='preset/models/osediff.pkl', help='Path to OSEDiff model')
    parser.add_argument('--ram_path', type=str, default=None, help='Path to RAM model')
    parser.add_argument('--ram_ft_path', type=str, default=None, help='Lora Path to RAM finetuned model')
    # Precision setting
    parser.add_argument("--mixed_precision", type=str, choices=['fp16', 'fp32'], default="fp16", help='Mixed precision mode')
    # Merge LoRA
    parser.add_argument("--merge_and_unload_lora", action='store_true', help='Merge LoRA weights before inference')
    # Tile settings
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224, help='VAE decoder tiled size')
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024, help='VAE encoder tiled size')
    parser.add_argument("--latent_tiled_size", type=int, default=96, help='Latent tiled size')
    parser.add_argument("--latent_tiled_overlap", type=int, default=32, help='Latent tiled overlap')
    # Additional arguments
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use for inference')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    parser.add_argument("--process_size", type=int, default=512, help='Size for processing')
    parser.add_argument('--inference_iterations', type=int, default=500, help='Number of inference iterations')
    parser.add_argument('--warmup_iterations', type=int, default=5, help='Number of warm-up iterations')
    
    return parser.parse_args()


def main():
    args = parse_args()
    args.merge_and_unload_lora = True
        
    # Initialize the model
    model = OSEDiff_inference_time(args)
    model.to(args.device)
    model.eval()
    
    # Initialize RAM model
    DAPE = ram(pretrained=args.ram_path,
               pretrained_condition=args.ram_ft_path,
               image_size=384,
               vit='swin_l')
    DAPE.eval()
    DAPE.to(args.device)
    
    # Weight type
    if args.mixed_precision == "fp16" and args.device == 'cuda':
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32
    
    # Set weight type
    DAPE = DAPE.to(dtype=weight_dtype)
    
    # Initialize timing variables
    total_time = 0.0
    batch_size = args.batch_size
    inference_iterations = args.inference_iterations
    warmup_iterations = args.warmup_iterations
    
    # Generate random tensors for inference
    # Pre-generate all tensors
    input_tensors = torch.randn((inference_iterations, batch_size, 3, args.process_size, args.process_size), device=args.device, dtype=weight_dtype)
    
    # Warm-up runs
    print(f"Running {warmup_iterations} warm-up iterations...")
    for _ in range(warmup_iterations):
        lq = input_tensors[_].clone()
        validation_prompt = get_validation_prompt(args, lq, DAPE, weight_dtype, device=args.device)
        with torch.no_grad():
            lq_processed = lq * 2 - 1  # normalization
            output_image = model(lq_processed, prompt=validation_prompt)
    
    torch.cuda.synchronize() if args.device == 'cuda' else None
    
    print(f"Starting inference for {inference_iterations} iterations...")
    # Inference runs with timing
    for idx in tqdm(range(inference_iterations), desc="Inference"):
        start_time = time.time()
        lq = input_tensors[idx].clone()
        validation_prompt = get_validation_prompt(args, lq, DAPE, weight_dtype, device=args.device)
        torch.cuda.synchronize()
        with torch.no_grad():
            lq_processed = lq * 2 - 1  # normalization
            output_image = model(lq_processed, prompt=validation_prompt)

        torch.cuda.synchronize()
        end_time = time.time()
        total_time += (end_time - start_time)
        
    avg_time = total_time / inference_iterations
    print(f'Average inference time per iteration: {avg_time:.4f} seconds.')
    

if __name__ == "__main__":
    main()