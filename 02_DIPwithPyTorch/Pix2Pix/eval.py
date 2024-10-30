import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from custom_dataset import CustomDataset
from FCN_network import FullyConvNetwork
from argparse import ArgumentParser

def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor to a NumPy array suitable for OpenCV.

    Args:
        tensor (torch.Tensor): A tensor of shape (C, H, W).

    Returns:
        numpy.ndarray: An image array of shape (H, W, C) with values in [0, 255] and dtype uint8.
    """
    # Move tensor to CPU, detach from graph, and convert to NumPy array
    image = tensor.cpu().detach().numpy()
    # Transpose from (C, H, W) to (H, W, C)
    image = np.transpose(image, (1, 2, 0))
    # Denormalize from [-1, 1] to [0, 1]
    image = (image + 1) / 2
    # Scale to [0, 255] and convert to uint8
    image = (image * 255).astype(np.uint8)
    return image

def save_images(inputs, targets, outputs, folder_name, image_id):
    """
    Save a set of input, target, and output images for visualization.

    Args:
        inputs (torch.Tensor): Batch of input images.
        targets (torch.Tensor): Batch of target images.
        outputs (torch.Tensor): Batch of output images from the model.
        folder_name (str): Directory to save the images ('train_results' or 'val_results').
        image_id (int): The index of the image in dataset.
    """
    os.makedirs(f'{folder_name}', exist_ok=True)

    input_img_np = tensor_to_image(inputs[0])
    target_img_np = tensor_to_image(targets[0])
    output_img_np = tensor_to_image(outputs[0])

    # Concatenate the images horizontally
    comparison = np.hstack((input_img_np, target_img_np, output_img_np))

    # Save the comparison image
    cv2.imwrite(f'{folder_name}/result_{image_id}.png', comparison)

if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--dataset', type=str, required=True)
    arg_parser.add_argument('--model', type=str, required=True)
    args = arg_parser.parse_args()

    # Set device to GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize datasets and dataloaders
    val_dataset = CustomDataset(name=args.dataset, list_file=f'val_list/{args.dataset}_val_list.txt')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Initialize model, loss function
    model = FullyConvNetwork().to(device)
    model.load_state_dict(torch.load(args.model, weights_only=True))
    criterion = nn.L1Loss()

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for i, (image_rgb, image_semantic) in enumerate(val_loader):
            # Move data to the device
            image_rgb = image_rgb.to(device)
            image_semantic = image_semantic.to(device)

            # Forward pass
            outputs = model(image_semantic)

            # Compute the loss
            loss = criterion(outputs, image_rgb)
            val_loss += loss.item()

            save_images(image_semantic, image_rgb, outputs, f'eval_results/{args.dataset}', i)

    # Calculate average validation loss
    avg_val_loss = val_loss / len(val_loader)
    print(f'Validation Loss: {avg_val_loss:.4f}')
