import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from custom_dataset import CustomDataset
from GAN_network import GeneratorNetwork, DiscriminatorNetwork, GANLoss
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
    arg_parser.add_argument('--net_G', type=str, required=True)
    arg_parser.add_argument('--net_D', type=str, required=True)
    args = arg_parser.parse_args()

    # Set device to GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize datasets and dataloaders
    val_dataset = CustomDataset(name=args.dataset, list_file=f'val_list/{args.dataset}_val_list.txt')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Initialize model, loss function
    net_G = GeneratorNetwork().to(device)
    net_D = DiscriminatorNetwork().to(device)
    net_G.load_state_dict(torch.load(args.net_G, weights_only=True))
    net_D.load_state_dict(torch.load(args.net_D, weights_only=True))
    criterion = GANLoss('lsgan').to(device)

    net_G.eval()
    net_D.eval()
    val_loss_G = 0.0
    val_loss_D_real = 0.0
    val_loss_D_fake = 0.0
    val_loss = 0.0

    with torch.no_grad():
        for i, (image_rgb, image_semantic) in enumerate(val_loader):
            # Move data to the device
            real_rgb = image_rgb.to(device)
            image_semantic = image_semantic.to(device)

            # Forward pass
            fake_rgb = net_G(image_semantic)

            fake_data = torch.cat((image_semantic, fake_rgb), 1)
            pred_fake = net_D(fake_data)
            loss_D_fake = criterion(pred_fake, False)
            
            real_data = torch.cat((image_semantic, real_rgb), 1)
            pred_real = net_D(real_data)
            loss_D_real = criterion(pred_real, True)
            
            # loss_D = (loss_D_fake + loss_D_real) * 0.5
            loss_G = criterion(pred_fake, True)

            # Update running loss
            val_loss_G += loss_G.item()
            val_loss_D_real += loss_D_real.item()
            val_loss_D_fake += loss_D_fake.item()

            save_images(image_semantic, image_rgb, fake_rgb, f'eval_results/{args.dataset}', i)

    # Calculate average validation loss
    avg_val_loss_G = val_loss_G / len(val_loader)
    avg_val_loss_D_real = val_loss_D_real / len(val_loader)
    avg_val_loss_D_fake = val_loss_D_fake / len(val_loader)
    print(f'Generator Validation Loss: {avg_val_loss_G:.4f}')
    print(f'Discriminator Validation Real Loss: {avg_val_loss_D_real:.4f}')
    print(f'Discriminator Validation Fake Loss: {avg_val_loss_D_fake:.4f}')
