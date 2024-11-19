import os
import cv2
import wandb
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from custom_dataset import CustomDataset
from GAN_network import GeneratorNetwork, DiscriminatorNetwork, GANLoss
from torch.optim import lr_scheduler
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

def save_images(inputs, targets, outputs, folder_name, epoch, i, num_images=1):
    """
    Save a set of input, target, and output images for visualization.

    Args:
        inputs (torch.Tensor): Batch of input images.
        targets (torch.Tensor): Batch of target images.
        outputs (torch.Tensor): Batch of output images from the model.
        folder_name (str): Directory to save the images ('train_results' or 'val_results').
        epoch (int): Current epoch number.
        num_images (int): Number of images to save from the batch.
    """
    os.makedirs(f'{folder_name}/epoch_{epoch}', exist_ok=True)
    # Convert tensors to images
    input_img_np = tensor_to_image(inputs[0])
    target_img_np = tensor_to_image(targets[0])
    output_img_np = tensor_to_image(outputs[0])

    # Concatenate the images horizontally
    comparison = np.hstack((input_img_np, target_img_np, output_img_np))

    # Save the comparison image
    cv2.imwrite(f'{folder_name}/epoch_{epoch}/result_{i + 1}.png', comparison)

def train_one_epoch(net_G, net_D, dataloader, optimizer_G, optimizer_D, criterion, device, epoch, num_epochs, use_wandb):
    """
    Train the model for one epoch.

    Args:
        net_G (nn.Module): The generator neural network model.
        net_D (nn.Module): The discriminator neural network model.
        dataloader (DataLoader): DataLoader for the training data.
        optimizer (Optimizer): Optimizer for updating model parameters.
        criterion (Loss): Loss function.
        device (torch.device): Device to run the training on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
        use_wandb (bool): If true, use wandb to log.
    """
    net_G.train()
    net_D.train()
    running_loss_G = 0.0
    running_loss_D_real = 0.0
    running_loss_D_fake = 0.0
    dataset = dataloader.dataset.name

    for i, (image_rgb, image_semantic) in enumerate(dataloader):
        # Move data to the device
        real_rgb = image_rgb.to(device)
        image_semantic = image_semantic.to(device)

        # Forward pass
        fake_rgb = net_G(image_semantic)

        # Save sample images every 5 epochs
        if epoch % 5 == 0 and i < 5:
            save_images(image_semantic, real_rgb, fake_rgb, f'train_results/{dataset}', epoch, i)

        for param in net_D.parameters():
            param.requires_grad = True
        optimizer_D.zero_grad()

        # backward D
        fake_data = torch.cat((image_semantic, fake_rgb), 1)
        pred_fake = net_D(fake_data.detach())
        loss_D_fake = criterion(pred_fake, False)
        
        real_data = torch.cat((image_semantic, real_rgb), 1)
        pred_real = net_D(real_data)
        loss_D_real = criterion(pred_real, True)
        
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        loss_D.backward()
        optimizer_D.step()

        for param in net_D.parameters():
            param.requires_grad = False
        optimizer_G.zero_grad()

        # backward G
        fake_data = torch.cat((image_semantic, fake_rgb), 1)
        pred_fake = net_D(fake_data)

        loss_G = criterion(pred_fake, True)
        loss_G.backward()
        optimizer_G.step()

        # Update running loss
        running_loss_G += loss_G.item()
        running_loss_D_real += loss_D_real.item()
        running_loss_D_fake += loss_D_fake.item()

    avg_running_loss_G = running_loss_G / len(dataloader)
    avg_running_loss_D_real = running_loss_D_real / len(dataloader)
    avg_running_loss_D_fake = running_loss_D_fake / len(dataloader)
    if use_wandb:
        wandb.log({"avg_running_loss_G": avg_running_loss_G,
                "avg_running_loss_D_real": avg_running_loss_D_real,
                "avg_running_loss_D_fake": avg_running_loss_D_fake})
    # Print loss information
    print(f'Epoch [{epoch + 1}/{num_epochs}], Generator Loss: {avg_running_loss_G:.4f}')
    print(f'Epoch [{epoch + 1}/{num_epochs}], Discriminator Real Loss: {avg_running_loss_D_real:.4f}')
    print(f'Epoch [{epoch + 1}/{num_epochs}], Discriminator Fake Loss: {avg_running_loss_D_fake:.4f}')

def validate(net_G, net_D, dataloader, criterion, device, epoch, num_epochs, use_wandb):
    """
    Validate the model on the validation dataset.

    Args:
        net_G (nn.Module): The generator neural network model.
        net_D (nn.Module): The discriminator neural network model.
        dataloader (DataLoader): DataLoader for the validation data.
        criterion (Loss): Loss function.
        device (torch.device): Device to run the validation on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
        use_wandb (bool): If true, use wandb to log.
    """
    net_G.eval()
    net_D.eval()
    val_loss_G = 0.0
    val_loss_D_real = 0.0
    val_loss_D_fake = 0.0
    dataset = dataloader.dataset.name

    with torch.no_grad():
        for i, (image_rgb, image_semantic) in enumerate(dataloader):
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

            # Save sample images every 5 epochs
            if epoch % 5 == 0 and i < 5:
                save_images(image_semantic, image_rgb, fake_rgb, f'val_results/{dataset}', epoch, i)

    # Calculate average validation loss
    avg_val_loss_G = val_loss_G / len(dataloader)
    avg_val_loss_D_real = val_loss_D_real / len(dataloader)
    avg_val_loss_D_fake = val_loss_D_fake / len(dataloader)
    if use_wandb:
        wandb.log({"avg_val_loss_G": avg_val_loss_G,
                "avg_val_loss_D_real": avg_val_loss_D_real,
                "avg_val_loss_D_fake": avg_val_loss_D_fake})
    print(f'Epoch [{epoch + 1}/{num_epochs}], Generator Validation Loss: {avg_val_loss_G:.4f}')
    print(f'Epoch [{epoch + 1}/{num_epochs}], Discriminator Validation Real Loss: {avg_val_loss_D_real:.4f}')
    print(f'Epoch [{epoch + 1}/{num_epochs}], Discriminator Validation Fake Loss: {avg_val_loss_D_fake:.4f}')

def main(args):
    """
    Main function to set up the training and validation processes.
    """

    if args.use_wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project="Pix2Pix-GAN",

            # track hyperparameters and run metadata
            config={
            "architecture": "GAN",
            "dataset": args.dataset,
            "epochs": 201,
            }
        )

    # Set device to GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize datasets and dataloaders
    train_dataset = CustomDataset(name=args.dataset,
                                  list_file=f'train_list/{args.dataset}_train_list.txt',
                                  preprocess=args.preprocess)
    val_dataset = CustomDataset(name=args.dataset,
                                list_file=f'val_list/{args.dataset}_val_list.txt',
                                preprocess=args.preprocess)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Initialize model, loss function, and optimizer
    net_G = GeneratorNetwork().to(device)
    net_D = DiscriminatorNetwork().to(device)
    criterion = GANLoss('vanilla').to(device)
    optimizer_G = optim.Adam(net_G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(net_D.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Add a learning rate scheduler for decay
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + 1 - 100) / float(100 + 1)
        return lr_l
    scheduler_G = lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_rule)
    scheduler_D = lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lambda_rule)

    # Training loop
    num_epochs = 201
    for epoch in range(num_epochs):
        train_one_epoch(net_G, net_D, train_loader, optimizer_G, optimizer_D,
                        criterion, device, epoch, num_epochs, args.use_wandb)
        validate(net_G, net_D, val_loader, criterion, device,
                 epoch, num_epochs, args.use_wandb)

        # Step the scheduler after each epoch
        scheduler_G.step()
        scheduler_D.step()

        # Save model checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            os.makedirs(f'checkpoints/{args.dataset}', exist_ok=True)
            torch.save(net_G.state_dict(), f'checkpoints/{args.dataset}/pix2pix_model_epoch_{epoch + 1}_G.pth')
            torch.save(net_D.state_dict(), f'checkpoints/{args.dataset}/pix2pix_model_epoch_{epoch + 1}_D.pth')

if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--dataset', type=str, required=True)
    arg_parser.add_argument('--use_wandb', action='store_true')
    arg_parser.add_argument('--preprocess', action='store_true')
    args = arg_parser.parse_args()

    main(args)
