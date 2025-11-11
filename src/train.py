# imports
import os
import argparse
from torch.utils.data import  DataLoader
import logging
import mlflow
import mlflow.pytorch  # type: ignore
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
import numpy as np

from src.data_ingestion import DeblurDataset
from src.model_class import DeblurUNet
from src.utils import train, evaluate
from src.utils import create_model_signature
from src.enhanced_loss import CombinedLoss




def main():
    parser = argparse.ArgumentParser(description='Training a Deblurring model')
    parser.add_argument('--train_data', type=str, required=True, help='path to training data')
    parser.add_argument('--test_data', type=str, required=True, help='path to val data')
    parser.add_argument('--registered_model_name', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--patience', type=int, default=50)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    
    mlflow.start_run()

    os.makedirs('checkpoints', exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"using device: {device}")
    print(f"using device: {device}")

    # transformation and data loading
    tsf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    logging.info(f"loading training data from {args.train_data}")
    train_dataset = DeblurDataset(args.train_data, transform=tsf)
    logging.info(f"loading testing dataset from {args.test_data}")
    test_dataset = DeblurDataset(args.test_data, transform=tsf, is_training=False)    # DataLoader configuration optimized for large images
    NUM_WORKERS = min(8, os.cpu_count() or 4)
    # pin_memory=True to speed up GPU transfer
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,  
        pin_memory=True if torch.cuda.is_available() else False,  # Faster GPU transfer
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=4  # ADD THIS - prefetch 4 batches per worker
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,  
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    
    # Model
    logging.info("Initializing model...")
    model = DeblurUNet().to(device)
    logging.info(f"Model moved to {device}")
    criterion = CombinedLoss()
    
    # Use command-line LR if provided, otherwise use 1e-4 default
    # Suggestion: Try 5e-4 or 1e-3 for faster initial convergence
    initial_lr = args.learning_rate if hasattr(args, 'learning_rate') else 2e-4
    
    optimiser = optim.AdamW(
        model.parameters(),
        lr=initial_lr,  # Use configurable learning rate
        betas=(0.9, 0.999),
        # eps=1e-8,
        weight_decay=1e-4
    )
    logging.info("Optimizer initialized")
    
    # Learning rate scheduler: reduces LR when validation PSNR plateaus
    # This enables "second convergence" by allowing finer optimization
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimiser,
    T_max=args.num_epochs,  # Total epochs
    eta_min=1e-6            # Minimum LR
)
    logging.info("Learning rate scheduler initialized (ReduceLROnPlateau)")
    
    mlflow.log_param("batch_size", args.batch_size)
    mlflow.log_param("num_epochs", args.num_epochs)
    mlflow.log_param("learning_rate", args.learning_rate)
    mlflow.log_param("patience", args.patience)

    # implement early stopping
    best_model_state = None
    best_current_ssim = 0.0
    best_current_psnr = 0.0
    patience_counter = 0    # Updated GradScaler API (PyTorch 2.0+)
    # from torch.amp.grad_scaler import GradScaler
    # scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
    
    for epoch in range(args.num_epochs):
        train_loss = train(model, train_loader, criterion, optimiser, device)
        val_loss, val_psnr, val_ssim = evaluate(model, test_loader, criterion, device)

            
        logging.info(
            f"Epoch {epoch+1}/{args.num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"PSNR: {val_psnr:.2f} dB | "
            f"SSIM: {val_ssim:.4f}"
        )
        
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("psnr", float(val_psnr), step=epoch)
        mlflow.log_metric("ssim", float(val_ssim), step=epoch)
        
        # Step the learning rate scheduler based on validation PSNR
        scheduler.step()
        current_lr = optimiser.param_groups[0]['lr']
        mlflow.log_metric("learning_rate", current_lr, step=epoch)
        logging.info(f"Current learning rate: {current_lr:.2e}")

        # check model improvement against main metric
        if val_psnr > best_current_psnr:
            best_current_psnr = val_psnr
            best_current_ssim = val_ssim

            # Save the best model state
            best_model_state = model.state_dict().copy()

            # Create checkpoints directory if it doesn't exist
            os.makedirs('checkpoints', exist_ok=True)
            
            # Save checkpoint
            checkpoint_path = f'checkpoints/best_model_epoch_{epoch+1}_psnr_{val_psnr:.2f}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimiser_state_dict': optimiser.state_dict(),
                'psnr': val_psnr,
                'ssim': val_ssim,
                'train_loss': train_loss,
                'val_loss': val_loss.item() if torch.is_tensor(val_loss) else val_loss
            }, checkpoint_path)

            # Note: Checkpoint saved locally, not logged to MLflow to avoid artifact URI issues
            # Can be logged manually later if needed
            logging.info(f"Checkpoint saved: {checkpoint_path}")

            # update the patience counter
            patience_counter = 0
            logging.info(f"New best model! PSNR: {val_psnr:.2f} dB, SSIM: {val_ssim:.4f} at epoch: {epoch+1}")
            print(f"New best model! PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}, at epoch: {epoch+1}")
        else:
            patience_counter +=1
        
        # early stopping
        if patience_counter >= args.patience:
            logging.info(f"early stopping triggered at epoch {epoch+1}")
            break
    
    # Load the best model state before logging
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logging.info(f"Loaded best model with PSNR: {best_current_psnr:.2f} dB, SSIM: {best_current_ssim:.4f}")
        print(f"Loaded best model with PSNR: {best_current_psnr:.2f} dB, SSIM: {best_current_ssim:.4f}")
    
    # Log the best model to MLflow
    signature = create_model_signature(model, device)
    mlflow.pytorch.log_model(  # type: ignore
        model, 
        artifact_path="deblur_model",  # Subdirectory name in MLflow artifacts
        registered_model_name=args.registered_model_name, 
        signature=signature
    )
    logging.info(f"saved model as: {args.registered_model_name}")
    mlflow.end_run()

if __name__ == "__main__":
    main()
