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

from src.data_ingestion import DeblurDataset
from src.model_class import DeblurUNet
from src.utils import train, evaluate
from src.utils import create_model_signature
from src.CharbonnierLoss import CharbonnierLoss




def main():
    parser = argparse.ArgumentParser(description='Training a Deblurring model')
    parser.add_argument('--train_data', type=str, required=True, help='path to training data')
    parser.add_argument('--test_data', type=str, required=True, help='path to val data')
    parser.add_argument('--registered_model_name', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=5)

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
    test_dataset = DeblurDataset(args.test_data, transform=tsf)

    # Reduced num_workers to prevent OOM - each worker loads full images before patching
    # Use 0 for debugging, increase to 2 if memory allows
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    # Model

    model = DeblurUNet().to(device)
    criterion = CharbonnierLoss()
    optimiser = optim.AdamW(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4
    )
    mlflow.log_param("batch_size", args.batch_size)
    mlflow.log_param("num_epochs", args.num_epochs)
    mlflow.log_param("learning_rate", args.learning_rate)
    mlflow.log_param("patience", args.patience)

    # implement early stopping
    best_model_state = None
    best_current_ssim = 0.0
    best_current_psnr = 0.0
    patience_counter = 0

    # Updated GradScaler API (PyTorch 2.0+)
    from torch.amp.grad_scaler import GradScaler
    scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
    for epoch in range(args.num_epochs):
        train_loss = train(model, train_loader, criterion, optimiser, device)
        val_loss, val_psnr, val_ssim = evaluate(model, test_loader, criterion, device)

        logging.info(f"epoch {epoch+1}/{args.num_epochs} train loss: {train_loss:.2f}, "
                 f"val loss: {val_loss:.2f}, psnr: {val_psnr}, ssim: {val_ssim}")
        
        # log metrics
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("psnr", float(val_psnr), step=epoch)
        mlflow.log_metric("ssim", float(val_ssim), step=epoch)

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
            logging.info(f"💾 Checkpoint saved: {checkpoint_path}")

            # update the patience counter
            patience_counter = 0
            logging.info(f"✅ New best model! PSNR: {val_psnr:.2f} dB, SSIM: {val_ssim:.4f} at epoch: {epoch+1}")
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





