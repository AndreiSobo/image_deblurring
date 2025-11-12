import os
import argparse
from torch.utils.data import DataLoader
import logging
import mlflow
import mlflow.pytorch
import torch
import torch.optim as optim
from torchvision import transforms
import copy
from pathlib import Path

from src.data_ingestion import DeblurDataset
from src.model_class import DeblurUNet
from src.utils import train, evaluate, create_model_signature
from src.enhanced_loss import CombinedLoss


def save_best_checkpoint(model, optimizer, epoch, metrics, checkpoint_dir):
    """
    Save the best model checkpoint with all relevant information.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch number
        metrics: Dictionary with 'psnr', 'ssim', 'train_loss', 'val_loss'
        checkpoint_dir: Directory to save checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove old best checkpoints to keep only the latest best
    for old_ckpt in checkpoint_dir.glob('best_model_*.pth'):
        old_ckpt.unlink()
    
    # Create new checkpoint
    checkpoint_path = checkpoint_dir / f"best_model_epoch_{epoch}_psnr_{metrics['psnr']:.2f}.pth"
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': copy.deepcopy(model.state_dict()),
        'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
        'psnr': metrics['psnr'],
        'ssim': metrics['ssim'],
        'train_loss': metrics['train_loss'],
        'val_loss': metrics['val_loss']
    }
    
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"✓ Best checkpoint saved: {checkpoint_path.name}")
    
    return checkpoint_path


def load_and_verify_best_model(model, checkpoint_path, val_loader, criterion, device):
    """
    Load the best model checkpoint and verify its performance.
    
    Returns:
        tuple: (verified_psnr, verified_ssim, expected_psnr, expected_ssim, is_valid)
    """
    logging.info(f"Loading best model from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    expected_psnr = checkpoint['psnr']
    expected_ssim = checkpoint['ssim']
    expected_epoch = checkpoint['epoch']
    
    logging.info(f"Expected performance (Epoch {expected_epoch}): PSNR={expected_psnr:.2f}, SSIM={expected_ssim:.4f}")
    
    # Verify by running evaluation
    logging.info("Verifying model performance on validation set...")
    model.eval()
    val_loss, verified_psnr, verified_ssim = evaluate(model, val_loader, criterion, device)
    
    logging.info(f"Verified performance: PSNR={verified_psnr:.2f}, SSIM={verified_ssim:.4f}")
    
    # Check if verification matches expected (within tolerance)
    psnr_diff = abs(verified_psnr - expected_psnr)
    ssim_diff = abs(verified_ssim - expected_ssim)
    
    is_valid = psnr_diff < 0.5 and ssim_diff < 0.01
    
    if is_valid:
        logging.info("✓ Model verification PASSED")
    else:
        logging.warning(f"⚠ Model verification FAILED: PSNR diff={psnr_diff:.2f}, SSIM diff={ssim_diff:.4f}")
    
    return verified_psnr, verified_ssim, expected_psnr, expected_ssim, is_valid


def main():
    parser = argparse.ArgumentParser(description='Training a Deblurring model')
    parser.add_argument('--train_data', type=str, required=True, help='path to training data')
    parser.add_argument('--test_data', type=str, required=True, help='path to val data')
    parser.add_argument('--registered_model_name', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Start MLflow run
    mlflow.start_run()
    
    # Log hyperparameters
    mlflow.log_params({
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "patience": args.patience,
        "train_data": args.train_data,
        "test_data": args.test_data
    })
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    mlflow.log_param("device", str(device))

    # Data loading
    tsf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    logging.info(f"Loading training data from {args.train_data}")
    train_dataset = DeblurDataset(args.train_data, transform=tsf, is_training=True)
    
    logging.info(f"Loading validation data from {args.test_data}")
    val_dataset = DeblurDataset(args.test_data, transform=tsf, is_training=False)
    
    # DataLoader configuration
    NUM_WORKERS = min(8, os.cpu_count() or 4)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,  
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True,
        prefetch_factor=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,  
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True,
        prefetch_factor=4
    )
    
    logging.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    mlflow.log_params({
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset)
    })
    
    # Model setup
    logging.info("Initializing model...")
    model = DeblurUNet().to(device)
    criterion = CombinedLoss()
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-4
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=1e-6
    )
    
    logging.info(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Training tracking variables
    best_psnr = 0.0
    best_ssim = 0.0
    best_epoch = 0
    best_checkpoint_path = None
    patience_counter = 0
    
    logging.info("=" * 80)
    logging.info("Starting training...")
    logging.info("=" * 80)
    
    # Training loop
    for epoch in range(args.num_epochs):
        # Train
        train_loss = train(model, train_loader, criterion, optimizer, device)
        
        # Evaluate
        val_loss, val_psnr, val_ssim = evaluate(model, val_loader, criterion, device)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log to console
        logging.info(
            f"Epoch {epoch+1:03d}/{args.num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"PSNR: {val_psnr:.2f} dB | "
            f"SSIM: {val_ssim:.4f} | "
            f"LR: {current_lr:.2e}"
        )
        
        # Log to MLflow (all epochs for visualization)
        # Use "epoch_" prefix to distinguish from final best metrics
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch_psnr": float(val_psnr),
            "epoch_ssim": float(val_ssim),
            "learning_rate": current_lr
        }, step=epoch)
        
        # Check for improvement
        if val_psnr > best_psnr:
            improvement = val_psnr - best_psnr
            best_psnr = val_psnr
            best_ssim = val_ssim
            best_epoch = epoch + 1
            
            # Save best checkpoint
            metrics = {
                'psnr': val_psnr,
                'ssim': val_ssim,
                'train_loss': train_loss,
                'val_loss': val_loss
            }
            best_checkpoint_path = save_best_checkpoint(
                model, optimizer, epoch + 1, metrics, args.checkpoint_dir
            )
            
            patience_counter = 0
            logging.info(f"★ NEW BEST MODEL! PSNR improved by {improvement:.2f} dB")
            print(f"★ NEW BEST MODEL! PSNR improved by {improvement:.2f} dB, with PSNR: {val_psnr:.2f} and SSIM: {val_ssim:.4f}")
        else:
            patience_counter += 1
            logging.info(f"No improvement ({patience_counter}/{args.patience})")
        
        # Step the scheduler
        scheduler.step()
        
        # Early stopping
        if patience_counter >= args.patience:
            logging.info("=" * 80)
            logging.info(f"Early stopping triggered at epoch {epoch+1}")
            logging.info("=" * 80)
            break
    
    # Training completed
    logging.info("=" * 80)
    logging.info("Training completed!")
    logging.info(f"Best model: Epoch {best_epoch}, PSNR={best_psnr:.2f} dB, SSIM={best_ssim:.4f}")
    logging.info("=" * 80)
    
    # Load and verify the best model
    if best_checkpoint_path is None:
        logging.error("ERROR: No best checkpoint was saved during training!")
        mlflow.log_param("model_verification", "failed_no_checkpoint")
        mlflow.end_run()
        return
    
    verified_psnr, verified_ssim, expected_psnr, expected_ssim, is_valid = load_and_verify_best_model(
        model, best_checkpoint_path, val_loader, criterion, device
    )
    
    # Log verification results
    mlflow.log_params({
        "best_epoch": best_epoch,
        "model_verification": "passed" if is_valid else "failed"
    })
    
    mlflow.log_metrics({
        "best_psnr": float(verified_psnr),
        "best_ssim": float(verified_ssim),
        "expected_psnr": float(expected_psnr),
        "expected_ssim": float(expected_ssim),
        "psnr_verification_diff": float(abs(verified_psnr - expected_psnr)),
        "ssim_verification_diff": float(abs(verified_ssim - expected_ssim))
    })
    
    # Log the verified best model to MLflow
    logging.info("Logging model to MLflow...")
    signature = create_model_signature(model, device)
    
    mlflow.pytorch.log_model(
        model, 
        artifact_path="deblur_model",
        registered_model_name=args.registered_model_name, 
        signature=signature
    )
    
    # Also log the checkpoint file as artifact
    mlflow.log_artifact(str(best_checkpoint_path), artifact_path="checkpoints")
    
    logging.info(f"✓ Model logged to MLflow as: {args.registered_model_name}")
    logging.info(f"✓ Final metrics - PSNR: {verified_psnr:.2f} dB, SSIM: {verified_ssim:.4f}")
    
    if not is_valid:
        logging.warning("⚠ WARNING: Model verification failed! Check the logs above.")
    
    mlflow.end_run()
    logging.info("Training script completed successfully!")


if __name__ == "__main__":
    main()