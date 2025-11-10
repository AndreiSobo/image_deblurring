import torch
import torch.nn as nn
from pytorch_msssim import MS_SSIM
import logging

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.84, beta=0.16, epsilon=1e-3):
        """
        combines SSIM and the current Charbonnier loss functions
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.ms_ssim = MS_SSIM(data_range=2.0, size_average=True, channel=3)
        self.failure_count = 0

    def forward(self, prediction, target):
        # CRITICAL FIX: Clamp outputs to prevent NaN in MS-SSIM
        prediction_clamped = torch.clamp(prediction, -1.0, 1.0)
        target_clamped = torch.clamp(target, -1.0, 1.0)
        
        # Charbonnier loss (always calculated as fallback)
        diff = prediction - target
        charbonnier_loss = torch.mean(torch.sqrt(diff**2 + self.epsilon **2))
        
        # MS-SSIM loss with safety check
        try:
            # Check tensor validity before MS-SSIM
            if torch.isnan(prediction_clamped).any() or torch.isinf(prediction_clamped).any():
                raise ValueError("NaN/Inf in prediction before MS-SSIM")
            if torch.isnan(target_clamped).any() or torch.isinf(target_clamped).any():
                raise ValueError("NaN/Inf in target before MS-SSIM")
                
            ms_ssim_val = self.ms_ssim(prediction_clamped, target_clamped)
            
            # Validate MS-SSIM output
            if torch.isnan(ms_ssim_val) or torch.isinf(ms_ssim_val):
                raise ValueError(f"Invalid MS-SSIM value: {ms_ssim_val}")
                
            # Clamp SSIM to valid range [0, 1] to prevent NaN
            ms_ssim_val = torch.clamp(ms_ssim_val, 0.0, 1.0)
            ms_ssim_loss = 1 - ms_ssim_val
            
            # Reset failure counter on success
            if self.failure_count > 0:
                logging.info(f"MS-SSIM recovered after {self.failure_count} failures")
                self.failure_count = 0
                
        except Exception as e:
            # Fallback to Charbonnier-only if MS-SSIM fails
            self.failure_count += 1
            if self.failure_count <= 3:  # Only log first few failures
                logging.warning(f"MS-SSIM failed ({self.failure_count}): {e}. Using Charbonnier-only loss.")
            ms_ssim_loss = charbonnier_loss  # Use Charbonnier as substitute

        return self.alpha * ms_ssim_loss + self.beta * charbonnier_loss
