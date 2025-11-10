import torch
import torch.nn as nn
from pytorch_msssim import MS_SSIM
import logging

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.84, beta=0.16, epsilon=1e-3):
        """
        Combines MS-SSIM (structural) with Charbonnier (pixel-wise).
        
        CRITICAL: MS-SSIM requires inputs in [0, data_range].
        Our images are normalized to [-1, 1], so we shift to [0, 2] for MS-SSIM.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.ms_ssim = MS_SSIM(data_range=2.0, size_average=True, channel=3)
        self.failure_count = 0

    def forward(self, prediction, target):
        # Charbonnier loss - works fine with [-1, 1] range
        diff = prediction - target
        charbonnier_loss = torch.mean(torch.sqrt(diff**2 + self.epsilon**2))
        
        # MS-SSIM requires [0, data_range] range
        # Shift from [-1, 1] to [0, 2]
        prediction_shifted = prediction + 1.0  # [-1, 1] → [0, 2]
        target_shifted = target + 1.0          # [-1, 1] → [0, 2]
        
        try:
            # Validate inputs are in correct range
            if prediction_shifted.min() < -0.01 or prediction_shifted.max() > 2.01:
                raise ValueError(
                    f"Prediction out of range: [{prediction_shifted.min():.3f}, {prediction_shifted.max():.3f}]. "
                    f"Expected [0, 2]"
                )
            
            # Compute MS-SSIM with shifted inputs
            ms_ssim_val = self.ms_ssim(prediction_shifted, target_shifted)
            
            # Validate MS-SSIM output
            if torch.isnan(ms_ssim_val) or torch.isinf(ms_ssim_val):
                raise ValueError(f"Invalid MS-SSIM value: {ms_ssim_val}")
            
            # MS-SSIM should be in [0, 1], but clamp for safety
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
