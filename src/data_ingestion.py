from torch.utils.data import Dataset, DataLoader
import logging
import os
import random
from PIL import Image
import torch
import torchvision.transforms.functional as TF

class DeblurDataset(Dataset):
    def __init__(self, data_dir, transform=None, patch_size=256, is_training=True):
        self.data_dir = data_dir
        self.transform = transform
        self.patch_size = patch_size
        self.is_training = is_training  # Enable augmentation only for training
        
        # Log the initial directory
        logging.info(f"Initializing GoPro dataset with data directory: {self.data_dir}")
        print(f"Initializing GoPro dataset with data directory: {self.data_dir}")
        
        self.image_pairs = self._get_image_pairs()
        
        if len(self.image_pairs) == 0:
            logging.error(f"No image pairs found in {data_dir}")
            print(f"No image pairs found in {data_dir}")
            raise ValueError(f"No image pairs found in {data_dir}")

    def _get_image_pairs(self):
        image_pairs = []
        logging.info(f"Searching for image pairs in: {self.data_dir}")
        print(f"Searching for image pairs in: {self.data_dir}")

        # GoPro dataset structure: data_dir contains sequence folders (e.g., GOPR0372_07_00)
        # Each sequence folder contains 'blur' and 'sharp' subdirectories
        
        if not os.path.exists(self.data_dir):
            logging.error(f"Data directory not found: {self.data_dir}")
            print(f"Data directory not found: {self.data_dir}")
            return image_pairs
        
        # Get all sequence folders in the data directory
        sequence_folders = [f for f in os.listdir(self.data_dir) 
                          if os.path.isdir(os.path.join(self.data_dir, f)) and f.startswith('GOPR')]
        
        logging.info(f"Found {len(sequence_folders)} sequence folders")
        print(f"Found {len(sequence_folders)} sequence folders")
        
        # Iterate through each sequence folder
        for seq_folder in sequence_folders:
            seq_path = os.path.join(self.data_dir, seq_folder)
            blur_dir = os.path.join(seq_path, 'blur')
            sharp_dir = os.path.join(seq_path, 'sharp')
            
            # Check if both blur and sharp directories exist
            if not os.path.exists(blur_dir) or not os.path.exists(sharp_dir):
                logging.warning(f"Missing 'blur' or 'sharp' directory in {seq_folder}")
                print(f"Missing 'blur' or 'sharp' directory in {seq_folder}")
                continue
            
            # Get all blur images
            blur_files = sorted([f for f in os.listdir(blur_dir) if f.endswith('.png')])
            logging.info(f"Found {len(blur_files)} blur images in {seq_folder}")
            print(f"Found {len(blur_files)} blur images in {seq_folder}")
            
            # Match blur and sharp images with the same filename
            for blur_filename in blur_files:
                blur_path = os.path.join(blur_dir, blur_filename)
                sharp_path = os.path.join(sharp_dir, blur_filename)
                
                # Check if corresponding sharp image exists
                if os.path.exists(sharp_path):
                    image_pairs.append((blur_path, sharp_path))
                else:
                    logging.warning(f"No matching sharp image for {blur_path}")
                    print(f"No matching sharp image for {blur_path}")
        logging.info(f"Found {len(image_pairs)} valid image pairs")
        print(f"Found {len(image_pairs)} valid image pairs")
        return image_pairs
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        blur_path, sharp_path = self.image_pairs[idx]
        
        # Load images on-demand (lazy loading)
        try:
            blur_img = Image.open(blur_path).convert('RGB')
            sharp_img = Image.open(sharp_path).convert('RGB')
        except Exception as e:
            logging.error(f"Error loading images at index {idx}: {e}")
            raise
        
        # Extract random patches directly in the dataset
        # This happens BEFORE DataLoader batching, reducing memory by ~94%
        blur_patch, sharp_patch = self._extract_random_patch(blur_img, sharp_img)
        
        # Apply data augmentation (only during training)
        if self.is_training:
            blur_patch, sharp_patch = self._apply_augmentation(blur_patch, sharp_patch)
        
        # Apply normalization transform if provided
        if self.transform:
            blur_patch = self.transform(blur_patch)
            sharp_patch = self.transform(sharp_patch)
        
        return blur_patch, sharp_patch
    
    def _extract_random_patch(self, blur_img, sharp_img):
        """Extract synchronized random patches from blur and sharp images."""
        bw, bh = blur_img.size
        sw, sh = sharp_img.size
        
        # Ensure both images have the same size (center crop if needed)
        if (bw, bh) != (sw, sh):
            tw, th = min(bw, sw), min(bh, sh)
            blur_img = TF.center_crop(blur_img, (th, tw))   #type:ignore
            sharp_img = TF.center_crop(sharp_img, (th, tw)) #type:ignore
        
        W, H = blur_img.size    #type:ignore
        
        # Handle images smaller than patch_size
        if W < self.patch_size or H < self.patch_size:
            pad_w = max(0, self.patch_size - W)
            pad_h = max(0, self.patch_size - H)
            padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
            blur_img = TF.pad(blur_img, padding, padding_mode='reflect')        #type:ignore
            sharp_img = TF.pad(sharp_img, padding, padding_mode='reflect')     #type:ignore
            W, H = blur_img.size                                                #type:ignore
        
        # Extract random synchronized patch
        x = random.randint(0, W - self.patch_size)
        y = random.randint(0, H - self.patch_size)
        
        blur_patch = TF.crop(blur_img, y, x, self.patch_size, self.patch_size)
        sharp_patch = TF.crop(sharp_img, y, x, self.patch_size, self.patch_size)
        
        return blur_patch, sharp_patch
    
    def _apply_augmentation(self, blur_patch, sharp_patch):
        """Apply synchronized data augmentation to patches."""
        # Horizontal flip
        if random.random() > 0.5:
            blur_patch = TF.hflip(blur_patch)
            sharp_patch = TF.hflip(sharp_patch)
        
        # Vertical flip
        if random.random() > 0.5:
            blur_patch = TF.vflip(blur_patch)
            sharp_patch = TF.vflip(sharp_patch)
        
        # Random 90° rotations (0, 90, 180, 270 degrees)
        k = random.choice([0, 1, 2, 3])
        if k > 0:
            # Convert to tensor for rotation, then back to PIL
            blur_tensor = TF.to_tensor(blur_patch)          #type:ignore
            sharp_tensor = TF.to_tensor(sharp_patch)        #type:ignore
            
            blur_tensor = torch.rot90(blur_tensor, k=k, dims=[1, 2])
            sharp_tensor = torch.rot90(sharp_tensor, k=k, dims=[1, 2])
            
            blur_patch = TF.to_pil_image(blur_tensor)
            sharp_patch = TF.to_pil_image(sharp_tensor)
        
        return blur_patch, sharp_patch