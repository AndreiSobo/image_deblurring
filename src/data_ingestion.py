from torch.utils.data import Dataset, DataLoader
import logging
import os
from PIL import Image

class DeblurDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
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
        # blur_img = Image.open(blur_path).convert('L')  # for monochrome
        blur_img = Image.open(blur_path).convert('RGB')
        sharp_img = Image.open(sharp_path).convert('RGB')
        
        if self.transform:
            blur_img = self.transform(blur_img)
            sharp_img = self.transform(sharp_img)
        
        # print(f"Blur image shape in the _getitem func: {blur_img.shape}")  
        return blur_img, sharp_img