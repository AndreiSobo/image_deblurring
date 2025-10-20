from torch.utils.data import Dataset, DataLoader
import logging
import os
import re
from PIL import Image

class DeblurDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # Log the initial directory
        logging.info(f"Initializing DeblurDataset with data directory: {self.data_dir}")
        print(f"Initializing DeblurDataset with data directory: {self.data_dir}")
        
        self.image_pairs = self._get_image_pairs()
        
        if len(self.image_pairs) == 0:
            logging.error(f"No image pairs found in {data_dir}")
            print(f"No image pairs found in {data_dir}")
            raise ValueError(f"No image pairs found in {data_dir}")

    def _get_image_pairs(self):
        image_pairs = []
        logging.info(f"Searching for image pairs in: {self.data_dir}")
        print(f"Searching for image pairs in: {self.data_dir}")

        blurred_dir = os.path.join(self.data_dir, 'blurred')
        sharp_dir = os.path.join(self.data_dir, 'sharp')
        
        # Log and print directory checks
        logging.info(f"Checking if 'blurred' directory exists: {blurred_dir}")
        print(f"Checking if 'blurred' directory exists: {blurred_dir}")
        logging.info(f"Checking if 'sharp' directory exists: {sharp_dir}")
        print(f"Checking if 'sharp' directory exists: {sharp_dir}")
        
        if not os.path.exists(blurred_dir) or not os.path.exists(sharp_dir):
            logging.error(f"'blurred' or 'sharp' directory not found in {self.data_dir}")
            print(f"'blurred' or 'sharp' directory not found in {self.data_dir}")
            return image_pairs

        blurred_files = os.listdir(blurred_dir)
        logging.info(f"Found {len(blurred_files)} files in blurred directory: {blurred_dir}")
        print(f"Found {len(blurred_files)} files in blurred directory: {blurred_dir}")
        
        # Collect all sharp filenames for quick lookup
        sharp_filenames = [filename for filename in os.listdir(sharp_dir) if filename.endswith('.png')]
        sharp_filename_set = set(sharp_filenames)  # To speed up lookup

        for filename in blurred_files:
            if filename.endswith('.png'):
                blur_path = os.path.join(blurred_dir, filename)

                # Attempt to match filenames with regex (for augmentation suffixes)
                sharp_filename_with_suffix = re.sub(r'(_flip|_rotate)', r'_gt\1', filename)
                sharp_path_with_suffix = os.path.join(sharp_dir, sharp_filename_with_suffix)
                
                # Check if sharp image exists with regex-based name
                if os.path.exists(sharp_path_with_suffix):
                    image_pairs.append((blur_path, sharp_path_with_suffix))
                    logging.info(f"Valid image pair with suffix: {blur_path} and {sharp_path_with_suffix}")
                    print(f"Valid image pair with suffix: {blur_path} and {sharp_path_with_suffix}")
                else:
                    # Check if sharp image exists with direct '_gt' match
                    base_filename = re.sub(r'_flip|_rotate', '', filename)
                    sharp_filename_direct = f"{base_filename[:-4]}_gt.png"  # Remove .png and add _gt.png
                    sharp_path_direct = os.path.join(sharp_dir, sharp_filename_direct)
                    
                    if os.path.exists(sharp_path_direct):
                        image_pairs.append((blur_path, sharp_path_direct))
                        logging.info(f"Valid image pair with direct match: {blur_path} and {sharp_path_direct}")
                        print(f"Valid image pair with direct match: {blur_path} and {sharp_path_direct}")
                    else:
                        logging.warning(f"No matching sharp image for {filename}")
                        print(f"No matching sharp image for {filename}")
        
        logging.info(f"Found {len(image_pairs)} valid image pairs")
        print(f"Found {len(image_pairs)} valid image pairs")
        return image_pairs

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        blur_path, sharp_path = self.image_pairs[idx]
        # blur_img = Image.open(blur_path).convert('RGB')
        blur_img = Image.open(blur_path).convert('L')
        sharp_img = Image.open(sharp_path).convert('L')
        
        if self.transform:
            blur_img = self.transform(blur_img)
            sharp_img = self.transform(sharp_img)
        
        # print(f"Blur image shape in the _getitem func: {blur_img.shape}")  
        return blur_img, sharp_img