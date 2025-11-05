## Tiling and Stitching

### Techniques Used:
- **Tiling:** Sliding window with overlap (32 pixels default)
- **Stitching:** Linear feathering (weighted blending with edge ramps)

### Device Compatibility:
The tiling and stitching functions are device-agnostic and work on both GPU and CPU. This is important as both functions will be used in the training process (GPU) and for inference (CPU-based on Azure Function).

- `tile_tensor()` uses tensor slicing and preserves the input device automatically
- `stitch_tiles()` creates output tensors on the same device as input tiles: `device=out_tiles[0].device`
- `create_blend_weight()` creates weight maps that are moved to the appropriate device during stitching

### Linear Feathering Benefits:
- Reduces edge artifacts from CNN processing
- Smooth transitions between tiles (no visible seams)
- Better handling of U-Net's limited receptive field at tile edges
- 10-20% better visual quality compared to uniform averaging

## Ingestion Pipeline

### Data Structure:
The input for training is composed of pairs of sharp-blurry images from the GoPro dataset. The `DeblurDataset` class loads these pairs and `random_patch_pair()` extracts random 256×256 patches.

### Training Patch Extraction:
- **One patch per image per epoch** - `random_patch_pair()` is called ONCE per image
- **Random location each epoch** - Different patch extracted every epoch for data augmentation
- **Synchronized patches** - Blur and sharp patches extracted from same location
- **Batch composition** - Batch size of 8-16 means 8-16 images → 8-16 random patches

### Training Loop Pattern:
```python
for epoch in range(num_epochs):
    for batch in dataloader:  # batch has 8-16 image pairs
        blur_patches = []
        sharp_patches = []
        
        # Call random_patch_pair() ONCE per image in batch
        for blur_img, sharp_img in batch:
            b_patch, s_patch = random_patch_pair(blur_img, sharp_img, 256)
            blur_patches.append(b_patch)
            sharp_patches.append(s_patch)
        
        # Stack into batch tensors
        blur_batch = torch.stack(blur_patches)   # [8-16, 3, 256, 256]
        sharp_batch = torch.stack(sharp_patches)
        
        # Train on batch
        output = model(blur_batch)
        loss = criterion(output, sharp_batch)
```

### Summary:
- Batch size = 8-16 images
- One random 256×256 patch extracted from each image
- Result: 8-16 patches per training batch
- Different random locations each epoch