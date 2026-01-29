"""
Preprocessing module for liver fibrosis images.
Implements CLAHE enhancement and image resizing.
"""
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

import sys
sys.path.insert(0, str(__file__).rsplit('src', 1)[0])
from config import IMAGE_SIZE, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance image contrast.
    
    CLAHE is particularly effective for medical images as it enhances local contrast
    while limiting noise amplification.
    
    Args:
        image: Input image as numpy array (BGR or grayscale)
        
    Returns:
        CLAHE-enhanced image
    """
    # Convert to LAB color space if color image
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert BGR to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=CLAHE_TILE_GRID_SIZE
        )
        l_enhanced = clahe.apply(l_channel)
        
        # Merge channels and convert back to BGR
        enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
        enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    else:
        # Grayscale image
        clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=CLAHE_TILE_GRID_SIZE
        )
        enhanced_image = clahe.apply(image)
    
    return enhanced_image


def resize_image(image: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """
    Resize image to specified square dimensions.
    
    Args:
        image: Input image as numpy array
        size: Target size (default: 384x384 from config)
        
    Returns:
        Resized image
    """
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_LANCZOS4)


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Complete preprocessing pipeline: load, CLAHE, resize.
    
    Args:
        image_path: Path to the input image
        
    Returns:
        Preprocessed image as numpy array
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Apply CLAHE enhancement
    enhanced = apply_clahe(image)
    
    # Resize to target size
    resized = resize_image(enhanced)
    
    return resized


class CLAHETransform:
    """
    PyTorch-compatible transform that applies CLAHE enhancement.
    Can be used in torchvision.transforms.Compose pipelines.
    """
    
    def __init__(self, clip_limit: float = CLAHE_CLIP_LIMIT, 
                 tile_grid_size: tuple = CLAHE_TILE_GRID_SIZE):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Apply CLAHE to PIL Image.
        
        Args:
            img: PIL Image
            
        Returns:
            CLAHE-enhanced PIL Image
        """
        # Convert PIL to numpy
        img_array = np.array(img)
        
        # Apply CLAHE (handle RGB vs BGR)
        if len(img_array.shape) == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            enhanced = apply_clahe(img_bgr)
            enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        else:
            clahe = cv2.createCLAHE(
                clipLimit=self.clip_limit,
                tileGridSize=self.tile_grid_size
            )
            enhanced_rgb = clahe.apply(img_array)
        
        return Image.fromarray(enhanced_rgb)
    
    def __repr__(self):
        return f"CLAHETransform(clip_limit={self.clip_limit}, tile_grid_size={self.tile_grid_size})"


def get_train_transforms() -> transforms.Compose:
    """
    Get training data augmentation transforms.
    
    Returns:
        Composed transforms for training data
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        CLAHETransform(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_val_transforms() -> transforms.Compose:
    """
    Get validation/test transforms (no augmentation).
    
    Returns:
        Composed transforms for validation data
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        CLAHETransform(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Denormalize a tensor for visualization.
    
    Args:
        tensor: Normalized image tensor
        
    Returns:
        Denormalized image as numpy array (0-255 uint8)
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    if tensor.device.type == 'cuda':
        mean = mean.cuda()
        std = std.cuda()
    
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy (H, W, C) format
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = (img * 255).astype(np.uint8)
    
    return img
