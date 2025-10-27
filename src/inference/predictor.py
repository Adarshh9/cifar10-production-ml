"""CIFAR-10 Predictor with Redis caching."""
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import hashlib
import json
import logging
from typing import List, Dict, Union

logger = logging.getLogger(__name__)


class CIFAR10Predictor:
    """Predictor for CIFAR-10 classification with caching."""
    
    # CIFAR-10 class names
    CLASSES = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    def __init__(self, model_loader, use_cache=True, cache_ttl=3600):
        self.model_loader = model_loader
        self.use_cache = use_cache
        self.cache_ttl = cache_ttl
        self.cache = None
        
        # Transform for CIFAR-10
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616]
            )
        ])
        
        logger.info("CIFAR10Predictor initialized")
    
    def set_cache(self, cache):
        """Set cache instance."""
        self.cache = cache
        logger.info(f"Cache connected: {cache is not None}")
    
    def _preprocess_image(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """Preprocess image for model input."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return self.transform(image).unsqueeze(0)
    
    def _compute_input_hash(self, image: Image.Image) -> str:
        """Compute hash of input image for caching."""
        # Convert to consistent format for hashing
        img_array = np.array(image.convert('RGB'))
        image_bytes = img_array.tobytes()
        return hashlib.md5(image_bytes).hexdigest()
    
    def predict(self, image: Union[Image.Image, np.ndarray]) -> Dict:
        """
        Predict class for single image.
        
        Args:
            image: PIL Image or numpy array
        
        Returns:
            Dict with prediction results
        """
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Check cache
        cache_key = None
        if self.use_cache and self.cache:
            cache_key = f"pred:{self._compute_input_hash(image)}"
            try:
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    result = json.loads(cached_result)
                    result['cached'] = True
                    logger.info(f"✅ Cache HIT for key: {cache_key[:16]}...")
                    return result
            except Exception as e:
                logger.warning(f"Cache get failed: {e}")
        
        # Preprocess
        input_tensor = self._preprocess_image(image)
        input_tensor = input_tensor.to(self.model_loader.device)
        
        # Inference
        self.model_loader.model.eval()
        with torch.no_grad():
            output = self.model_loader.model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
            # Get prediction
            confidence, predicted = torch.max(probabilities, 0)
            
            result = {
                'prediction': int(predicted.item()),
                'class_name': self.CLASSES[predicted.item()],
                'probabilities': probabilities.cpu().numpy().tolist(),
                'confidence': float(confidence.item()),
                'model_version': self.model_loader.model_version,
                'cached': False
            }
        
        # Store in cache
        if self.use_cache and self.cache and cache_key:
            try:
                self.cache.set(cache_key, json.dumps(result), ttl=self.cache_ttl)
                logger.info(f"✅ Cached result for key: {cache_key[:16]}...")
            except Exception as e:
                logger.warning(f"Cache set failed: {e}")
        
        return result
    
    def predict_batch(self, images: List[Union[Image.Image, np.ndarray]]) -> Dict:
        """
        Predict classes for multiple images.
        
        Args:
            images: List of PIL Images or numpy arrays
        
        Returns:
            Dict with batch prediction results
        """
        # Convert all to PIL if needed
        pil_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                pil_images.append(Image.fromarray(img))
            else:
                pil_images.append(img)
        
        # Preprocess all images
        batch_tensor = torch.cat([self._preprocess_image(img) for img in pil_images])
        batch_tensor = batch_tensor.to(self.model_loader.device)
        
        # Inference
        self.model_loader.model.eval()
        with torch.no_grad():
            outputs = self.model_loader.model(batch_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            confidences, predictions = torch.max(probabilities, 1)
            
            # Format results
            predictions_list = predictions.cpu().numpy().tolist()
            confidences_list = confidences.cpu().numpy().tolist()
            probs_list = probabilities.cpu().numpy().tolist()
            
            return {
                'predictions': predictions_list,
                'class_names': [self.CLASSES[p] for p in predictions_list],
                'confidences': confidences_list,
                'probabilities': probs_list,
                'batch_size': len(images),
                'model_version': self.model_loader.model_version
            }
