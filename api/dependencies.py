from src.pipeline.inference import SERInferencePipeline
import torch

# Global instance for dependency injection
_pipeline = None

def get_inference_pipeline() -> SERInferencePipeline:
    global _pipeline
    if _pipeline is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # We assume the model is at root or mounted via docker
        _pipeline = SERInferencePipeline(model_path="best_model.pth", device=device)
    return _pipeline
