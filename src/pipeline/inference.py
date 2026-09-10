import torch
import torch.nn.functional as F
from src.models.wav2vec_classifier import Wav2Vec2Classifier
from src.data.preprocessor import AudioPreprocessor
import time

class SERInferencePipeline:
    
    EMOTIONS = ["calm", "happy", "angry", "stressed"]
    
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.preprocessor = AudioPreprocessor()
        
        # Load model (assuming Wav2Vec2 for now, can be abstracted)
        self.model = Wav2Vec2Classifier(n_classes=4).to(self.device)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        except FileNotFoundError:
            pass # Handle gracefully in API if model not found
            
        self.model.eval()
        
    def predict(self, audio_tensor: torch.Tensor):
        start_time = time.time()
        
        # Preprocess to match length/sample rate expected by model
        # Preprocessor returns (seq_len,). Add batch dim.
        if len(audio_tensor.shape) == 1:
            inputs = audio_tensor.unsqueeze(0).to(self.device)
        else:
            inputs = audio_tensor.to(self.device)
            
        with torch.no_grad():
            logits = self.model(inputs)
            probs = F.softmax(logits, dim=1).squeeze(0)
            
        confidences = probs.cpu().numpy()
        pred_idx = confidences.argmax()
        
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            "emotion": self.EMOTIONS[pred_idx],
            "confidence": float(confidences[pred_idx]),
            "distribution": {
                self.EMOTIONS[i]: float(confidences[i]) for i in range(4)
            },
            "processing_latency_ms": latency_ms
        }
