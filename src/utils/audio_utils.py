import torch
import soundfile as sf
import io

def buffer_to_tensor(buffer: bytes, target_sr: int = 16000) -> torch.Tensor:
    """Reads audio bytes into a torch tensor."""
    audio_data, sr = sf.read(io.BytesIO(buffer))
    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
        
    tensor = torch.from_numpy(audio_data).float()
    
    if sr != target_sr:
        import torchaudio.transforms as T
        resampler = T.Resample(sr, target_sr)
        tensor = resampler(tensor)
        
    return tensor
