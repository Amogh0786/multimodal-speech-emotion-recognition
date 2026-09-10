import os
import zipfile
import urllib.request
from loguru import logger
from pathlib import Path

class RAVDESSDownloader:
    """Downloads and extracts the RAVDESS dataset."""
    
    URL = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
    
    def __init__(self, raw_dir: str):
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.zip_path = self.raw_dir / "ravdess.zip"
        
    def download(self):
        if not (self.raw_dir / "Actor_01").exists():
            logger.info(f"Downloading RAVDESS dataset to {self.zip_path}...")
            urllib.request.urlretrieve(self.URL, self.zip_path)
            logger.info("Extracting dataset...")
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.raw_dir)
            logger.info("Extraction complete.")
            os.remove(self.zip_path)
        else:
            logger.info("RAVDESS dataset already exists.")

if __name__ == "__main__":
    downloader = RAVDESSDownloader("data/raw")
    downloader.download()
