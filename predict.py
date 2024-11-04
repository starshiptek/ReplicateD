import os
import torch
from typing import List
from cog import BasePredictor, Input, Path
import decord
import cv2
from PIL import Image
import numpy as np
from transformers import AutoProcessor, AutoModelForVision2Seq
import concurrent.futures
import requests
from tqdm import tqdm

REPLICATE_BASE_URL = "https://weights.replicate.delivery/default/Video-LLaVA-7B"

def download_file(url: str, filename: str):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        
        # Create cache dir if it doesn't exist
        if not os.path.exists("cache"):
            os.makedirs("cache")

        # Define model files to download
        files_to_download = {
            "pytorch_model.bin": f"{REPLICATE_BASE_URL}/pytorch_model.bin",
            "config.json": f"{REPLICATE_BASE_URL}/config.json",
            "generation_config.json": f"{REPLICATE_BASE_URL}/generation_config.json",
            "preprocessor_config.json": f"{REPLICATE_BASE_URL}/preprocessor_config.json",
            "special_tokens_map.json": f"{REPLICATE_BASE_URL}/special_tokens_map.json",
            "tokenizer_config.json": f"{REPLICATE_BASE_URL}/tokenizer_config.json",
            "tokenizer.model": f"{REPLICATE_BASE_URL}/tokenizer.model",
        }

        # Download files in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for filename, url in files_to_download.items():
                if not os.path.exists(f"cache/{filename}"):
                    future = executor.submit(download_file, url, f"cache/{filename}")
                    futures.append(future)
            concurrent.futures.wait(futures)

        # Load model and processor
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForVision2Seq.from_pretrained(
            "cache",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained("cache")

    def extract_frames(self, video_path: str, num_frames: int = 8) -> List[Image.Image]:
        """Extract frames from video"""
        video = decord.VideoReader(video_path)
        total_frames = len(video)
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = video.get_batch(indices).asnumpy()
        return [Image.fromarray(frame) for frame in frames]

    def predict(
        self,
        video: Path = Input(description="Video file to analyze"),
        prompt: str = Input(
            description="Prompt for the model",
            default="Describe what is happening in this video."
        ),
    ) -> str:
        """Run a single prediction on the model"""
        frames = self.extract_frames(str(video))
        inputs = self.processor(
            images=frames,
            text=prompt,
            return_tensors="pt"
        ).to(self.device, torch.float16)

        outputs = self.model.generate(
            **inputs,
            max_length=512,
            num_beams=5,
            temperature=0.9,
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )
        
        response = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return response.strip()
