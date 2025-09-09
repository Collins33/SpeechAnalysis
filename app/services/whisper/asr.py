import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration


class WhisperASR:
    def __init__(self, model_size: str = "small"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = f"openai/whisper-{model_size}"

        # Load processor and model
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id).to(self.device)

    def transcribe_audio(self, audio_path: str) -> str:
        # Load audio file
        waveform, sample_rate = torchaudio.load(audio_path)

        # Ensure 16kHz sample rate
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)

        # Preprocess audio for Whisper
        inputs = self.processor(
            waveform.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(self.device)

        # Generate prediction
        predicted_ids = self.model.generate(inputs)

        # Decode transcription
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription
