from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch
from src.exceptions import CustomException
from src.logger import logging
import sys

from src.entity.config import ClassifyEmotionConfig, ConfigEntity
from src.entity.artifact import ClassifyEmotionArtifact  # Ensure this import is correct

class EmotionClassifier:
    def __init__(self):
        try:
            logging.info("Initializing EmotionClassifier...")
            self.classify_emotion_config = ClassifyEmotionConfig(config=ConfigEntity())
            self.processor = ViTImageProcessor.from_pretrained(self.classify_emotion_config.model_name)
            self.model = ViTForImageClassification.from_pretrained(self.classify_emotion_config.model_name)
            logging.info(f"Model and processor loaded from {self.classify_emotion_config.model_name}")

        except Exception as e:
            logging.error("Failed to initialize EmotionClassifier", exc_info=True)
            raise CustomException(e, sys)

    def predict(self, face_img: Image.Image) -> ClassifyEmotionArtifact:
        try:
            logging.info("Starting prediction process...")
            if face_img.mode != "RGB":
                logging.info("Image mode is not RGB. Converting...")
                face_img = face_img.convert("RGB")

            inputs = self.processor(images=face_img, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)

            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]
            idx = int(probs.argmax(-1))
            label = self.model.config.id2label[idx]
            conf = probs[idx].item()

            logging.info(f"Prediction complete: Label={label}, Confidence={conf:.4f}")

            return ClassifyEmotionArtifact(
                emotion_id=idx,
                emotion_label=label,
                confidence=conf
            )

        except Exception as e:
            logging.error("Error during emotion prediction", exc_info=True)
            raise CustomException(e, sys)
