from PIL import Image

from src.components.detect_face import FaceDetector
from src.components.classify_emotion import EmotionClassifier

from src.exceptions import CustomException
from src.logger import logging

from src.entity.config import EmotionProcessorConfig, ConfigEntity
from src.entity.artifact import EmotionProcessArtifact  # Ensure this import is defined correctly

import sys
from typing import List


class EmotionProcessor:
    def __init__(self):
        try:
            logging.info("Initializing EmotionProcessor...")
            self.emotion_processor_config = EmotionProcessorConfig(config=ConfigEntity())
            self.face_detector = FaceDetector()
            self.classifier = EmotionClassifier()
            logging.info("EmotionProcessor initialized successfully.")
        except Exception as e:
            logging.error("Failed to initialize EmotionProcessor", exc_info=True)
            raise CustomException(e, sys)

    def process_image(self, image: Image.Image) -> List[EmotionProcessArtifact]:
        try:
            logging.info("Starting emotion processing on image...")
            result: List[EmotionProcessArtifact] = []
            boxes_artifact = self.face_detector.detect_faces(image)
            boxes = boxes_artifact.boxes

            if boxes is None or len(boxes) == 0:
                logging.warning("No faces detected in the image.")
                return result

            for i, box in enumerate(boxes):
                try:
                    x1, y1, x2, y2 = [int(b) for b in box]
                    face = image.crop((x1, y1, x2, y2))
                    prediction_artifact = self.classifier.predict(face)
                    logging.info(f"Face {i+1}: Detected emotion '{prediction_artifact.emotion_label}' with confidence {prediction_artifact.confidence:.4f}")
                    result.append(EmotionProcessArtifact(
                        face_index=i + 1,
                        box=[x1, y1, x2, y2],
                        emotion_label=prediction_artifact.emotion_label,
                        emotion_id=prediction_artifact.emotion_id,
                        confidence=round(prediction_artifact.confidence * 100, 2),
                        emoji=self.emotion_processor_config.emoji_map.get(prediction_artifact.emotion_label.lower(), "")
                    ))
                except Exception as err:
                    logging.error(f"Error processing face {i+1}: {str(err)}", exc_info=True)
                    result.append(EmotionProcessArtifact(
                        face_index=i + 1,
                        box=[0, 0, 0, 0],
                        emotion_label="Error",
                        emotion_id=None,
                        confidence=0.0,
                        emoji="",
                        error=str(err)
                    ))

            return result

        except Exception as e:
            logging.error("Error during image processing", exc_info=True)
            raise CustomException(e, sys)
