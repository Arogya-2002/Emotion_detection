from PIL import Image
from src.exceptions import CustomException
from src.logger import logging
import sys
from src.entity.config import ConfigEntity, DetectFaceConfig
from src.entity.artifact import DetectFaceArtifact  # Ensure this import path is valid


class FaceDetector:
    def __init__(self):
        try:
            logging.info("Initializing FaceDetector...")
            self.detect_face_config = DetectFaceConfig(config=ConfigEntity())
            logging.info("Face detector configuration loaded successfully.")
        except Exception as e:
            logging.error("Failed to initialize FaceDetector", exc_info=True)
            raise CustomException(e, sys)

    def detect_faces(self, image: Image.Image) -> DetectFaceArtifact:
        try:
            logging.info("Starting face detection...")
            boxes, _ = self.detect_face_config.detector.detect(image)
            logging.info(f"Detected {len(boxes)} faces.")

            # Ensure the box values are tuples
            formatted_boxes = [tuple(box) for box in boxes]
            return DetectFaceArtifact(boxes=formatted_boxes)

        except Exception as e:
            logging.error("Error during face detection", exc_info=True)
            raise CustomException(e, sys)
