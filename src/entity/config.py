# from dataclasses import dataclass

# @dataclass
# class AppConfig:
#     model_name: str = "trpakov/vit-face-expression"
#     keep_all_faces: bool = True


from src.exceptions import CustomException
from src.logger import logging
from src.constants import MODEL_NAME, EMOJI_MAP
import sys
from facenet_pytorch import MTCNN

class ConfigEntity:
    def __init__(self):
        self.model_name = MODEL_NAME
        self.detector = MTCNN(keep_all=True)
        self.emoji_map = EMOJI_MAP


class  ClassifyEmotionConfig:
    def __init__(self,config: ConfigEntity):
        try:
            self.model_name = config.model_name

        except Exception as e:
            raise CustomException(e, sys) from e
        

class DetectFaceConfig:
    def __init__(self, config: ConfigEntity):
        try:
            self.detector = config.detector
            logging.info("Face detector initialized successfully.")
        except Exception as e:
            logging.error("Failed to initialize face detector", exc_info=True)
            raise CustomException(e, sys) from e
        

class EmotionProcessorConfig:
    def __init__(self,config: ConfigEntity):
        try:
             self.emoji_map = config.emoji_map
        except Exception as e:
            logging.error("Failed to initialize EmotionProcessorConfig", exc_info=True)
            raise CustomException(e, sys) from e