from typing import NamedTuple
import os
from dotenv import load_dotenv
from src.logger import logger

class Config(NamedTuple):
    azure_openai_api_key: str
    azure_openai_endpoint: str
    azure_openai_api_version: str
    youtube_api_key: str

_config = None

def get_config() -> Config:
    global _config
    if _config is None:
        load_dotenv()
        
        required_vars = [
            'AZURE_OPENAI_API_KEY',
            'AZURE_OPENAI_ENDPOINT',
            'AZURE_OPENAI_API_VERSION',
            'YOUTUBE_API_KEY'
        ]
        
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
            
        _config = Config(
            azure_openai_api_key=os.getenv('AZURE_OPENAI_API_KEY'),
            azure_openai_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            azure_openai_api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
            youtube_api_key=os.getenv('YOUTUBE_API_KEY')
        )
        logger.info("Configuration loaded successfully")
    return _config