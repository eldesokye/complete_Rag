from .OCREnums import OCREnums
from stores.OCR.providers.MistralProvider import MistralProvider
import logging

logger = logging.getLogger("uvicorn.error")

class OCRProviderFactory:
    def __init__(self, config):
        self.config = config

    def create_provider(self, provider: str):
        if not getattr(self.config, "OCR_ENABLED", True):
            return None

        if provider == OCREnums.MISTRAL.value:
            if not getattr(self.config, "MISTRAL_API_KEY", None):
                logger.warning("MISTRAL_API_KEY not configured for OCR")
                return None
            return MistralProvider(
                api_key=self.config.MISTRAL_API_KEY,
                model_id=getattr(self.config, "OCR_MODEL_ID", "mistral-ocr-latest"),
                include_image_base64=True,
            )
        
        logger.warning(f"Unsupported OCR provider requested: {provider}")
        return None
