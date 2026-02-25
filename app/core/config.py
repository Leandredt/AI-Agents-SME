cat > app/core/config.py << 'EOL'
import yaml
import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    deepl_api_key: str
    mistral_model_name: str = "mistralai/Mistral-7B-v0.1"
    lora_dir: str = "./data/models/lora"
    datasets_dir: str = "./data/datasets"
    reports_dir: str = "./data/reports"
    templates_dir: str = "./data/templates"

    class Config:
        env_file = ".env"

def load_config():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return Settings(**config)

EOL
