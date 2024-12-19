from pydantic_settings import BaseSettings
from dotenv import load_dotenv


class AppConfig(BaseSettings):
    # LLM
    openai_llm_model_gpt4_turbo: str = "gpt4-turbo"
    openai_llm_deployment_name_gpt4_turbo: str = "gpt4-turbo"
    openai_azure_endpoint_gpt4_turbo: str = "https://genai-nexus.api.corpinter.net/apikey/"
    openai_api_version_gpt4_turbo: str = "2023-07-01-preview"
    openai_api_key_gpt4_turbo: str

    openai_llm_model_gpt4o: str = "gpt4-o"
    openai_llm_deployment_name_gpt4o: str = "gpt4-o"
    openai_azure_endpoint_gpt4o: str = "https://genai-nexus.api.corpinter.net/apikey/openai/deployments/gpt-4o/chat/completions"
    openai_api_version_gpt4o: str = "2023-07-01-preview"
    openai_api_key_gpt4o: str


def read_settings() -> AppConfig:
    load_dotenv()
    return AppConfig()
