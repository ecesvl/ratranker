from langchain_core.language_models import BaseChatModel
from langchain_openai import AzureChatOpenAI
from src.utils.settings import read_settings


def get_gpt4turbo() -> BaseChatModel:
    config = read_settings()
    return AzureChatOpenAI(
        azure_deployment=config.openai_llm_deployment_name_gpt4_turbo,
        api_version=config.openai_api_version_gpt4_turbo,
        api_key=config.openai_api_key_gpt4_turbo,
        azure_endpoint=config.openai_azure_endpoint_gpt4_turbo,
        tiktoken_model_name=config.openai_llm_model_gpt4_turbo,
    )


def get_gpt4o() -> BaseChatModel:
    config = read_settings()
    return AzureChatOpenAI(
        azure_deployment=config.openai_llm_deployment_name_gpt4o,
        api_version=config.openai_api_version_gpt4o,
        api_key=config.openai_api_key_gpt4o,
        azure_endpoint=config.openai_azure_endpoint_gpt4o,
        tiktoken_model_name=config.openai_llm_model_gpt4o,
    )
