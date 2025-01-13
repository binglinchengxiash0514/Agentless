import os
import time
from typing import Dict, Optional, Union

import anthropic
import openai
import tiktoken
from azure.identity import DefaultAzureCredential
from openai import AzureOpenAI


def num_tokens_from_messages(message, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if isinstance(message, list):
        # use last message.
        num_tokens = len(encoding.encode(message[0]["content"]))
    else:
        num_tokens = len(encoding.encode(message))
    return num_tokens


def create_chatgpt_config(
    message: Union[str, list],
    max_tokens: int,
    temperature: float = 1,
    batch_size: int = 1,
    system_message: str = "You are a helpful assistant.",
    model: str = "gpt-3.5-turbo",
) -> Dict:
    if isinstance(message, list):
        config = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": batch_size,
            "messages": [{"role": "system", "content": system_message}] + message,
        }
    else:
        config = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": batch_size,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": message},
            ],
        }
    return config


def handler(signum, frame):
    # swallow signum and frame
    raise Exception("end of time")


def request_chatgpt_engine(config, logger, base_url=None, max_retries=40, timeout=100):
    ret = None
    retries = 0

    client = openai.OpenAI(base_url=base_url)

    while ret is None and retries < max_retries:
        try:
            # Attempt to get the completion
            logger.info("Creating API request")

            ret = client.chat.completions.create(**config)

        except openai.OpenAIError as e:
            if isinstance(e, openai.BadRequestError):
                logger.info("Request invalid")
                print(e)
                logger.info(e)
                raise Exception("Invalid API Request")
            elif isinstance(e, openai.RateLimitError):
                print("Rate limit exceeded. Waiting...")
                logger.info("Rate limit exceeded. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(5)
            elif isinstance(e, openai.APIConnectionError):
                print("API connection error. Waiting...")
                logger.info("API connection error. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(5)
            else:
                print("Unknown error. Waiting...")
                logger.info("Unknown error. Waiting...")
                print(e)
                logger.info(e)
                time.sleep(1)

        retries += 1

    logger.info(f"API response {ret}")
    return ret


def create_anthropic_config(
    message: str,
    max_tokens: int,
    temperature: float = 1,
    batch_size: int = 1,
    system_message: str = "You are a helpful assistant.",
    model: str = "claude-2.1",
    tools: list = None,
) -> Dict:
    if isinstance(message, list):
        config = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": message,
        }
    else:
        config = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": message}]},
            ],
        }

    if tools:
        config["tools"] = tools

    return config


def request_azure_engine(
    config,
    logger,
    azure_endpoint: Optional[str] = None,
    azure_tenant_id: Optional[str] = None,
    azure_client_id: Optional[str] = None,
    azure_api_key: Optional[str] = None,
    max_retries: int = 40,
    timeout: int = 100,
) -> Optional[Dict]:
    """Send a request to Azure OpenAI's chat completions API.

    Args:
        config: Configuration dictionary for the request
        logger: Logger instance for tracking API interactions
        azure_endpoint: Azure OpenAI endpoint URL (optional, defaults to AZURE_OPENAI_ENDPOINT env var)
        azure_tenant_id: Azure tenant ID for AD auth (optional, defaults to AZURE_TENANT_ID env var)
        azure_client_id: Azure client ID for AD auth (optional, defaults to AZURE_CLIENT_ID env var)
        azure_api_key: Azure API key (optional, defaults to AZURE_OPENAI_KEY env var)
        max_retries: Maximum number of retry attempts (default: 40)
        timeout: Timeout in seconds (default: 100)

    Returns:
        API response object or None if request fails
    """
    ret = None
    retries = 0

    # Get Azure OpenAI configuration from parameters or environment
    azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
    if not azure_endpoint:
        raise ValueError(
            "Azure endpoint must be provided via parameter or AZURE_OPENAI_ENDPOINT environment variable"
        )

    try:
        # Use Azure AD credentials if available, fallback to API key
        azure_tenant_id = azure_tenant_id or os.getenv("AZURE_TENANT_ID")
        azure_client_id = azure_client_id or os.getenv("AZURE_CLIENT_ID")

        if azure_tenant_id and azure_client_id:
            credential = DefaultAzureCredential()
            client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_version="2024-02-15-preview",
                azure_ad_token_provider=credential,
            )
        else:
            api_key = azure_api_key or os.getenv("AZURE_OPENAI_KEY")
            if not api_key:
                raise ValueError(
                    "Azure API key must be provided via parameter or AZURE_OPENAI_KEY environment variable when not using Azure AD"
                )
            client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                api_version="2024-02-15-preview",
            )

        while ret is None and retries < max_retries:
            try:
                logger.info("Creating Azure API request")
                ret = client.chat.completions.create(**config)

            except openai.OpenAIError as e:
                if isinstance(e, openai.BadRequestError):
                    logger.info("Azure request invalid")
                    print(e)
                    logger.info(e)
                    raise Exception("Invalid Azure API Request")
                elif isinstance(e, openai.RateLimitError):
                    print("Azure rate limit exceeded. Waiting...")
                    logger.info("Azure rate limit exceeded. Waiting...")
                    print(e)
                    logger.info(e)
                    time.sleep(5)
                elif isinstance(e, openai.APIConnectionError):
                    print("Azure API connection error. Waiting...")
                    logger.info("Azure API connection error. Waiting...")
                    print(e)
                    logger.info(e)
                    time.sleep(5)
                elif isinstance(e, openai.AuthenticationError):
                    logger.error("Azure authentication failed")
                    print(e)
                    logger.error(e)
                    raise Exception("Azure Authentication Failed")
                else:
                    print("Unknown error. Waiting...")
                    logger.info("Unknown error. Waiting...")
                    print(e)
                    logger.info(e)
                    time.sleep(1)

            retries += 1

        logger.info(f"Azure API response {ret}")
        return ret

    except Exception as e:
        logger.error(f"Azure authentication error: {e}")
        raise


def request_anthropic_engine(
    config, logger, max_retries=40, timeout=500, prompt_cache=False
):
    ret = None
    retries = 0

    client = anthropic.Anthropic()

    while ret is None and retries < max_retries:
        try:
            start_time = time.time()
            if prompt_cache:
                # following best practice to cache mainly the reused content at the beginning
                # this includes any tools, system messages (which is already handled since we try to cache the first message)
                config["messages"][0]["content"][0]["cache_control"] = {
                    "type": "ephemeral"
                }
                ret = client.beta.prompt_caching.messages.create(**config)
            else:
                ret = client.messages.create(**config)
        except Exception as e:
            logger.error("Unknown error. Waiting...", exc_info=True)
            # Check if the timeout has been exceeded
            if time.time() - start_time >= timeout:
                logger.warning("Request timed out. Retrying...")
            else:
                logger.warning("Retrying after an unknown error...")
            time.sleep(10 * retries)
        retries += 1

    return ret
