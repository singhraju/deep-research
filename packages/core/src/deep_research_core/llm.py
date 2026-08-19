from langchain_openai import ChatOpenAI

from deep_research_utils.app_constant import AppConstants
from deep_research_utils import EHAPBase
import os

if __name__ == "__main__":
    EHAP = EHAPBase(base_url=AppConstants.EHAP_BASE_URL,
                    client_id=AppConstants.EHAP_CLIENT_ID,
                    client_secret=AppConstants.EHAP_CLIENT_SECRET,
                    verify=AppConstants.SSL_CERT_FILE or False)
    # Chat model
    model_medium_reasoning = ChatOpenAI(
        base_url=AppConstants.OPENAI_BASE_URL,
        model=AppConstants.EHAP_LLM_MODEL,
        api_key=EHAP.get_token(),
        # output_version="completions/v1", # todo: check the version EHAP supports.
        # https://reference.langchain.com/python/langchain-openai/chat_models/base/ChatOpenAI
        extra_body={
            "reasoning_effort": "medium",  # Choices: "low", "medium", "high"
            "summary": None  # Choices: 'detailed', 'auto', or None 
        },
        http_client=AppConstants.http_client_,
        http_async_client=AppConstants.http_async_client_,
    )
    ai_msg = model_medium_reasoning.invoke("What are the benefits of LangChain?")

    # Access the text content
    print(ai_msg.content)
