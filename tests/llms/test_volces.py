from unittest.mock import Mock, patch
import os
import pytest

from mem0.configs.llms.base import BaseLlmConfig
from mem0.llms.volces import VolcesLLM


@pytest.fixture
def mock_volces_client():
    with patch("mem0.llms.volces.OpenAI") as mock_openai:
        mock_client = Mock()
        mock_openai.return_value = mock_client
        yield mock_client


def test_volces_llm_base_url():
    # case1: default config with volces official base url
    config = BaseLlmConfig(model="ep-20241224105714-46q2f", temperature=0.7, max_tokens=100, top_p=1.0, api_key="api_key")
    llm = VolcesLLM(config)
    assert str(llm.client.base_url) == "https://ark.cn-beijing.volces.com/api/v3/"

    # case2: with env variable VOLCES_API_BASE
    provider_base_url = "https://api.provider.com/v1/"
    os.environ["VOLCES_API_BASE"] = provider_base_url
    config = BaseLlmConfig(model="ep-20241224105714-46q2f", temperature=0.7, max_tokens=100, top_p=1.0, api_key="api_key")
    llm = VolcesLLM(config)
    assert str(llm.client.base_url) == provider_base_url

    # case3: with config.volces_base_url
    config_base_url = "https://api.config.com/v1/"
    config = BaseLlmConfig(
        model="ep-20241224105714-46q2f", 
        temperature=0.7, 
        max_tokens=100, 
        top_p=1.0, 
        api_key="api_key", 
        volces_base_url=config_base_url
    )
    llm = VolcesLLM(config)
    assert str(llm.client.base_url) == config_base_url


def test_generate_response_without_tools(mock_volces_client):
    config = BaseLlmConfig(model="volces-chat", temperature=0.7, max_tokens=100, top_p=1.0)
    llm = VolcesLLM(config)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]

    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="I'm doing well, thank you for asking!"))]
    mock_volces_client.chat.completions.create.return_value = mock_response

    response = llm.generate_response(messages)

    mock_volces_client.chat.completions.create.assert_called_once_with(
        model="ep-20241224105714-46q2f", messages=messages, temperature=0.7, max_tokens=100, top_p=1.0
    )
    assert response == "I'm doing well, thank you for asking!"


def test_generate_response_with_tools(mock_volces_client):
    config = BaseLlmConfig(model="ep-20241224105714-46q2f", temperature=0.7, max_tokens=100, top_p=1.0)
    llm = VolcesLLM(config)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Add a new memory: Today is a sunny day."},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "add_memory",
                "description": "Add a memory",
                "parameters": {
                    "type": "object",
                    "properties": {"data": {"type": "string", "description": "Data to add to memory"}},
                    "required": ["data"],
                },
            },
        }
    ]

    mock_response = Mock()
    mock_message = Mock()
    mock_message.content = "I've added the memory for you."

    mock_tool_call = Mock()
    mock_tool_call.function.name = "add_memory"
    mock_tool_call.function.arguments = '{"data": "Today is a sunny day."}'

    mock_message.tool_calls = [mock_tool_call]
    mock_response.choices = [Mock(message=mock_message)]
    mock_volces_client.chat.completions.create.return_value = mock_response

    response = llm.generate_response(messages, tools=tools)

    mock_volces_client.chat.completions.create.assert_called_once_with(
        model="ep-20241224105714-46q2f", 
        messages=messages, 
        temperature=0.7, 
        max_tokens=100, 
        top_p=1.0, 
        tools=tools, 
        tool_choice="auto"
    )

    assert response["content"] == "I've added the memory for you."
    assert len(response["tool_calls"]) == 1
    assert response["tool_calls"][0]["name"] == "add_memory"
    assert response["tool_calls"][0]["arguments"] == {"data": "Today is a sunny day."}