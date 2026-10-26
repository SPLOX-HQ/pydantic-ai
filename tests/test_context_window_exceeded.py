"""Each model class raises `ContextWindowExceeded` from its provider's own sign of a request over the
model's context window, and a plain `ModelHTTPError` for every other refusal."""

from __future__ import annotations

import json
import pickle
from typing import Any, Literal

import httpx2
import pytest

from pydantic_ai import Agent, ContextWindowExceeded, ModelHTTPError
from pydantic_ai.models import Model

from .conftest import try_import

with try_import() as openai_imports:
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

with try_import() as anthropic_imports:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

with try_import() as google_imports:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

Provider = Literal['openai-chat', 'openai-responses', 'anthropic', 'google']

# The providers' own answers to a request over the window.
OVER = {
    'openai-chat': {
        'error': {
            'message': "This model's maximum context length is 128000 tokens. However, your messages resulted in 130001 tokens. Please reduce the length of the messages.",
            'type': 'invalid_request_error',
            'param': 'messages',
            'code': 'context_length_exceeded',
        }
    },
    'openai-responses': {
        'error': {
            'message': 'Your input exceeds the context window of this model. Please adjust your input and try again.',
            'type': 'invalid_request_error',
            'param': 'input',
            'code': 'context_length_exceeded',
        }
    },
    'anthropic': {
        'type': 'error',
        'error': {'type': 'invalid_request_error', 'message': 'prompt is too long: 208310 tokens > 200000 maximum'},
    },
    'google': {
        'error': {
            'code': 400,
            'message': 'The input token count (1048577) exceeds the maximum number of tokens allowed (1048576).',
            'status': 'INVALID_ARGUMENT',
        }
    },
}

# Another refusal of the same request, in the same shape: not over the window.
OTHER = {
    'openai-chat': {
        'error': {'message': 'Invalid value.', 'type': 'invalid_request_error', 'param': 'x', 'code': 'invalid_value'}
    },
    'openai-responses': {
        'error': {'message': 'Invalid value.', 'type': 'invalid_request_error', 'param': 'x', 'code': 'invalid_value'}
    },
    'anthropic': {
        'type': 'error',
        'error': {'type': 'invalid_request_error', 'message': 'messages: text content blocks must be non-empty'},
    },
    'google': {
        'error': {'code': 400, 'message': 'Request contains an invalid argument.', 'status': 'INVALID_ARGUMENT'}
    },
}


def model_answering(provider: Provider, body: dict[str, Any]) -> Model:
    def answer(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(400, json=body)

    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(answer))
    if provider == 'openai-chat':
        return OpenAIChatModel('gpt-5.2', provider=OpenAIProvider(api_key='k', http_client=http_client))
    if provider == 'openai-responses':
        return OpenAIResponsesModel('gpt-5.2', provider=OpenAIProvider(api_key='k', http_client=http_client))
    if provider == 'anthropic':
        return AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='k', http_client=http_client))
    return GoogleModel('gemini-2.5-pro', provider=GoogleProvider(api_key='k', http_client=http_client))


PROVIDERS = [
    pytest.param('openai-chat', marks=pytest.mark.skipif(not openai_imports(), reason='openai not installed')),
    pytest.param('openai-responses', marks=pytest.mark.skipif(not openai_imports(), reason='openai not installed')),
    pytest.param('anthropic', marks=pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed')),
    pytest.param('google', marks=pytest.mark.skipif(not google_imports(), reason='google not installed')),
]


@pytest.mark.parametrize('provider', PROVIDERS)
@pytest.mark.parametrize('stream', [False, True])
async def test_over_the_window_is_context_window_exceeded(provider: Provider, stream: bool, allow_model_requests: None):
    agent = Agent(model_answering(provider, OVER[provider]))
    with pytest.raises(ContextWindowExceeded) as raised:
        if stream:
            async with agent.run_stream('hello') as result:
                await result.get_output()
        else:
            await agent.run('hello')
    assert raised.value.status_code == 400
    assert raised.value.__cause__ is not None
    # it crosses a durable step as itself
    assert type(pickle.loads(pickle.dumps(raised.value))) is ContextWindowExceeded


@pytest.mark.parametrize('provider', PROVIDERS)
async def test_another_refusal_is_a_model_http_error(provider: Provider, allow_model_requests: None):
    with pytest.raises(ModelHTTPError) as raised:
        await Agent(model_answering(provider, OTHER[provider])).run('hello')
    assert type(raised.value) is ModelHTTPError
    assert json.dumps(raised.value.body)
