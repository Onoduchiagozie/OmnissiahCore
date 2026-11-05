"""OpenAI Responses API with bounded retries and completion checks."""
import json
import logging
import os
import time
import requests

log = logging.getLogger(__name__)
RETRYABLE = {408, 429, 500, 502, 503, 504}


class OpenAIProviderError(RuntimeError):
    pass


def generate(config, system, user, stream=False):
    key = os.getenv('OPENAI_API_KEY')
    if not key:
        raise OpenAIProviderError('OPENAI_API_KEY is not configured.')
    models = list(dict.fromkeys([config['model']] + config.get('fallback_models', [])))
    retries = max(0, min(int(config.get('retries_per_model', 1)), 2))
    deadline = time.monotonic() + min(float(config.get('timeout', 240)), 240)
    last_error = 'no response'
    for model in models:
        for attempt in range(retries + 1):
            emitted = False
            completed = False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise OpenAIProviderError('OpenAI request time budget exhausted.')
            payload = {
                'model': model, 'instructions': system, 'input': user,
                'max_output_tokens': config.get('max_tokens', 8192),
                'stream': stream, 'store': False,
            }
            try:
                with requests.post(
                    config.get('api_base', 'https://api.openai.com/v1').rstrip('/')+'/responses',
                    headers={'Authorization': 'Bearer '+key}, json=payload,
                    stream=stream, timeout=(min(10, remaining), min(60, remaining)),
                ) as response:
                    status = response.status_code
                    log.info('OpenAI model=%s attempt=%d status=%d request_id=%s', model, attempt+1, status, response.headers.get('x-request-id', ''))
                    if status >= 400:
                        last_error = f'HTTP {status} from {model}'
                        if status == 429:
                            try:
                                code = response.json().get('error', {}).get('code')
                            except ValueError:
                                code = None
                            if code == 'insufficient_quota':
                                raise OpenAIProviderError('OpenAI API credit or spending quota exhausted; check API billing. Model switching cannot fix this.')
                        if status not in RETRYABLE:
                            raise OpenAIProviderError('OpenAI '+last_error+'; request not retried.')
                    elif stream:
                        for line in response.iter_lines(chunk_size=1):
                            if time.monotonic() > deadline:
                                raise OpenAIProviderError('OpenAI stream time budget exhausted.')
                            if not line or not line.startswith(b'data:'):
                                continue
                            raw = line[5:].strip()
                            if raw == b'[DONE]':
                                break
                            event = json.loads(raw)
                            kind = event.get('type')
                            if kind == 'response.output_text.delta' and event.get('delta'):
                                emitted = True
                                yield event['delta']
                            elif kind == 'response.completed':
                                _require_complete(event.get('response', {}))
                                completed = True
                            elif kind in ('response.incomplete', 'response.failed', 'error'):
                                raise OpenAIProviderError('OpenAI generation did not complete: '+str(kind))
                        if not completed:
                            raise OpenAIProviderError('OpenAI stream ended without confirmed completion.')
                        if not emitted:
                            raise OpenAIProviderError('OpenAI completed without answer text.')
                        return
                    else:
                        data = response.json()
                        _require_complete(data)
                        texts = [part.get('text', '') for item in data.get('output', []) if item.get('type') == 'message' for part in item.get('content', []) if part.get('type') == 'output_text']
                        text = ''.join(texts)
                        if not text.strip():
                            raise OpenAIProviderError('OpenAI completed without answer text.')
                        yield text
                        return
            except (requests.RequestException, ValueError) as exc:
                if emitted:
                    raise OpenAIProviderError('OpenAI stream interrupted after text began; no model switch performed.') from None
                last_error = type(exc).__name__
            if attempt < retries:
                time.sleep(min(2**attempt, max(0, deadline-time.monotonic())))
    raise OpenAIProviderError('OpenAI models exhausted: '+last_error)


def _require_complete(data):
    if data.get('status') != 'completed':
        reason = (data.get('incomplete_details') or {}).get('reason', data.get('status', 'missing status'))
        raise OpenAIProviderError('OpenAI answer incomplete: '+str(reason))
