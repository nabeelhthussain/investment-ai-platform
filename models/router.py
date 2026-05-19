"""
LLM Router — single interface for all Claude API calls.

All agents import call_llm() from here. Swapping models, adding fallbacks,
or switching providers only requires changes in this one file.
"""
import time
import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from config import ANTHROPIC_API_KEY, PRIMARY_MODEL, FAST_MODEL


_client = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        if not ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key."
            )
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APIStatusError)),
)
def call_llm(
    prompt: str,
    system: str = "",
    model: str = None,
    max_tokens: int = 4096,
    temperature: float = 0.2,
) -> str:
    """
    Core LLM call. Returns the text response as a string.

    Args:
        prompt:     User message content.
        system:     System prompt. Defaults to empty.
        model:      Override model. Defaults to PRIMARY_MODEL.
        max_tokens: Max tokens in response.
        temperature: 0.0 = deterministic, higher = more creative.

    Returns:
        Response text as string.
    """
    client = get_client()
    model = model or PRIMARY_MODEL

    messages = [{"role": "user", "content": prompt}]

    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
        temperature=temperature,
    )
    if system:
        kwargs["system"] = system

    response = client.messages.create(**kwargs)
    return response.content[0].text


def call_llm_fast(prompt: str, system: str = "", max_tokens: int = 2048) -> str:
    """Use the fast/cheap model for simple extraction tasks."""
    return call_llm(prompt, system=system, model=FAST_MODEL, max_tokens=max_tokens)


def call_llm_with_context(
    query: str,
    context_chunks: list[dict],
    system: str,
    max_tokens: int = 4096,
) -> str:
    """
    Standard RAG call: inject retrieved chunks into the prompt,
    instruct the model to cite sources inline.

    Each chunk dict should have: text, source, doc_type, date, section.
    """
    if not context_chunks:
        context_block = "[No source documents retrieved for this query.]"
    else:
        parts = []
        for i, chunk in enumerate(context_chunks):
            source_label = (
                f"[Source {i+1}: {chunk.get('doc_type','unknown')} | "
                f"{chunk.get('date','n/d')} | "
                f"{chunk.get('section','')}"
                f"{' | p.' + str(chunk.get('page','')) if chunk.get('page') else ''}]"
            )
            parts.append(f"{source_label}\n{chunk['text']}")
        context_block = "\n\n---\n\n".join(parts)

    citation_instruction = (
        "\n\nIMPORTANT: Every factual claim you make must be followed immediately by "
        "its source citation in the format [Source N]. If you cannot find supporting "
        "evidence in the provided source documents, explicitly state "
        "'[UNCERTAIN — not grounded in available documents]' rather than inferring. "
        "Do not fabricate information."
    )

    full_prompt = (
        f"## Retrieved Source Documents\n\n{context_block}\n\n"
        f"## Query\n\n{query}{citation_instruction}"
    )

    return call_llm(full_prompt, system=system, max_tokens=max_tokens)
