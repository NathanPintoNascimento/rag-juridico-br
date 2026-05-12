import anthropic
from config import ANTHROPIC_API_KEY, LLM_MODEL, LLM_MAX_TOKENS
from retriever import format_context

SYSTEM_PROMPT = """Você é um assistente jurídico especializado em direito brasileiro.

REGRAS:
1. Responda SOMENTE com base nas fontes fornecidas.
2. Cite as fontes como [FONTE X].
3. Se não houver contexto suficiente, diga isso.
4. Nunca invente artigos, leis ou jurisprudência.
"""


def generate_answer(query: str, context_chunks: list[dict]) -> str:
    if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY == "sua_chave_aqui":
        return _fallback_answer(context_chunks)

    context = format_context(context_chunks)

    user_message = f"""CONTEXTO:
{context}

PERGUNTA:
{query}
"""

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    message = client.messages.create(
        model=LLM_MODEL,
        max_tokens=LLM_MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_message}
        ],
    )

    return message.content[0].text


def _fallback_answer(context_chunks: list[dict]) -> str:
    lines = ["API não configurada. Trechos mais relevantes:\n"]

    for i, chunk in enumerate(context_chunks[:3], 1):
        lines.append(f"Trecho {i} — {chunk['citation']}")
        lines.append(chunk["text"][:500])
        lines.append("")

    return "\n".join(lines)
