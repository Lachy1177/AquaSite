import json
import re
import logging

logger = logging.getLogger(__name__)

_FALLBACK_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were", "be",
    "been", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "that", "this", "these",
    "those", "it", "its", "which", "who", "what", "where", "when", "how",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "they",
    "their", "there", "here", "not", "no", "all", "some", "any", "so",
    "about", "as", "than", "then", "image", "images", "picture", "pictures",
    "photo", "photos", "frame", "frames", "show", "shows", "find", "get",
    "give", "contains", "containing", "contain", "display", "want", "need",
    "please", "something", "like", "looking", "underwater", "scene",
}


def _fallback_split(query: str) -> list[str]:
    tokens = re.split(r"[\s,]+", query.lower())
    return [t for t in tokens if t and t not in _FALLBACK_STOP_WORDS and len(t) > 1]


def extract_search_terms(query: str, taxonomy_labels: list[str]) -> list[str]:
    """
    Use Claude Haiku to extract biological/ecological search keywords from a
    natural-language query, mapped against the known taxonomy label vocabulary.
    Falls back to simple stop-word splitting if no API key is set or the call fails.
    """
    from django.conf import settings

    api_key = getattr(settings, "ANTHROPIC_API_KEY", "")
    if not api_key:
        return _fallback_split(query)

    label_list = ", ".join(taxonomy_labels[:200])  # cap to avoid huge prompts

    prompt = (
        f"You are a keyword extractor for an underwater marine image search system.\n"
        f"Known taxonomy labels in the database: {label_list}\n\n"
        f"User search query: \"{query}\"\n\n"
        f"Extract only the biological or ecological search keywords from the query "
        f"that match or are semantically close to the known labels. "
        f"Return ONLY a JSON array of lowercase strings, nothing else. "
        f"Example output: [\"coral\", \"fish\", \"sand\"]"
    )

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=128,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
        terms = json.loads(raw)
        if isinstance(terms, list) and all(isinstance(t, str) for t in terms):
            return [t.lower().strip() for t in terms if t.strip()]
    except Exception as exc:
        logger.warning("Claude keyword extraction failed, using fallback: %s", exc)

    return _fallback_split(query)
