import re

_STOP_WORDS = {
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


def extract_search_terms(query: str, taxonomy_labels: list[str]) -> list[str]:
    tokens = re.split(r"[\s,]+", query.lower())
    return [t for t in tokens if t and t not in _STOP_WORDS and len(t) > 1]
