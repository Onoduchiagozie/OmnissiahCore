"""Local, deterministic query normalization; no LLM or network calls."""
from dataclasses import asdict, dataclass
import re
from typing import Mapping


@dataclass(frozen=True)
class QueryPlan:
    original_query: str
    normalized_query: str
    dense_query: str
    sparse_query: str
    intent: str
    changes: tuple[str, ...]
    answer_guidance: str

    def to_dict(self) -> dict:
        return asdict(self)


class QueryRewriter:
    """Expand known names and comparison syntax without supplying lore facts.

    Deliberately conservative: unknown names are preserved, no book is selected,
    and a comparison is not rewritten into a duel unless the user said that.
    The alias map is injectable so additions do not require retriever edits.
    """

    DEFAULT_ALIASES = {
        'ferrus': 'Ferrus Manus',
        'fulgrimm': 'Fulgrim',
        'istvan': 'Isstvan',
        'drop site': 'Dropsite',
        'kakaponi': 'Kakophoni',
        'sharokin': 'Sharrowkyn',
        'slaneshi': 'Slaaneshi',
        'sea of sauls': 'Sea of Souls',
        '3rd legion': "Emperor's Children",
        '5th legion': 'White Scars',
    }
    SEARCH_STOPWORDS = frozenset({
        'a', 'about', 'an', 'and', 'are', 'as', 'at', 'be', 'between', 'by', 'can',
        'compare', 'could', 'describe', 'did', 'do', 'does', 'during', 'explain',
        'for', 'from', 'give', 'had', 'has', 'have', 'how', 'i', 'in', 'is',
        'it', 'me', 'of', 'on', 'please', 'say', 'tell', 'than', 'that', 'the',
        'their', 'these', 'they', 'this', 'to', 'us', 'versus', 'vs', 'was',
        'were', 'what', 'when', 'where', 'which', 'who', 'why', 'with', 'would',
        'you',
    })

    def __init__(self, aliases: Mapping[str, str] | None = None):
        self.aliases = dict(self.DEFAULT_ALIASES if aliases is None else aliases)

    def rewrite(self, query: str) -> QueryPlan:
        normalized = re.sub(r'\s+', ' ', query).strip()
        changes = []
        for alias, canonical in sorted(self.aliases.items(), key=lambda item: len(item[0]), reverse=True):
            suffix = r'(?!\s+Manus\b)' if alias.casefold() == 'ferrus' else ''
            pattern = r'(?<!\w)' + re.escape(alias) + r'(?!\w)' + suffix
            replaced = re.sub(pattern, canonical, normalized, flags=re.IGNORECASE)
            if replaced != normalized:
                changes.append(alias+' -> '+canonical)
                normalized = replaced
        # Numeral normalization is tied to a named location, never applied to
        # every occurrence of "five" or "v" in an arbitrary question.
        for words, numeral in ((r'five|5|v', 'V'), (r'three|3|iii', 'III')):
            replaced = re.sub(r'\bIsstvan\s+(?:'+words+r')\b', 'Isstvan '+numeral, normalized, flags=re.IGNORECASE)
            if replaced != normalized:
                changes.append('Isstvan numeral -> '+numeral)
                normalized = replaced
        versus = bool(re.search(r'\b(?:vs|versus)\b\.?', normalized, re.IGNORECASE))
        if versus or re.search(r'\b(?:compare|difference between)\b', normalized, re.IGNORECASE):
            intent = 'comparison'
        elif re.search(r'\bwhy\b', normalized, re.IGNORECASE):
            intent = 'explanation'
        else:
            intent = 'lookup'
        if versus:
            body = re.sub(r'\b(?:vs|versus)\b\.?', 'and', normalized, flags=re.IGNORECASE)
            dense = 'Compare '+body.rstrip('?.! ')+'.'
            changes.append('vs -> comparison question')
        else:
            dense = normalized
        words = re.findall(r"[\w']+", normalized)
        keywords = [word for word in words if word.casefold() not in self.SEARCH_STOPWORDS]
        sparse = ' '.join(keywords) or normalized
        guidance = {
            'comparison': (
                'Answer as a comparison of the named subjects using the supplied evidence. '
                'Explain supported similarities, differences, and relevant relationships. '
                'Do not assume the user meant only a duel or invent a hypothetical winner.'
            ),
            'explanation': (
                'Answer the why-question using causes supported by the supplied evidence. '
                'Distinguish stated motives from inference; do not invent missing causes.'
            ),
            'lookup': (
                'Answer the original question directly using the supplied evidence. '
                'For a broad question give a coherent overview; respect any scope the user supplied.'
            ),
        }[intent]
        return QueryPlan(query, normalized, dense, sparse, intent, tuple(changes), guidance)
