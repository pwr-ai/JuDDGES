"""Polish lemmatization for BM25 (Weaviate has no Polish analyzer).

`chunk_text` is indexed with `word` tokenization and no stemming, so "umowa kredytowa" and
"umowy kredytowej" share no term. A `Lemmatizer` turns a text into space-separated lowercase
lemmas with Polish stopwords removed. Indexed as a lemma field, with the same function applied
to the query, both sides meet on the lemma. Production does not index it: the evaluation in
docs/explanation/polish-lemmatization-bm25.md did not clear its gate. The experiment fields are
`chunk_text_lemma_<backend>` on `ExpChunksLemma`.

Tokens are split as Weaviate's `word` tokenizer does (runs of letters and digits, lowercased).
The backends are optional dependencies (`simplemma==2.0.0`, `morfeusz2==1.99.15`):
- `simplemma`: dictionary lookup, one lemma per token.
- `morfeusz_all`: every distinct Morfeusz2 (SGJP) lemma of a token; without a tagger there is
  no good way to choose one (e.g. "kary" -> kary / kar / kara), so all are indexed.

The output depends on the backend and its dictionary version, so the indexer and the query side
must use the same backend and the same library version (`lemmatizer_version`).
"""

import re
from collections.abc import Callable, Iterable
from functools import lru_cache

from juddges.data.documents_weaviate_db import POLISH_STOPWORDS

_TOKEN = re.compile(r"[^\W_]+")
_STOPWORDS = frozenset(POLISH_STOPWORDS)
CACHE_SIZE = 500_000
DEFAULT_BACKEND = "simplemma"
BACKENDS = ("simplemma", "morfeusz_all")


def tokenize(text: str) -> list[str]:
    """Lowercased runs of letters and digits, as Weaviate's `word` tokenization."""
    return _TOKEN.findall(text.lower())


def _clean_lemma(lemma: str) -> str:
    """SGJP marks homonyms as `lemma:s1` / `lemma:v2`; keep the base form."""
    return lemma.split(":", 1)[0].lower()


@lru_cache(maxsize=1)
def _morfeusz():
    import morfeusz2

    return morfeusz2.Morfeusz(generate=False, expand_tags=False)


def _morfeusz_lemmas(token: str) -> tuple[str, ...]:
    """Distinct lemmas of a token's first segment, in dictionary order.

    Morfeusz may split a token into segments (e.g. "zrobiłem" -> "zrobił" + "em"); the
    agglutinate and other secondary segments are dropped. Unknown words come back as the token.
    """
    lemmas: dict[str, None] = {}
    for start, _end, (_form, lemma, tag, _names, _quals) in _morfeusz().analyse(token):
        if start == 0 and not tag.startswith("aglt"):
            lemmas.setdefault(_clean_lemma(lemma), None)
    lemmas.pop("", None)
    return tuple(lemmas) or (token,)


def _simplemma(token: str) -> tuple[str, ...]:
    import simplemma

    return (simplemma.lemmatize(token, lang="pl").lower(),)


_BACKEND_FUNCS: dict[str, Callable[[str], tuple[str, ...]]] = {
    "morfeusz_all": _morfeusz_lemmas,
    "simplemma": _simplemma,
}


class Lemmatizer:
    """Token -> lemmas with a per-instance LRU cache; one instance per process."""

    def __init__(self, backend: str = DEFAULT_BACKEND, cache_size: int = CACHE_SIZE) -> None:
        if backend not in _BACKEND_FUNCS:
            raise ValueError(f"unknown lemmatizer backend {backend!r}, expected one of {BACKENDS}")
        self.backend = backend
        self._lemmas = lru_cache(maxsize=cache_size)(_BACKEND_FUNCS[backend])
        # stopwords as surface forms and as lemmas (e.g. "będą" -> "być" also drops "byłoby")
        self.stop_lemmas = _STOPWORDS | {
            lemma for word in _STOPWORDS for lemma in self._lemmas(word)
        }

    def lemmas(self, tokens: Iterable[str]) -> list[str]:
        out: list[str] = []
        for token in tokens:
            if token in _STOPWORDS:
                continue
            out.extend(lemma for lemma in self._lemmas(token) if lemma not in self.stop_lemmas)
        return out

    def __call__(self, text: str | None) -> str:
        """Space-separated lemmas of `text`, stopwords removed."""
        return " ".join(self.lemmas(tokenize(text or "")))


def lemmatizer_version(backend: str = DEFAULT_BACKEND) -> str:
    """Backend plus library/dictionary version, stored with an index built from it."""
    if backend.startswith("morfeusz"):
        import morfeusz2

        return f"{backend}:morfeusz2-{morfeusz2.__version__}:{_morfeusz().dict_id()}"
    import simplemma

    return f"{backend}:simplemma-{simplemma.__version__}"
