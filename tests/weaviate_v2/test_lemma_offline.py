"""Offline tests of the Polish lemmatizer evaluated for a `chunk_text_lemma` BM25 field.

The backends are optional dependencies (`morfeusz2`, `simplemma`); a test of a backend that is
not installed is skipped.
"""

import importlib.util

import pytest

from juddges.data.weaviate_v2.lemma import Lemmatizer, lemmatizer_version, tokenize

_MODULES = {"simplemma": "simplemma", "morfeusz_all": "morfeusz2"}
BACKENDS = [
    pytest.param(
        backend,
        marks=pytest.mark.skipif(
            importlib.util.find_spec(module) is None, reason=f"{module} not installed"
        ),
    )
    for backend, module in _MODULES.items()
]


def test_tokenize_matches_weaviate_word_tokenization():
    assert tokenize("Art. 385(1) k.c.; umowy-kredytu ŻÓŁW_x") == [
        "art",
        "385",
        "1",
        "k",
        "c",
        "umowy",
        "kredytu",
        "żółw",
        "x",
    ]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "forms",
    [
        ("umowa kredytowa", "umowy kredytowej", "umów kredytowych"),
        ("zachowek", "zachowku", "zachowkiem"),
        ("kredyt frankowy", "kredytu frankowego", "kredyty frankowe"),
    ],
)
def test_inflected_forms_share_lemmas(backend, forms):
    lemmatize = Lemmatizer(backend)
    sets = [set(lemmatize(f).split()) for f in forms]
    common = set.intersection(*sets)
    assert len(common) >= len(forms[0].split()), (forms, sets)


@pytest.mark.parametrize("backend", BACKENDS)
def test_stopwords_dropped_as_surface_forms_and_as_lemmas(backend):
    lemmatize = Lemmatizer(backend)
    out = lemmatize("Sąd uznał, że umowa jest nieważna, ponieważ była sprzeczna z ustawą")
    words = out.split()
    for stop in ("że", "jest", "ponieważ", "z", "być"):
        assert stop not in words
    assert "umowa" in words


@pytest.mark.parametrize("backend", BACKENDS)
def test_empty_and_none(backend):
    lemmatize = Lemmatizer(backend)
    assert lemmatize(None) == ""
    assert lemmatize("  ,;  ") == ""


@pytest.mark.parametrize("backend", BACKENDS)
def test_numbers_and_unknown_words_are_kept(backend):
    lemmatize = Lemmatizer(backend)
    assert lemmatize("art 385 qwxzy").split()[-2:] == ["385", "qwxzy"]


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="unknown lemmatizer backend"):
        Lemmatizer("stemmer")


@pytest.mark.parametrize("backend", BACKENDS)
def test_version_names_backend_and_library(backend):
    assert lemmatizer_version(backend).startswith(f"{backend}:")
