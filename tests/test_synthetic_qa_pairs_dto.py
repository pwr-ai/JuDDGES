import unittest

from juddges.data.models import LangCode, SyntheticQAPairs
from juddges.exception import (
    GeneratedQAPairsEmptyError,
    GeneratedQAPairsLenghtMismatchError,
    GeneratedQAPairsNotUniqueError,
    LanguageMismatchError,
)


class TestSyntheticQAPairs(unittest.TestCase):
    def test_len(self):
        scenarios = [
            {"q": [], "a": [], "expected": 0},
            {"q": ["1"], "a": [], "expected": 1},
            {"q": ["1"], "a": ["2"], "expected": 1},
            {"q": ["1", "2"], "a": ["1", "2"], "expected": 2},
            {"q": ["1"], "a": ["1", "2"], "expected": 1},
        ]
        for scenario in scenarios:
            dto = SyntheticQAPairs(questions=scenario["q"], answers=scenario["a"])
            self.assertEqual(len(dto), scenario["expected"])

    def test_raises_on_empty(self):
        scenarios = [
            {"q": [], "a": []},
            {"q": [], "a": [""]},
            {"q": [""], "a": []},
        ]
        for scenario in scenarios:
            dto = SyntheticQAPairs(questions=scenario["q"], answers=scenario["a"])
            with self.assertRaises(GeneratedQAPairsEmptyError):
                dto.test_empty()

    def test_raises_on_length_mismatch(self):
        scenarios = [
            {"q": ["1"], "a": []},
            {"q": [], "a": ["1"]},
            {"q": ["1"], "a": ["1", "2"]},
            {"q": ["1", "2"], "a": ["1"]},
        ]
        for scenario in scenarios:
            dto = SyntheticQAPairs(questions=scenario["q"], answers=scenario["a"])
            with self.assertRaises(GeneratedQAPairsLenghtMismatchError):
                dto.test_equal_length()

    def test_raises_on_not_unique(self):
        dto = SyntheticQAPairs(questions=["1", "1"], answers=["1", "2"])
        with self.assertRaises(GeneratedQAPairsNotUniqueError):
            dto.test_unique_questions()

    def test_raises_on_language_mismatch(self):
        scenarios = [
            {
                "q": ["Jakie podstawowe prawo stosuje sąd w tej sprawie?"],
                "a": ["The court applies the Code of Civil Procedure."],
                "expected_lang": LangCode.POLISH,
            },
            {
                "q": ["What basic law does the court apply in this case?"],
                "a": ["Sąd stosuje Kodeks postępowania cywilnego."],
                "expected_lang": LangCode.POLISH,
            },
            {
                "q": ["What is the name of the accused?"],
                "a": ["Jędrzej Brzęczyszczykiewicz"],
                "expected_lang": LangCode.ENGLISH,
            },
            {
                "q": ["Jakie jest imię i nazwisko oskarżonego?"],
                "a": ["Helga Müller-Fäßler"],
                "expected_lang": LangCode.POLISH,
            },
        ]
        for scenario in scenarios:
            dto = SyntheticQAPairs(questions=scenario["q"], answers=scenario["a"])
            with self.assertRaises(LanguageMismatchError):
                dto.test_language(lang=scenario["expected_lang"])
