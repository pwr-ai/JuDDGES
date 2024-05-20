import re
from collections import defaultdict
from typing import Any, Generator
from xml.etree import ElementTree
from xml.etree.ElementTree import Element

from juddges.preprocessing.parser_base import DocParserBase

MULTIPLE_NEWLINES = re.compile(r"(\n\s*)+\n+")


class SimplePlJudgementsParser(DocParserBase):
    """The simplest parser for the simple XML format used by the Polish courts.

    It extracts the text from XML file, without adhering to any specific structure.

    """

    @property
    def schema(self) -> list[str]:
        return ["text_legal_bases", "num_pages", "vol_number", "vol_type", "text"]

    def parse(self, document: str) -> dict[str, Any]:
        et = ElementTree.fromstring(document)

        xblock_elements = et.findall("xBlock")
        assert len(xblock_elements) == 1, "There should be only one xBlock element"
        content_root, *_ = xblock_elements

        return {
            "text_legal_bases": self.extract_legal_bases(et),
            "num_pages": int(et.attrib["xToPage"]),
            "vol_number": int(et.attrib["xVolNmbr"]),
            "vol_type": et.attrib["xVolType"],
            "text": self.extract_text(content_root),
        }

    def extract_legal_bases(self, element: Element) -> list[dict[str, str]]:
        """Extracts unique legal bases from XML (contains text from judgement as opposed to API)."""
        legal_bases = [
            {
                "text": elem.text or "",
                "art": elem.attrib["xArt"].strip(),
                "isap_id": elem.attrib["xIsapId"].strip(),
                "title": elem.attrib["xTitle"].strip(),
                "address": elem.attrib["xAddress"].strip(),
            }
            for elem in element.findall(".//xLexLink")
        ]

        legal_bases = [lb for lb in legal_bases if (lb["text"].strip() and lb["art"])]
        return self._get_longest_text_legal_bases(legal_bases)

    @staticmethod
    def _get_longest_text_legal_bases(legal_bases: list[dict[str, str]]) -> list[dict[str, str]]:
        """Extracts the longest legal base from the list of legal bases."""
        legal_bases_dict = defaultdict(list)
        for lb in legal_bases:
            legal_bases_dict[lb["isap_id"]].append(lb)

        longest_text_legal_bases = []
        for isap_id, lbs in legal_bases_dict.items():
            longest_legal_base = max(lbs, key=lambda lb: len(lb["text"]))
            longest_text_legal_bases.append(longest_legal_base)

        return longest_text_legal_bases

    @staticmethod
    def extract_text(element: Element) -> str:
        text = ""
        for elem_txt in element.itertext():
            if elem_txt is None:
                continue
            if txt := elem_txt.strip(" "):
                text += txt

        text = re.sub(MULTIPLE_NEWLINES, "\n\n", text).strip()

        return text


def itertext(element: Element, prefix: str = "") -> Generator[str, None, None]:
    """Extension of the Element.itertext method to handle special tags in pl court XML."""
    tag = element.tag
    if not isinstance(tag, str) and tag is not None:
        return

    t: str | None
    match (tag, element.attrib):
        case ("xName", {"xSffx": suffix}):
            element.tail = element.tail.strip() if element.tail else None
            t = f"{element.text}{suffix} "
        case ("xEnum", _):
            bullet_elem = element.find("xBullet")
            if bullet_elem:
                prefix = bullet_elem.text or ""
                element.remove(bullet_elem)
            t = ""
        case ("xEnumElem", _):
            t = prefix
        case _:
            t = element.text

    if t:
        yield t

    for e in element:
        yield from itertext(e, prefix)
        t = e.tail

        if t:
            yield t
