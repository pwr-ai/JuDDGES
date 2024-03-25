import re
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
        return ["num_pages", "vol_number", "vol_type", "text"]

    def parse(self, document: str) -> dict[str, Any]:
        et = ElementTree.fromstring(document)

        xblock_elements = et.findall("xBlock")
        assert len(xblock_elements) == 1, "There should be only one xBlock element"
        content_root, *_ = xblock_elements

        return {
            "num_pages": int(et.attrib["xToPage"]),
            "vol_number": int(et.attrib["xVolNmbr"]),
            "vol_type": et.attrib["xVolType"],
            "text": self.extract_text(content_root),
        }

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
