from enum import Enum

from pydantic import BaseModel, Field


class TakNie(str, Enum):
    TAK = "Tak"
    NIE = "Nie"


class PodstawaPrawna(str, Enum):
    ART_23_KC = "23 KC"
    ART_24_KC = "24 KC"
    INNE = "Inne"


class Zadanie(str, Enum):
    ZANIECHANIE = "Zaniechania działania"
    USUNIECIE_SKUTKOW = "Żądanie ażeby pozwany dopełnił czynności potrzebnych do usunięcia skutków naruszenia. W szczególności ażeby złożyła oświadczenie odpowiedniej treści i w odpowiedniej formie"
    ZADOSCUCZYNIENIE = "Zadośćuczynienia pieniężnego"
    ZAPLATA_NA_CEL_SPOLECZNY = "Zapłaty odpowiedniej sumy pieniężnej na wskazany cel społeczny"


class Dowod(str, Enum):
    PRZESLUCHANIE_SWIADKOW = "Przesłuchanie świadków"
    PRZESLUCHANIE_STRON = "Przesłuchanie stron"
    DOKUMENTY = "Dokumenty"
    SCREENSHOT = "Zrzut ekranu/print screen"


class PortalInternetowy(str, Enum):
    FACEBOOK = "Facebook"
    INSTAGRAM = "Instagram"
    TIKTOK = "TikTok"
    LINKEDIN = "Linkedin"
    FORUM = "Forum dyskusyjne"
    INNE = "Inne miejsce w internecie"


class DobroOsobiste(str, Enum):
    ZDROWIE = "Zdrowie"
    WOLNOSC = "Wolność"
    CZESC = "Cześć"
    SWOBODA_SUMIENIA = "Swoboda sumienia"
    NAZWISKO = "Nazwisko lub pseudonim"
    WIZERUNEK = "Wizerunek"
    TAJEMNICA_KORESPONDENCJI = "Tajemnica korespondencji"
    NIETYKALNOSC_MIESZKANIA = "Nietykalność mieszkania"
    TWORCZOSC = "Twórczość naukowa, artystyczna, wynalazcza i racjonalizatorska"
    INNE = "Inne"


class MiejsceNaruszenia(str, Enum):
    PUBLICZNE = "Miejsce publiczne"
    PRYWATNE = "Miejsce prywatne"
    PRACY = "Miejsce pracy"
    INTERNET = "W internecie"


class RodzajNaruszajacego(str, Enum):
    INSTYTUCJA_FIRMA = "Instytucja/firma"
    OSOBA_PRYWATNA = "Osoba prywatna"


class PersonalRightsAnnotation(BaseModel):
    naruszenie_dobr_osobistych: TakNie = Field(
        ..., description="Czy sprawa dotyczy naruszenia dóbr osobistych"
    )
    podstawa_prawna: list[PodstawaPrawna] | None = Field(
        None, description="Podstawa prawna – artykuły Kodeksu Cywilnego"
    )
    inne_podstawy_prawne: list[str] | None = Field(
        None,
        description="Inne podstawy prawne, jeśli wybrano opcję 'Inne'. Każdy artykuł wprowadź oddzielnie",
    )
    rodzaj_naruszajacego: RodzajNaruszajacego | None = Field(
        None, description="Kto naruszył dobra osobiste - czy instytucja/firma czy osoba prywatna"
    )
    rodzaj_dobra_osobistego: list[DobroOsobiste] | None = Field(
        None, description="Rodzaj dobra osobistego, które zostało naruszone"
    )
    opis_naruszenia: str | None = Field(
        None, description="Opis naruszenia dobra osobistego np. zacytowanie"
    )
    miejsce_naruszenia: MiejsceNaruszenia | None = Field(
        None, description="Jakie było miejsce naruszenia"
    )
    naruszenie_media_spolecznosciowe: bool | None = Field(
        None,
        description="Czy naruszenie zaszło na mediach społecznościowych. Jeśli niedotyczy pozostaw puste",
    )
    portale_spolecznosciowe: list[PortalInternetowy] | None = Field(
        None,
        description="Portale społecznościowe lub internetowe, gdzie doszło do naruszenia. Jeśli niedotyczy pozostaw puste",
    )
    skala_naruszenia: int | None = Field(
        None,
        description="Skala 1-5 oceny stopnia naruszenia dóbr osobistych. 0 - brak naruszenia, 1 - bardzo lekkie, 2 - lekkie, 3 - średnie, 4 - mocne, 5 - bardzo mocne",
        ge=0,
        le=5,
    )
    zadania: list[Zadanie] | None = Field(None, description="Rodzaje żądań występujące w sprawie")
    dowody: list[Dowod] | None = Field(None, description="Rodzaje dowodów przedstawione w sprawie")
