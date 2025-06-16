from enum import Enum

from pydantic import BaseModel as PydanticModel
from pydantic import Field
from pydantic_core.core_schema import ModelField

from label_studio_toolkit.schemas.utils import SchemaUtilsMixin


# todo: check if this is needed
def generate_default_fallback_validate(original_field_validate):
    def default_fallback_validate(self, v, values, *, loc, cls=None):
        value, error = original_field_validate(self, v, values, loc=loc, cls=cls)
        if error and not self.required:
            value, error = self.default, None
        return value, error

    return default_fallback_validate


# todo: check if this is needed
class BaseModel(PydanticModel):
    @classmethod
    def parse_obj_with_default_fallback(cls, obj):
        original_field_validate = ModelField.validate
        try:
            ModelField.validate = generate_default_fallback_validate(original_field_validate)
            return cls.parse_obj(obj)
        finally:
            ModelField.validate = original_field_validate

    @classmethod
    def model_validate(cls, obj, *args, **kwargs):
        if isinstance(obj, dict):
            for field_name, field in cls.__annotations__.items():
                if field_name in obj and obj[field_name] is not None:
                    origin = getattr(field, "__origin__", None)
                    if origin is list and isinstance(obj[field_name], str):
                        obj[field_name] = [obj[field_name]]

        return super().model_validate(obj, *args, **kwargs)


class TakNie(str, Enum):
    TAK = "Tak"
    NIE = "Nie"


class SprawaFrankowiczow(str, Enum):
    TAK = "Tak"
    NIE = "Nie"
    BADANIE_WZORCA = "Badanie wzorca umownego"


class TypSadu(str, Enum):
    SAD_REJONOWY = "Sąd Rejonowy"
    SAD_OKREGOWY = "Sąd Okręgowy"
    SAD_FRANKOWY = "Sąd Frankowy"
    SAD_OCHRONY_KONKURENCJI = "Sąd Ochrony Konkurencji i Konsumentów"


class InstancjaSadu(str, Enum):
    SAD_I_INSTANCJI = "Sąd I instancji"
    SAD_ODWOLAWCZY = "Sąd odwoławczy"


class RodzajRoszczenia(str, Enum):
    USTALENIE_ISTNIENIA = "O ustalenie istnienia/nieistnienia stosunku prawnego"
    UKSZTALTOWANIE = "O ukształtowanie stosunku prawnego"
    ZAPLATA = "O zapłatę"
    ROSZCZENIA_DODATKOWE = "Roszczenia dodatkowe"
    WZORZEC_UMOWNY = "Dotyczy wzorca umownego"


class TypModyfikacji(str, Enum):
    RODZAJ_ROSZCZENIA = "Rodzaj roszczenia"
    KWOTY_ROSZCZENIA = "Kwoty roszczenia"


class StatusKredytobiorcy(str, Enum):
    KONSUMENT = "Konsument"
    PRZEDSIEBIORCA = "Przedsiębiorca"
    NIE_SPRAWDZANO = "Nie sprawdzano"


class TypWspoluczestnictwa(str, Enum):
    MALZENSTWO = "Małżeństwo"
    KONKUBINAT = "Konkubinat"
    SPADKODAWCY = "Spadkodawcy"
    INNY_RODZAJ = "Inny rodzaj"


class StronaUmowy(str, Enum):
    STRONA_UMOWY = "Strona umowy"
    NASTEPCA_PRAWNY = "Następca prawny"


class UmowaKredytowa(str, Enum):
    BEZPOSREDNIO_W_BANKU = "Zawarta bezpośrednio w banku"
    Z_UDZIALEM_POSREDNIKA = "Z udziałem pośrednika"


class WalutaSplaty(str, Enum):
    PLN = "PLN"
    CHF = "CHF"


class PrzedmiotAneksu(str, Enum):
    ZMIANA_WALUTY_SPLATY = "Zmiana waluty spłaty"
    INNE_KWESTIE = "Inne kwestie"


class TypRozstrzygniecia(str, Enum):
    UWZGLEDNIENIE_CAŁOSCI = "Uwzględnienie powództwa w całości"
    UWZGLEDNIENIE_CZESCI = "Uwzględnienie powództwa w części"
    ODDALENIE_POWODZTWA = "Oddalenie powództwa"
    ODDALENIE_APELACJI = "Oddalenie apelacji"
    ZMIANA_WYROKU = "Zmiana wyroku"
    PRZEKAZANIE_DO_PONOWNEGO_ROZPOZNANIA = "Przekazanie do ponownego rozpoznania"


class SesjaSadowa(str, Enum):
    ROZPRAWA = "Rozprawa"
    POSIEDZENIE_NIEJAWNE = "Posiedzenie niejawne"


class Dowody(str, Enum):
    PRZESLUCHANIE_STRON = "Przesłuchanie stron"
    PRZESLUCHANIE_SWIADKOW = "Przesłuchanie świadków"
    DOWOD_Z_OPINII_BIEGLEGO = "Dowód z opinii biegłego"
    DOWOD_Z_DOKUMENTOW = "Dowód z dokumentów"


class TeoriaPrawna(str, Enum):
    TEORIA_DWOCH_KONDYKCJI = "Teoria dwóch kondykcji"
    TEORIA_SALDA = "Teoria salda"


class DataRozpoczeciaOdsetek(str, Enum):
    OD_DNIA_WEZWANIA = "Od dnia wezwania do zapłaty"
    OD_DATY_WYTOCZENIA = "Od daty wytoczenia powództwa"
    OD_DATY_WYROKU = "Od daty wydania wyroku"
    OD_INNEJ_DATY = "Od innej daty"


class WynikSprawy(str, Enum):
    WYGRANA_KREDYTOBIORCY = "Wygrana kredytobiorcy"
    WYGRANA_BANKU = "Wygrana banku"
    CZESCIOWE_UWZGLEDNIENIE = "Częściowe uwzględnienie roszczeń obu stron"


class Zarzut(str, Enum):
    TAK = "Tak"
    NIE = "Nie"
    NIE_PODNIOSIONO = "Nie podniesiono takiego zarzutu"


class Zabezpieczenie(str, Enum):
    TAK = "Tak"
    NIE = "Nie"
    NIE_BYLO_WNIOSKU = "Nie było wniosku o zabezpieczenie"


class SwissFrancJudgmentAnnotation(BaseModel, SchemaUtilsMixin):
    sprawa_frankowiczow: SprawaFrankowiczow | None = Field(
        None, description="Czy sprawa dotyczy kredytu frankowego (CHF)?"
    )
    apelacja: str | None = Field(
        None,
        description="Określenie apelacji, w której znajduje się sąd rozpoznający sprawę. 'Zanonimizowano' jeśli niemożliwe do ustalenia bo zanonimizowano.",
    )
    data_wyroku: str | None = Field(None, description="Data wydania wyroku w formacie YYYY-MM-DD")
    typ_sadu: TypSadu | None = Field(None, description="Typ sądu rozpoznającego sprawę")
    instancja_sadu: InstancjaSadu | None = Field(
        None, description="Czy sąd jest I instancji czy odwoławczy"
    )
    podstawa_prawna: str | None = Field(None, description="Podstawa prawna roszczenia")
    podstawa_prawna_podana: TakNie | None = Field(
        None, description="Czy powód podał podstawę prawną?"
    )
    rodzaj_roszczenia: RodzajRoszczenia | None = Field(None, description="Rodzaj roszczenia")
    modyfikacje_powodztwa: TakNie | None = Field(None, description="Czy były modyfikacje powództwa")
    typ_modyfikacji: TypModyfikacji | None = Field(None, description="Typ modyfikacji powództwa")
    status_kredytobiorcy: StatusKredytobiorcy | None = Field(
        None, description="Status kredytobiorcy"
    )
    wspoluczestnictwo_powodowe: TakNie | None = Field(
        None, description="Czy współuczestnictwo po stronie powodowej"
    )
    wspoluczestnictwo_pozwanego: TakNie | None = Field(
        None, description="Czy współuczestnictwo po stronie pozwanej"
    )
    typ_wspoluczestnictwa: TypWspoluczestnictwa | None = Field(
        None, description="Rodzaj współuczestnictwa"
    )
    strony_umowy: StronaUmowy | None = Field(
        None, description="Czy powód był stroną umowy czy następcą prawnym"
    )
    wczesniejsze_skargi_do_rzecznika: TakNie | None = Field(
        None, description="Czy były uprzednie skargi do rzecznika finansowego"
    )
    umowa_kredytowa: UmowaKredytowa | None = Field(
        None, description="Czy umowa kredytowa zawierana bezpośrednio w banku czy z pośrednikiem"
    )
    klauzula_niedozwolona: TakNie | None = Field(
        None, description="Istnienie klauzuli niedozwolonej w umowie"
    )
    wpisana_do_rejestru_uokik: TakNie | None = Field(
        None, description="Czy klauzula wpisana do rejestru klauzul niedozwolonych UOKiK"
    )
    waluta_splaty: WalutaSplaty | None = Field(None, description="Waluta spłaty")
    aneks_do_umowy: TakNie | None = Field(None, description="Czy był aneks do umowy")
    data_aneksu: str | None = Field(None, description="Data aneksu w formacie YYYY-MM-DD")
    przedmiot_aneksu: PrzedmiotAneksu | None = Field(None, description="Czego dotyczył aneks")
    status_splaty_kredytu: TakNie | None = Field(
        None, description="Czy kredyt był spłacony, w tym w trakcie procesu"
    )
    rozstrzygniecie_sadu: str | None = Field(None, description="Rozstrzygnięcie")
    typ_rozstrzygniecia: TypRozstrzygniecia | None = Field(None, description="Typ rozstrzygnięcia")
    sesja_sadowa: SesjaSadowa | None = Field(
        None, description="Czy wyrok wydano na rozprawie czy na posiedzeniu niejawnym"
    )
    dowody: list[Dowody] | None = Field(None, description="Jakie dowody zostały przeprowadzone")
    oswiadczenie_niewaznosci: TakNie | None = Field(
        None, description="Czy odbierano oświadczenie powoda o skutkach nieważności umowy"
    )
    odwolanie_do_sn: TakNie | None = Field(None, description="Czy odwołano się do orzecznictwa SN")
    odwolanie_do_tsue: TakNie | None = Field(
        None, description="Czy odwołano się do orzecznictwa TSUE"
    )
    teoria_prawna: TeoriaPrawna | None = Field(
        None, description="Teoria prawna, na której oparto wyrok"
    )
    zarzut_zatrzymania: Zarzut | None = Field(
        None, description="Czy uwzględniono zarzut zatrzymania"
    )
    zarzut_potracenia: Zarzut | None = Field(None, description="Czy uwzględniono zarzut potrącenia")
    odsetki_ustawowe: TakNie | None = Field(None, description="Czy uwzględniono odsetki ustawowe")
    data_rozpoczecia_odsetek: DataRozpoczeciaOdsetek | None = Field(
        None, description="Data rozpoczęcia naliczania odsetek"
    )
    koszty_postepowania: TakNie | None = Field(
        None, description="Czy zasądzono zwrot kosztów postępowania"
    )
    beneficjent_kosztow: str | None = Field(
        None, description="Na rzecz której strony zasądzono zwrot kosztów"
    )
    zabezpieczenie_udzielone: Zabezpieczenie | None = Field(
        None, description="Czy udzielono zabezpieczenia"
    )
    rodzaj_zabezpieczenia: str | None = Field(None, description="Rodzaj zabezpieczenia")
    zabezpieczenie_pierwsza_instancja: TakNie | None = Field(
        None, description="Czy zabezpieczenia udzielił sąd I instancji"
    )
    czas_trwania_sprawy: str | None = Field(
        None, description="Czas rozpoznania sprawy – od złożenia pozwu do wydania wyroku"
    )
    wynik_sprawy: WynikSprawy | None = Field(
        None, description="Ocena, czy bank czy kredytobiorca wygrał sprawę"
    )
    szczegoly_wyniku_sprawy: str | None = Field(
        None, description="Szczegóły dotyczące wyniku sprawy"
    )
