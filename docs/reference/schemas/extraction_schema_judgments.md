# Schemat ekstrakcji: Orzeczenia sądowe

> Wszystkie pola mają jawnie określony typ danych i metodę agregacji.
> Obiekty JSON rozbite na flat kolumny. Free text pola uzupełnione o enum klasyfikatory.
> Gotowy do implementacji w HuggingFace Datasets / Parquet / PostgreSQL.

---

## Profil agregowalności

```
87 pól łącznie:
  ENUM .............. 27 pól (31%)  → GROUP BY, COUNT, rozkłady
  BOOL .............. 17 pól (20%)  → proporcje, korelacje
  INT/FLOAT .......... 9 pól (10%)  → avg, median, histogram
  SCORE (1-5) ........ 6 pól ( 7%)  → rozkłady, korelacje
  DATE ............... 1 pole ( 1%)  → szeregi czasowe
  ENUM_ARRAY ......... 5 pól ( 6%)  → multi-label frequency
  KEYWORDS ........... 3 pola ( 3%)  → frequency, TF-IDF
  ID_REF ............. 6 pól ( 7%)  → citation networks
  FREE_TEXT .......... 13 pól (15%)  → embedding, NLP

  Natychmiastowa agregacja (bez NLP): 78%
```

---

## A. IDENTYFIKACJA (6 pól)

| # | Pole | Typ danych | Agregacja | Opis | Wartości dozwolone |
|---|------|-----------|-----------|------|--------------------|
| 1 | `document_number` | ID | — | Sygnatura akt, np. `I ACa 123/23` | — |
| 2 | `date_issued` | DATE | szeregi czasowe | Data wydania orzeczenia (ISO 8601) | — |
| 3 | `document_type` | ENUM | GROUP BY | Typ orzeczenia | `wyrok` / `postanowienie` / `uchwala` / `zarzadzenie` |
| 4 | `court_type` | ENUM | GROUP BY | Typ sądu | `sad_rejonowy` / `sad_okregowy` / `sad_apelacyjny` / `sad_najwyzszy` / `wsa` / `nsa` / `trybunal_konstytucyjny` |
| 5 | `court_name` | ENUM | GROUP BY | Nazwa sądu (z zamkniętej listy ~400 sądów) | Pełna nazwa, np. `Sąd Okręgowy w Warszawie` |
| 6 | `court_chamber` | ENUM | GROUP BY | Wydział/izba | `cywilny` / `karny` / `pracy` / `gospodarczy` / `rodzinny` / `ubezpieczen_spolecznych` / `finansowy` / `ogolnoadministracyjny` / `izba_cywilna` / `izba_karna` / `izba_pracy` / `izba_kontroli_nadzwyczajnej` / `inne` |

---

## B. KLASYFIKACJA MERYTORYCZNA — ENUM (21 pól)

Pola natychmiastowo agregowalne: pie chart, bar chart, cross-tabulation, chi-square, heatmap.

| # | Pole | Typ danych | Wartości dozwolone | Opis |
|---|------|-----------|--------------------|----- |
| 7 | `legal_domain` | ENUM | `cywilne` / `karne` / `administracyjne` / `pracy` / `ubezpieczen_spolecznych` / `gospodarcze` / `rodzinne` / `podatkowe` / `karne_skarbowe` / `wykroczeniowe` | Dziedzina prawa |
| 8 | `case_subject_category` | ENUM | `kredyt_frankowy` / `odszkodowanie_szkoda_osobowa` / `odszkodowanie_szkoda_majatkowa` / `zadoscuczynienie` / `umowa_sprzedaz` / `umowa_najem_dzierzawa` / `umowa_o_dzielo_zlecenie` / `prawo_pracy_zwolnienie` / `prawo_pracy_wynagrodzenie` / `prawo_pracy_dyskryminacja` / `spadki_zachowek` / `rozwod_alimenty` / `wlasnosc_nieruchomosci` / `sluzebnosc_zasiedzenie` / `spolki_odpowiedzialnosc` / `upadlosc_restrukturyzacja` / `zamowienia_publiczne` / `decyzja_administracyjna` / `podatek_vat` / `podatek_pit_cit` / `podatek_od_nieruchomosci` / `egzekucja` / `ochrona_konsumentow` / `dane_osobowe_rodo` / `prawo_budowlane` / `ochrona_srodowiska` / `wlasnosc_intelektualna` / `prawo_karne_narkotyki` / `prawo_karne_przeciwko_mieniu` / `prawo_karne_przeciwko_osobie` / `inne` | Główny temat merytoryczny sprawy |
| 9 | `decision_type` | ENUM | `uwzgledniono_w_calosci` / `uwzgledniono_w_czesci` / `oddalono` / `umorzono` / `uchylono` / `uchylono_i_przekazano` / `zmieniono` / `utrzymano_w_mocy` / `odrzucono` / `uniewinniono` / `skazano` / `warunkowo_umorzono` | Typ rozstrzygnięcia |
| 10 | `instance_number` | ENUM | `pierwsza` / `druga` / `kasacja` / `skarga_kasacyjna` / `wznowienie` / `skarga_nadzwyczajna` | Instancja |
| 11 | `party_power_asymmetry` | ENUM | `symetryczna` / `umiarkowana_asymetria` / `silna_asymetria` | Relacja siły stron |
| 12 | `plaintiff_type` | ENUM | `osoba_fizyczna` / `osoba_fizyczna_przedsiebiorca` / `spolka_kapitalowa` / `spolka_osobowa` / `organ_administracji` / `prokurator` / `rzecznik_konsumentow` / `fundacja_stowarzyszenie` / `jst` / `skarb_panstwa` / `spoldzielnia` / `zwiazek_zawodowy` / `inne` | Typ powoda/skarżącego/oskarżyciela |
| 13 | `defendant_type` | ENUM | (te same wartości co `plaintiff_type`) | Typ pozwanego/organu/oskarżonego |
| 14 | `panel_composition_type` | ENUM | `jednoosobowy` / `trojkowy` / `siedmioosobowy` / `pelny_sklad` / `izba` | Skład orzekający |
| 15 | `standard_of_review` | ENUM | `de_novo` / `ograniczona_kontrola` / `arbitralnosc` / `razace_naruszenie` | Standard kontroli |
| 16 | `primary_reasoning_method` | ENUM | `jezykowa` / `celowosciowa` / `systemowa` / `historyczna` / `prounijna` / `prokonstytucyjna` / `analogia` / `a_contrario` / `funkcjonalna` | Dominująca metoda wykładni |
| 17 | `remedy_primary` | ENUM | `odszkodowanie` / `zadoscuczynienie` / `wykonanie_umowy` / `powstrzymanie` / `ustalenie` / `uksztaltowanie` / `uchylenie_decyzji` / `kara_pozbawienia_wolnosci` / `kara_grzywny` / `kara_ograniczenia_wolnosci` / `srodek_karny` / `brak` / `inne` | Główny środek ochrony/kara |
| 18 | `judicial_creativity_type` | ENUM | `nowa_wykladnia` / `rozszerzenie_zasady` / `odejscie_od_linii` / `zawezenie_zasady` / `brak` | Typ kreatywności sędziowskiej |
| 19 | `dissent_strength` | ENUM | `brak` / `techniczna` / `czesciowa` / `fundamentalna` | Siła zdania odrębnego |
| 20 | `overruling_scope` | ENUM | `brak` / `expressis_verbis` / `implicitne` / `czesciowe` | Zakres uchylenia wcześniejszego orzeczenia |
| 21 | `ratio_scope` | ENUM | `waski` / `szeroki` / `brak_mozliwosci_oceny` | Zakres ratio decidendi |
| 22 | `systemic_impact` | ENUM | `niski` / `sredni` / `wysoki` / `precedensowy` | Wpływ systemowy |
| 23 | `jurisprudence_consistency` | ENUM | `zgodne` / `odchylenie` / `zmiana_linii` / `nowa_linia` / `brak_ustalonej_linii` | Zgodność z linią orzeczniczą |
| 24 | `proportionality_test_type` | ENUM | `proporcjonalnosc_sensu_stricto` / `koniecznosc` / `adekwatnosc` / `pelny_test` / `nie_zastosowano` | Typ testu proporcjonalności |
| 25 | `penalty_type` | ENUM | `pozbawienie_wolnosci` / `ograniczenie_wolnosci` / `grzywna` / `srodek_karny` / `srodek_zabezpieczajacy` / `nawiazka` / `nie_dotyczy` | Tylko sprawy karne: typ kary |
| 26 | `primary_legal_act` | ENUM | `kodeks_cywilny` / `kodeks_postepowania_cywilnego` / `kodeks_karny` / `kodeks_postepowania_karnego` / `kodeks_pracy` / `kodeks_spolek_handlowych` / `prawo_bankowe` / `ustawa_vat` / `ustawa_pit` / `ustawa_cit` / `ordynacja_podatkowa` / `kpa` / `ppsa` / `prawo_budowlane` / `prawo_zamowien_publicznych` / `ustawa_o_ochronie_konkurencji` / `rodo` / `prawo_upadlosciowe` / `konstytucja_rp` / `inne` | Główny akt prawny |
| 27 | `representative_type_plaintiff` | ENUM | `adwokat` / `radca_prawny` / `prokurator` / `rzecznik` / `pro_se` / `brak_danych` | Typ pełnomocnika powoda |

---

## C. KLASYFIKACJA — BOOL (17 pól)

Agregacja: proporcje (% true), korelacje z innymi polami, trendy czasowe.

| # | Pole | Opis |
|---|------|------|
| 28 | `eu_law_dimension` | Czy orzeczenie odwołuje się do prawa UE |
| 29 | `constitutional_dimension` | Czy powoływano się na Konstytucję RP |
| 30 | `preliminary_ruling_relevance` | Czy sąd rozważał pytanie prejudycjalne do TSUE |
| 31 | `has_dissent` | Czy istnieje zdanie odrębne |
| 32 | `overrules_prior` | Czy orzeczenie uchyla wcześniejszą zasadę |
| 33 | `identified_jurisprudence_line` | Czy sąd identyfikuje ustaloną linię orzeczniczą |
| 34 | `judge_specialization_match` | Czy sprawa trafiła do sędziego specjalizującego się w tej dziedzinie |
| 35 | `remedy_innovative` | Czy zastosowano innowacyjny środek ochrony |
| 36 | `proportionality_applied` | Czy sąd zastosował test proporcjonalności |
| 37 | `proportionality_explicit` | Czy test proporcjonalności był wyrażony expressis verbis |
| 38 | `retroactivity_question` | Czy sprawa dotyczyła retroaktywności |
| 39 | `transitional_provisions` | Czy stosowano przepisy przejściowe |
| 40 | `pro_se_party` | Czy któraś strona działała bez pełnomocnika |
| 41 | `legal_aid_used` | Czy korzystano z pomocy prawnej z urzędu |
| 42 | `outcome_predictable` | Czy wynik był przewidywalny na podstawie ustalonej linii |
| 43 | `circular_reasoning` | Czy uzasadnienie zawiera rozumowanie kołowe |
| 44 | `public_interest_mentioned` | Czy sąd powołał się na interes publiczny |

---

## D. METRYKI NUMERYCZNE (9 pól)

Agregacja: avg, median, percentyle, histogram, korelacje, regresja.

| # | Pole | Typ | Opis | Zakres |
|---|------|-----|------|--------|
| 45 | `case_duration_days` | INT | Czas trwania sprawy w tej instancji (dni) | 1-3650 |
| 46 | `remand_count` | INT | Ile razy sprawa była przekazywana do ponownego rozpoznania | 0-10 |
| 47 | `legal_references_count` | INT | Liczba cytowanych przepisów | 0-200 |
| 48 | `cited_cases_count` | INT | Liczba cytowanych orzeczeń | 0-100 |
| 49 | `legal_issues_count` | INT | Liczba zagadnień prawnych w sprawie | 1-15 |
| 50 | `text_length_chars` | INT | Długość pełnego tekstu orzeczenia | 1000-1000000 |
| 51 | `uzasadnienie_length_chars` | INT | Długość samego uzasadnienia (bez sentencji) | 500-500000 |
| 52 | `ratio_facts_to_law` | FLOAT | Proporcja faktów do rozważań prawnych (0.0-1.0) | 0.0-1.0 |
| 53 | `awarded_amount_total_pln` | FLOAT | Łączna kwota zasądzona (PLN), 0 jeśli nie dotyczy | 0-∞ |

---

## E. SKALE (SCORE 1-5) (6 pól)

Agregacja: avg, median, rozkład, korelacje, ranking sądów/dziedzin.

| # | Pole | Opis | Skala |
|---|------|------|-------|
| 54 | `reasoning_coherence_score` | Spójność logiczna uzasadnienia | 1=niespójne, 5=wzorcowe |
| 55 | `cognitive_complexity_score` | Złożoność poznawcza sprawy | 1=prosta, 5=bardzo złożona |
| 56 | `clarity_score` | Klarowność komunikacyjna uzasadnienia | 1=niezrozumiałe, 5=wzorcowo jasne |
| 57 | `procedural_complexity_score` | Złożoność proceduralna | 1=prosta, 5=bardzo skomplikowana |
| 58 | `judicial_creativity_significance` | Znaczenie innowacji sędziowskiej (0 jeśli brak) | 0=brak, 1=marginalne, 5=przełomowe |
| 59 | `argument_strength_avg` | Średnia siła argumentów sądu | 1=słabe, 5=bardzo przekonujące |

---

## F. MULTI-LABEL ENUM (5 pól)

Agregacja: częstości poszczególnych wartości, co-occurrence matrix, multi-label classification.

| # | Pole | Wartości dozwolone | Opis |
|---|------|--------------------|----- |
| 60 | `reasoning_methods_used` | `jezykowa` / `celowosciowa` / `systemowa` / `historyczna` / `prounijna` / `prokonstytucyjna` / `analogia` / `a_contrario` / `funkcjonalna` / `komparatystyczna` | Wszystkie metody wykładni użyte (multi-label) |
| 61 | `evidence_types_used` | `dokument_urzedowy` / `dokument_prywatny` / `zeznania_swiadkow` / `opinia_bieglego` / `przesluchanie_stron` / `oględziny` / `nagranie` / `dane_elektroniczne` / `ekspertyza` / `domniemanie` | Typy dowodów przeprowadzonych |
| 62 | `evidence_types_decisive` | (te same wartości co `evidence_types_used`) | Typy dowodów decydujących o rozstrzygnięciu |
| 63 | `precedent_treatment_types` | `podazono` / `odroznienie` / `uchylenie` / `rozszerzenie` / `krytyka` / `aprobata_z_zastrzezeniami` | Typy traktowania cytowanych precedensów |
| 64 | `persuasive_authority_types` | `doktryna_krajowa` / `doktryna_zagraniczna` / `orzecznictwo_zagraniczne` / `soft_law` / `raporty_organizacji` / `orzecznictwo_etpc` / `orzecznictwo_tsue` / `brak` | Źródła autorytetu perswazyjnego |

---

## G. KEYWORDS — listy z otwartego słownika (3 pola)

Agregacja: word frequency, TF-IDF, topic modeling, co-occurrence.

| # | Pole | Typ | Opis |
|---|------|-----|----- |
| 65 | `keywords` | KEYWORDS (5-15) | Słowa kluczowe w j. polskim |
| 66 | `legal_concepts` | KEYWORDS (3-10) | Pojęcia prawne (np. "klauzule abuzywne", "bezpodstawne wzbogacenie") |
| 67 | `constitutional_provisions_cited` | KEYWORDS (0-10) | Artykuły Konstytucji (np. "art. 2", "art. 32 ust. 1") |

---

## H. REFERENCJE — identyfikatory do budowania grafów (6 pól)

Agregacja: citation networks, PageRank, frequency ranking, co-citation analysis.

| # | Pole | Format elementu | Opis |
|---|------|-----------------|------|
| 68 | `legal_references` | `{act, article, paragraph}` | Cytowane przepisy krajowe |
| 69 | `eu_references` | `{type, reference, article}` | Cytowane prawo UE |
| 70 | `cited_cases` | `{court, signature, date, treatment}` | Cytowane orzeczenia z typem traktowania |
| 71 | `judges_panel` | `{name, role}` | Skład orzekający |
| 72 | `related_cases` | `{signature, relationship}` | Powiązane sprawy |
| 73 | `overruled_cases` | `{signature}` | Uchylone orzeczenia |

---

## I. TEKST OTWARTY (13 pól)

Agregacja wymaga NLP: embeddings, clustering, topic modeling, NER, similarity.
Każde pole ma companion enum (sekcja B/C) do natychmiastowej agregacji.

| # | Pole | Companion enum/bool/score | Opis |
|---|------|---------------------------|------|
| 74 | `title` | → `case_subject_category`, `legal_domain` | Tytuł opisowy (max 200 znaków) |
| 75 | `summary` | → `case_subject_category`, `decision_type` | Streszczenie 3-5 zdań |
| 76 | `thesis` | → `case_subject_category`, `primary_legal_act` | Teza prawna (1-3 zdań) |
| 77 | `factual_state` | → `case_subject_category`, `plaintiff_type`, `defendant_type` | Pełna narracja o stanie faktycznym |
| 78 | `legal_state` | → `primary_legal_act`, `legal_domain` | Podstawa prawna z artykułami |
| 79 | `decision_summary` | → `decision_type`, `remedy_primary` | Skrót rozstrzygnięcia (2-3 zdań) |
| 80 | `legal_effect` | → `systemic_impact` | Praktyczny skutek prawny |
| 81 | `ratio_decidendi` | → `ratio_scope`, `judicial_creativity_type` | Wiążąca zasada prawna |
| 82 | `obiter_dicta` | → (brak — z natury nieprzewidywalne) | Uwagi poboczne sądu (JSON array of strings) |
| 83 | `dissent_reasoning` | → `dissent_strength`, `has_dissent` | Uzasadnienie zdania odrębnego |
| 84 | `social_issue` | → `public_interest_mentioned` | Kontekst społeczny sprawy |
| 85 | `surprising_elements` | → `outcome_predictable` | Co było nieoczekiwane w rozstrzygnięciu |
| 86 | `distinguishing_facts` | → `case_subject_category` | Fakty odróżniające od typowych spraw |

---

## J. POLE STRUKTURALNE — JSON (1 pole)

| # | Pole | Wyciągnięte agregaty | Opis |
|---|------|---------------------|------|
| 87 | `awarded_amounts` | → `awarded_amount_total_pln`, `remedy_primary` | `[{type, amount, currency, recipient}]` — zasądzone kwoty |

---

## Kontrolowane słowniki — pełne definicje

### `case_subject_category` — taksonomia tematyczna (31 kategorii)

| Wartość | Dziedzina | Typowe sprawy |
|---------|-----------|---------------|
| `kredyt_frankowy` | cywilne | Unieważnienie umowy, klauzule abuzywne, spread, LIBOR/SARON |
| `odszkodowanie_szkoda_osobowa` | cywilne | Wypadki komunikacyjne, błędy medyczne, szkoda na osobie |
| `odszkodowanie_szkoda_majatkowa` | cywilne | Szkoda w mieniu, utracone korzyści, szkoda rzeczywista |
| `zadoscuczynienie` | cywilne | Naruszenie dóbr osobistych, krzywda, cierpienie |
| `umowa_sprzedaz` | cywilne/gosp. | Rękojmia, gwarancja, wady rzeczy, nieruchomości |
| `umowa_najem_dzierzawa` | cywilne | Eksmisja, zaległości czynszowe, wypowiedzenie |
| `umowa_o_dzielo_zlecenie` | cywilne | Kwalifikacja umowy, wynagrodzenie, odpowiedzialność |
| `prawo_pracy_zwolnienie` | pracy | Wypowiedzenie, dyscyplinarne, przywrócenie do pracy |
| `prawo_pracy_wynagrodzenie` | pracy | Nadgodziny, premie, wyrównanie, mobbing |
| `prawo_pracy_dyskryminacja` | pracy | Dyskryminacja, nierówne traktowanie, molestowanie |
| `spadki_zachowek` | rodzinne | Zachowek, testament, dział spadku |
| `rozwod_alimenty` | rodzinne | Rozwód, separacja, alimenty, opieka |
| `wlasnosc_nieruchomosci` | cywilne | Własność, współwłasność, rozgraniczenie |
| `sluzebnosc_zasiedzenie` | cywilne | Służebność drogi koniecznej, zasiedzenie |
| `spolki_odpowiedzialnosc` | gospodarcze | Odpowiedzialność wspólników, organów, art. 299 KSH |
| `upadlosc_restrukturyzacja` | gospodarcze | Upadłość, sanacja, układ z wierzycielami |
| `zamowienia_publiczne` | administracyjne | Przetargi, odwołania KIO, wykluczenie |
| `decyzja_administracyjna` | administracyjne | Kontrola legalności decyzji, bezczynność organu |
| `podatek_vat` | podatkowe | Odliczenie, stawka, zwolnienie, MPP, WDT/WNT |
| `podatek_pit_cit` | podatkowe | KUP, przychód, ulgi, rezydencja podatkowa |
| `podatek_od_nieruchomosci` | podatkowe | Stawka, zwolnienie, kwalifikacja budowli |
| `egzekucja` | cywilne | Egzekucja komornicza, powództwa przeciwegzekucyjne |
| `ochrona_konsumentow` | cywilne/gosp. | Klauzule abuzywne, praktyki nieuczciwe, UOKiK |
| `dane_osobowe_rodo` | administracyjne | RODO, decyzje PUODO, naruszenie danych |
| `prawo_budowlane` | administracyjne | Pozwolenie na budowę, legalizacja samowoli |
| `ochrona_srodowiska` | administracyjne | Decyzje środowiskowe, kary, emisje |
| `wlasnosc_intelektualna` | cywilne/gosp. | Znaki towarowe, patenty, prawa autorskie |
| `prawo_karne_narkotyki` | karne | Posiadanie, obrót substancjami |
| `prawo_karne_przeciwko_mieniu` | karne | Kradzież, oszustwo, przywłaszczenie |
| `prawo_karne_przeciwko_osobie` | karne | Pobicie, groźby, zabójstwo, uszczerbek na zdrowiu |
| `inne` | — | Pozostałe |

### `court_chamber` — taksonomia wydziałów (13 kategorii)

| Wartość | Stosowana w sądach | Typowe sygnatury |
|---------|--------------------|------------------|
| `cywilny` | rejonowe, okręgowe, apelacyjne | I C, I ACa, I CSK |
| `karny` | rejonowe, okręgowe, apelacyjne | II K, II AKa |
| `pracy` | rejonowe, okręgowe | IV P, IV Pa |
| `gospodarczy` | rejonowe, okręgowe | VIII GC, X GNc |
| `rodzinny` | rejonowe | III RC, III Nsm |
| `ubezpieczen_spolecznych` | okręgowe | VI U |
| `finansowy` | WSA, NSA | I SA, II FSK |
| `ogolnoadministracyjny` | WSA, NSA | II SA, II OSK |
| `izba_cywilna` | SN | I CSK, III CZP |
| `izba_karna` | SN | II KK, I KZP |
| `izba_pracy` | SN | I PK, III UZP |
| `izba_kontroli_nadzwyczajnej` | SN | I NSNc |
| `inne` | — | — |

### `plaintiff_type` / `defendant_type` — taksonomia stron (13 kategorii)

| Wartość | Opis | Przykłady |
|---------|------|-----------|
| `osoba_fizyczna` | Konsument, obywatel nieprowadzący działalności | Jan Kowalski, konsument |
| `osoba_fizyczna_przedsiebiorca` | Osoba fizyczna prowadząca JDG | Podatnik, przedsiębiorca |
| `spolka_kapitalowa` | Sp. z o.o., S.A. | ABC Sp. z o.o., Bank XYZ S.A. |
| `spolka_osobowa` | Sp. jawna, komandytowa, partnerska | Kancelaria sp. partnerska |
| `organ_administracji` | Organ administracji publicznej | Dyrektor IS, Prezydent miasta, Minister |
| `prokurator` | Prokuratura | Prokurator Rejonowy |
| `rzecznik_konsumentow` | Powiatowy/miejski rzecznik konsumentów | Rzecznik Konsumentów |
| `fundacja_stowarzyszenie` | Organizacja pozarządowa | Fundacja ABC |
| `jst` | Jednostka samorządu terytorialnego | Gmina, Powiat, Województwo |
| `skarb_panstwa` | Skarb Państwa z różnymi statio fisci | SP - GDDKiA, SP - Starostwo |
| `spoldzielnia` | Spółdzielnia mieszkaniowa, pracy | SM "Osiedle" |
| `zwiazek_zawodowy` | Związek zawodowy | NSZZ "Solidarność" |
| `inne` | Inne | — |

### `evidence_types_used` / `evidence_types_decisive` (10 kategorii)

| Wartość | Opis | Moc dowodowa |
|---------|------|-------------|
| `dokument_urzedowy` | Akt notarialny, wpis do KRS, zaświadczenie | Domniemanie prawdziwości (art. 244 KPC) |
| `dokument_prywatny` | Umowa, faktura, korespondencja email | Dowód że osoba złożyła oświadczenie |
| `zeznania_swiadkow` | Zeznania świadków | Swobodna ocena sądu |
| `opinia_bieglego` | Opinia biegłego sądowego | Ekspertyza specjalistyczna |
| `przesluchanie_stron` | Przesłuchanie stron (informacyjne lub dowodowe) | Subsydiarny środek dowodowy |
| `ogledziny` | Oględziny miejsca, przedmiotu | Bezpośrednie poznanie sądu |
| `nagranie` | Nagranie audio/wideo | Zależy od legalności uzyskania |
| `dane_elektroniczne` | Logi, dane GPS, dane bankowe, metadane | Zależy od uwierzytelnienia |
| `ekspertyza` | Ekspertyza prywatna (nie biegłego sądowego) | Dokument prywatny ze szczególnym walorem |
| `domniemanie` | Domniemanie faktyczne lub prawne | Przerzuca ciężar dowodu |

### `precedent_treatment_types` (6 kategorii)

| Wartość | Definicja | Wpływ na sieć cytowań |
|---------|-----------|----------------------|
| `podazono` | Sąd stosuje zasadę z cytowanego orzeczenia | Wzmacnia precedens |
| `odroznienie` | Sąd rozróżnia stan faktyczny — nie stosuje precedensu | Zawęża zakres |
| `uchylenie` | Sąd wprost odchodzi od wcześniejszej zasady | Osłabia/kończy precedens |
| `rozszerzenie` | Sąd rozszerza zasadę na nowy stan faktyczny | Rozwija precedens |
| `krytyka` | Sąd krytykuje ale nie odchodzi formalnie | Sygnalizuje potencjalną zmianę |
| `aprobata_z_zastrzezeniami` | Sąd co do zasady zgadza się, ale z modyfikacjami | Ewolucja precedensu |

---

## Instrukcje ekstrakcji

### Zasady ogólne
- Ekstraktuj WYŁĄCZNIE z tekstu dokumentu
- Język: polski
- Daty: ISO 8601 (YYYY-MM-DD)
- Puste pola: `""` (string), `[]` (array), `null` (object), `0` (int/score), `false` (bool)
- ENUM: TYLKO wartości z listy, `inne`/`brak` gdy żadna nie pasuje

### Reguły mapowania na enum

**`case_subject_category`** → klasyfikuj na podstawie meritum sprawy:
- "unieważnienie umowy kredytu indeksowanego do CHF" → `kredyt_frankowy`
- "odszkodowanie za wypadek komunikacyjny" → `odszkodowanie_szkoda_osobowa`
- "przywrócenie do pracy po zwolnieniu dyscyplinarnym" → `prawo_pracy_zwolnienie`
- "skarga na decyzję Dyrektora IS w przedmiocie VAT" → `podatek_vat`

**`court_chamber`** → mapuj z wydziału/sygnatury:
- "I Wydział Cywilny", sygnatura `I C ...` → `cywilny`
- "II Wydział Karny", sygnatura `II K ...` → `karny`
- "I SA/Wa ..." (WSA finansowy) → `finansowy`
- "I CSK ..." (SN Izba Cywilna) → `izba_cywilna`

**`plaintiff_type` / `defendant_type`** → identyfikuj z opisu stron:
- "powód Jan Kowalski" + brak wzmianki o działalności → `osoba_fizyczna`
- "pozwana ABC Sp. z o.o." → `spolka_kapitalowa`
- "skarżący zaskarżył decyzję Dyrektora..." → plaintiff=`osoba_fizyczna`/`spolka`, defendant=`organ_administracji`

**`primary_reasoning_method`** → identyfikuj dominującą metodę:
- sąd analizuje brzmienie przepisu słowo po słowie → `jezykowa`
- sąd bada cel regulacji, ratio legis → `celowosciowa`
- sąd analizuje przepis w kontekście całego aktu → `systemowa`
- sąd porównuje z prawem UE, interpretacja prounijna → `prounijna`

**`party_power_asymmetry`**:
- konsument vs bank/ubezpieczyciel → `silna_asymetria`
- dwie spółki porównywalnej wielkości → `symetryczna`
- osoba fizyczna vs organ administracji → `silna_asymetria`
- mała firma vs duża korporacja → `umiarkowana_asymetria`

### Walidacja
1. Każde pole ENUM używa TYLKO wartości z listy
2. `court_chamber` jest spójny z `court_type` i `court_name`
3. Pola BOOL: wyłącznie `true`/`false`, nie `null`
4. Pola INT: nieujemne, w zdefiniowanym zakresie
5. `ratio_facts_to_law`: wartość od 0.0 do 1.0
6. `reasoning_methods_used` nie może być puste — minimum 1 metoda
7. `penalty_type`: ustawiony na `nie_dotyczy` gdy `legal_domain` ≠ `karne` / `karne_skarbowe`
8. `has_dissent` = `true` wymaga `dissent_strength` ≠ `brak`
9. `overrules_prior` = `true` wymaga `overruling_scope` ≠ `brak`
10. `proportionality_applied` = `true` wymaga `proportionality_test_type` ≠ `nie_zastosowano`
