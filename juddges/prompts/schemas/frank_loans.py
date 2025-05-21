from textwrap import dedent

SWISS_FRANC_LOAN_SCHEMA = dedent("""
    apelacja: string, description: "Określenie apelacji, w której znajduje się sąd rozpoznający sprawę", example: "Apelacja warszawska"
    typ_sadu: enum [Sąd Rejonowy, Sąd Okręgowy, Sąd Frankowy], description: "Typ sądu rozpoznającego sprawę", example: "Sąd Rejonowy"
    instancja_sadu: enum [Sąd I instancji, Sąd odwoławczy], description: "Czy sąd jest I instancji czy odwoławczy", example: "Sąd I instancji"
    podstawa_prawna: string, description: "Podstawa prawna roszczenia", example: "Art. 385(1) Kodeksu cywilnego"
    podstawa_prawna_podana: boolean, description: "Czy powód podał podstawę prawną?", example: true
    rodzaj_roszczenia: enum [O ustalenie istnienia/nieistnienia stosunku prawnego, O ukształtowanie stosunku prawnego, O zapłatę, Roszczenia dodatkowe], description: "Rodzaj roszczenia", example: "O zapłatę"
    modyfikacje_powodztwa: boolean, description: "Czy były modyfikacje powództwa", example: true
    typ_modyfikacji: enum [Rodzaj roszczenia, Kwoty roszczenia], description: "Typ modyfikacji powództwa", example: "Kwoty roszczenia"
    status_kredytobiorcy: enum [Konsument, Przedsiębiorca], description: "Status kredytobiorcy", example: "Konsument"
    wspoluczestnictwo_powodowe: boolean, description: "Czy współuczestnictwo po stronie powodowej", example: false
    typ_wspoluczestnictwa: enum [Małżeństwo, Konkubinat, Inni członkowie rodziny, Spadkobiercy], description: "Rodzaj współuczestnictwa", example: "Małżeństwo"
    rola_pozwanego: enum [Strona umowy, Następca prawny], description: "Czy pozwany faktycznie był stroną umowy czy następcą prawnym", example: "Strona umowy"
    wczesniejsze_skargi_do_rzecznika: boolean, description: "Czy były uprzednie skargi do rzecznika finansowego", example: false
    umowa_kredytowa: enum [Zawarta bezpośrednio w banku, Z udziałem pośrednika], description: "Czy umowa kredytowa zawierana bezpośrednio w banku czy z pośrednikiem", example: "Zawarta bezpośrednio w banku"
    klauzula_niedozwolona: boolean, description: "Istnienie klauzuli niedozwolonej w umowie", example: true
    wpisana_do_rejestru_uokik: boolean, description: "Czy klauzula wpisana do rejestru klauzul niedozwolonych UOKiK", example: true
    waluta_splaty: enum [PLN, CHF], description: "Waluta spłaty", example: "CHF"
    aneks_do_umowy: boolean, description: "Czy był aneks do umowy", example: true
    data_aneksu: date (ISO 8601), description: "Data aneksu", example: "2023-01-15"
    przedmiot_aneksu: enum [Zmiana waluty spłaty, Inne kwestie], description: "Czego dotyczył aneks", example: "Zmiana waluty spłaty"
    status_splaty_kredytu: boolean, description: "Czy kredyt był spłacony, w tym w trakcie procesu", example: true
    data_wyroku: date (ISO 8601), description: "Data wydania wyroku", example: "2024-11-30"
    rozstrzygniecie_sadu: string, description: "Rozstrzygnięcie", example: "Oddalenie powództwa"
    typ_rozstrzygniecia: enum [Uwzględnienie powództwa w całości, Uwzględnienie powództwa w części, Oddalenie powództwa, Oddalenie apelacji, Zmiana wyroku, Przekazanie do ponownego rozpoznania], description: "Typ rozstrzygnięcia", example: "Oddalenie powództwa"
    sesja_sadowa: enum [Rozprawa, Posiedzenie niejawne], description: "Czy wyrok wydano na rozprawie czy na posiedzeniu niejawnym", example: "Rozprawa"
    dowody: list of enum [Przesłuchanie stron, Przesłuchanie świadków, Dowód z opinii biegłego], description: "Jakie dowody zostały przeprowadzone", example: ["Przesłuchanie stron", "Dowód z opinii biegłego"]
    oswiadczenie_niewaznosci: boolean, description: "Czy odbierano oświadczenie powoda o skutkach nieważności umowy", example: false
    odwolanie_do_sn: boolean, description: "Czy odwołano się do orzecznictwa SN", example: true
    odwolanie_do_tsue: boolean, description: "Czy odwołano się do orzecznictwa TSUE", example: true
    teoria_prawna: enum [Teoria dwóch kondykcji, Teoria salda], description: "Teoria prawna, na której oparto wyrok", example: "Teoria dwóch kondykcji"
    zarzut_zatrzymania_lub_potracenia: boolean, description: "Czy uwzględniono zarzut zatrzymania lub potrącenia", example: false
    odsetki_ustawowe: boolean, description: "Czy uwzględniono odsetki ustawowe", example: true
    data_rozpoczecia_odsetek: enum [Od dnia wezwania do zapłaty, Od daty wytoczenia powództwa, Od daty wydania wyroku, Od innej daty], description: "Data rozpoczęcia naliczania odsetek", example: "Od daty wydania wyroku"
    koszty_postepowania: boolean, description: "Czy zasądzono zwrot kosztów postępowania", example: true
    beneficjent_kosztow: string, description: "Na rzecz której strony zasądzono zwrot kosztów", example: "Pozwany"
    zabezpieczenie_udzielone: boolean, description: "Czy udzielono zabezpieczenia", example: true
    rodzaj_zabezpieczenia: string, description: "Rodzaj zabezpieczenia", example: "Wstrzymanie egzekucji"
    zabezpieczenie_pierwsza_instancja: boolean, description: "Czy zabezpieczenia udzielił sąd I instancji", example: true
    czas_trwania_sprawy: string, description: "Czas rozpoznania sprawy – od złożenia pozwu do wydania wyroku", example: "2 lata 3 miesiące
    wynik_sprawy: enum [Wygrana kredytobiorcy, Wygrana banku, Częściowe uwzględnienie roszczeń obu stron], description: "Ocena, czy bank czy kredytobiorca wygrał sprawę", example: "Wygrana kredytobiorcy"
    szczegoły_wyniku_sprawy: string, description: "Szczegóły dotyczące wyniku sprawy", example: "Kredytobiorca wygrał, umowa uznana za nieważną
    sprawa_frankowiczów: boolean, description: "Czy sprawa dotyczy kredytu frankowego (CHF)", example: true
""").strip()
