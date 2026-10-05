# Pilot: bramka spraw o agitację wyborczą

## Stan i cel

To jest **pakiet do ręcznej anotacji**, bez zweryfikowanych etykiet. Cel: sprawdzić, czy mały model potrafi rozpoznawać wyroki warte kosztownej ekstrakcji informacji o sprawach dotyczących agitacji wyborczej. Nie zmieniać obecnego pipeline'u ani nie traktować wyjścia GPT-4o jako prawdy wzorcowej.

Kod `juddges/use_cases/agitation.py` ustawia `MIN_THRESHOLD = 5` i przekazuje tę wartość do `filter_judgments`. Wyniki zapisane lokalnie zawierają 7 733 unikalne wyroki: 6 760 poniżej progu i 973 od progu 5 wzwyż. W starszej notatce w vaultcie podano próg 10; to nie jest aktualna wartość w skrypcie.

## Źródło i próba

Źródła, dostępne tylko lokalnie i ignorowane przez Git:

- `data/analysis/agitacja_wyborcza/all_judgments_merged.pkl` — wyniki wyszukiwania przed progiem.
- `data/analysis/agitacja_wyborcza/judgments_with_extraction.pkl` — ekstrakcje tylko dla wyników od progu 5.

`scripts/annotation/build_agitation_pilot.py` wybiera deterministycznie 100 dokumentów (seed `20261005`): po 25 z `query_count = 1`, `2–4`, `5–9`, `≥10`. Grupy źródłowe liczą odpowiednio 5 029, 1 731, 535 i 438 dokumentów. Skrypt odrzuca powtórzenia sygnatury (`docket_number`) oraz identycznego tekstu. To konserwatywne przybliżenie grupowania po sprawie: przed zamrożeniem testu trzeba sprawdzić powiązane sygnatury, apelacje i bliskie duplikaty, których prosty klucz nie wyłapie.

Uruchomienie w worktree `.worktrees/feat-74-agitation-annotation-pilot/`; `../../data` wskazuje lokalne dane w głównym checkoutcie tylko do odczytu, a `../../.venv` jest istniejącym środowiskiem tego repo:

```bash
../../.venv/bin/python scripts/annotation/build_agitation_pilot.py \
  --merged ../../data/analysis/agitacja_wyborcza/all_judgments_merged.pkl \
  --extractions ../../data/analysis/agitacja_wyborcza/judgments_with_extraction.pkl \
  --output data/analysis/agitacja_wyborcza/pilot_v1
```

Katalog docelowy musi nie istnieć, żeby uniknąć nadpisania ręcznych ocen. `sampling_manifest.json` zawiera hashe plików wejściowych, seed, kolejność kart i stratum. `tasks.json` zawiera identyfikator wyroku i pełny tekst, ale nie wynik ekstrakcji ani `query_count` — to pakiet ślepy dla recenzenta. `weak_labels.json` zawiera 50 dostępnych etykiet GPT z połowy próby; należy go otworzyć dopiero po ręcznej ocenie.

## Instrukcja dla recenzenta

Zaimportować `tasks.json` jako zadania do Label Studio lub równoważnego narzędzia. Schemat UI:

```xml
<View>
  <Header value="$judgment_id" />
  <Text name="judgment" value="$text" />
  <Choices name="agitation_case" toName="judgment" choice="single" required="true">
    <Choice value="yes" />
    <Choice value="no" />
    <Choice value="uncertain" />
  </Choices>
  <TextArea name="rationale" toName="judgment" placeholder="Wskaż fragment i powód decyzji, zwłaszcza przy uncertain" />
</View>
```

**Robocza definicja** do zatwierdzenia z ekspertem prawa wyborczego: `yes` oznacza, że treść wyroku dotyczy postępowania związanego z agitacją wyborczą / materiałem wyborczym w ramach analizowanego przypadku użycia; `no` — inne sprawy, nawet gdy wyszukiwarka dopasowała słowa z zapytań; `uncertain` — brak wystarczających danych albo spór interpretacyjny. Recenzent zapisuje krótki fragment uzasadniający decyzję. Ekspert doprecyzowuje zasady dla wzmianek o art. 111, spraw granicznych i wyroków apelacyjnych **przed** główną anotacją.

Najpierw 20 kart pilotażowych i korekta instrukcji, potem wszystkie 100. Co najmniej przypadki `uncertain` oraz losowa część `yes`/`no` przechodzą drugą ocenę i adjudykację. Zachować surowe oceny i decyzję końcową osobno. Dopóki każda karta nie ma zatwierdzonej etykiety końcowej, nie nazywać zbioru gold setem.

## Jak liczyć wyniki po zatwierdzeniu etykiet

- Najpierw `precision`, `recall` i macierz pomyłek obecnej reguły `query_count ≥ 5`; szczególnie policzyć fałszywie odrzucone sprawy poniżej progu.
- Następnie porównać prosty model na tych samych kartach. Dzielić po sprawie, a próg modelu wybierać tylko na walidacji. Zbiór 100 kart służy do oszacowania błędów i instrukcji anotacji; jest za mały na wiarygodny ranking wielu modeli.
- **Próba jest celowo zbalansowana między zakresami**, a nie reprezentatywna dla 7 733 wyników. Podawać wyniki per stratum; do estymacji dla całej puli zważyć liczniki macierzy pomyłek udziałami `5029/7733`, `1731/7733`, `535/7733`, `438/7733` i dopiero z nich wyliczyć precision/recall, albo zebrać osobną losową próbę produkcyjną. Nie raportować surowej średniej z 25×4 jako produkcyjnego recall/precision.
- Etykiety GPT z `weak_labels.json` służą do analizy rozbieżności i ewentualnego trenowania na słabych danych. Nie zastępują ręcznych etykiet testowych; dla `query_count < 5` w ogóle ich nie ma.

## Bramka do następnego etapu

Po zatwierdzeniu 100 etykiet: raport błędów aktualnej reguły, definicja progu kosztu pomyłek, osobny zbiór treningowy i walidacyjny bez przecieku spraw. Dopiero wtedy porównanie małych modeli, licencji, MLflow, fine-tuning, kwantyzacja i pomiar na GPU. Wdrożenie w klastrze pozostaje oddzielną decyzją po tych wynikach.
