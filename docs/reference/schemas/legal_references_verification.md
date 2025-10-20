# Legal References Type Verification - Polish Legal System

## Web Search Findings

### Research Question

Does the current enumeration for `legal_references.type` field correctly represent types of legal acts in Polish law?

### Current Schema

```python
"type": "string (statute/article/case_law/regulation)"
```

### Findings from Polish Legal System Research

#### Complete Legal Act Hierarchy in Poland

Based on research of Polish legal system (sources: konstytucja.pl, isap.sejm.gov.pl, gov.pl):

1. **Konstytucja** (Constitution)
   - Najwyższa moc prawna w Polsce
   - Uchwalona przez Zgromadzenie Narodowe

2. **Ustawa** (Statute/Act)
   - Uchwalana przez Sejm i Senat
   - Podstawowe akty prawne regulujące różne dziedziny życia

3. **Rozporządzenie** (Regulation/Decree)
   - Wydawane przez organy wykonawcze (Rada Ministrów, ministrowie)
   - Musi mieć upoważnienie ustawowe

4. **Akty prawa miejscowego** (Local law acts)
   - Uchwały rad gmin, powiatów, sejmików wojewódzkich
   - Obowiązują na określonym terenie

5. **Umowy międzynarodowe** (International treaties)
   - Ratyfikowane umowy międzynarodowe
   - Po ratyfikacji stają się częścią polskiego porządku prawnego

6. **Przepisy UE** (EU regulations)
   - Rozporządzenia UE bezpośrednio obowiązujące
   - Dyrektywy wymagające implementacji

7. **Orzecznictwo** (Case law)
   - Wyroki Trybunału Konstytucyjnego
   - Uchwały Sądu Najwyższego
   - Wyroki sądów powszechnych i administracyjnych

8. **Statuty** (Statutes - organizational)
   - Akty wewnętrzne organizacji
   - Regulaminy

### Recommended Enumeration

```python
"type": "enum (konstytucja/ustawa/rozporzadzenie/akt_prawa_miejscowego/umowa_miedzynarodowa/przepis_ue/orzecznictwo/statut)"
```

### Examples in Legal Documents

**Konstytucja:**

- Art. 32 Konstytucji RP (równość wobec prawa)
- Art. 64 Konstytucji RP (prawo własności)

**Ustawa:**

- Ustawa z dnia 23 kwietnia 1964 r. - Kodeks cywilny
- Ustawa o podatku od towarów i usług

**Rozporządzenie:**

- Rozporządzenie Ministra Finansów w sprawie...
- Rozporządzenie Rady Ministrów w sprawie...

**Orzecznictwo:**

- Wyrok Sądu Najwyższego sygn. I CSK 123/21
- Uchwała SN III CZP 45/20

### Verification Status

✅ **NEEDS UPDATE** - Current schema is incomplete for Polish legal system

### Recommended Action

Expand the enumeration to include all 8 types of legal acts recognized in Polish legal system.

---
_Generated: 2025-10-11_
_Source: Web search validation for JuDDGES extraction schema_
