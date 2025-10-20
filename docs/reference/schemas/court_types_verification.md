# Court Types Verification - Polish Judicial System

## Web Search Findings

### Research Question

Is the current enumeration for `judgment_specific.court_type` complete for Polish court system?

### Current Schema

```python
"court_type": "string (district/regional/appeal/supreme)"
```

### Findings from Polish Judicial System Research

#### Complete Polish Court Structure

Based on research of Polish judicial system (sources: ms.gov.pl, iustitia.pl, sn.pl):

#### Sądy Powszechne (Common Courts)

1. **Sąd Rejonowy** (District Court)
   - Najniższy szczebel w hierarchii sądów powszechnych
   - Rozpatruje sprawy w pierwszej instancji
   - English: District Court

2. **Sąd Okręgowy** (Regional Court)
   - Średni szczebel w hierarchii
   - Sąd pierwszej instancji dla poważniejszych spraw
   - Sąd odwoławczy od wyroków sądów rejonowych
   - English: Regional Court

3. **Sąd Apelacyjny** (Court of Appeal)
   - Sąd drugiej instancji
   - Rozpatruje apelacje od wyroków sądów okręgowych
   - English: Court of Appeal

4. **Sąd Najwyższy** (Supreme Court)
   - Najwyższy organ sądowniczy
   - Rozpatruje kasacje
   - Podejmuje uchwały wyjaśniające przepisy
   - English: Supreme Court

#### Sądy Administracyjne (Administrative Courts)

5. **Wojewódzki Sąd Administracyjny** (Provincial Administrative Court)
   - Rozpatruje skargi na decyzje organów administracji
   - Pierwsza instancja w sądownictwie administracyjnym
   - English: Provincial Administrative Court

6. **Naczelny Sąd Administracyjny** (Supreme Administrative Court)
   - Najwyższy organ sądownictwa administracyjnego
   - Rozpatruje skargi kasacyjne
   - English: Supreme Administrative Court

#### Sądy Szczególne (Special Courts)

7. **Trybunał Konstytucyjny** (Constitutional Tribunal)
   - Orzeka o zgodności ustaw z Konstytucją
   - Nie jest częścią systemu sądów powszechnych
   - English: Constitutional Tribunal

8. **Trybunał Stanu** (State Tribunal)
   - Orzeka w sprawach odpowiedzialności konstytucyjnej
   - English: State Tribunal

### Recommended Enumeration

```python
"court_type": "enum (rejonowy/okregowy/apelacyjny/najwyzszy/wojewodzki_administracyjny/naczelny_administracyjny/trybunał_konstytucyjny/trybunal_stanu)"
```

### Polish to English Mapping

```python
COURT_TYPE_TRANSLATIONS = {
    "rejonowy": "District Court",
    "okregowy": "Regional Court",
    "apelacyjny": "Court of Appeal",
    "najwyzszy": "Supreme Court",
    "wojewodzki_administracyjny": "Provincial Administrative Court",
    "naczelny_administracyjny": "Supreme Administrative Court",
    "trybunal_konstytucyjny": "Constitutional Tribunal",
    "trybunal_stanu": "State Tribunal"
}
```

### Examples in Legal Documents

**Sąd Rejonowy:**

- "Sąd Rejonowy w Warszawie, VII Wydział Cywilny"
- "Sąd Rejonowy dla m.st. Warszawy"

**Sąd Okręgowy:**

- "Sąd Okręgowy w Krakowie, I Wydział Cywilny"
- "Wyrok Sądu Okręgowego z dnia..."

**Sąd Apelacyjny:**

- "Sąd Apelacyjny w Gdańsku, I Wydział Cywilny"
- "Sąd Apelacyjny w Warszawie uchyla zaskarżony wyrok..."

**Sąd Najwyższy:**

- "Wyrok Sądu Najwyższego z dnia 15 stycznia 2024 r., sygn. I CSK 123/23"
- "Uchwała Sądu Najwyższego - Izba Cywilna"

**Wojewódzki Sąd Administracyjny:**

- "Wojewódzki Sąd Administracyjny w Warszawie, III SA/Wa 1234/23"
- "Wyrok WSA w sprawie interpretacji podatkowej"

**Naczelny Sąd Administracyjny:**

- "Wyrok Naczelnego Sądu Administracyjnego z dnia..."
- "NSA uchyla wyrok WSA i przekazuje sprawę do ponownego rozpoznania"

### Court Jurisdiction Overview

| Court Type | Jurisdiction | Instance |
|-----------|-------------|----------|
| Sąd Rejonowy | Civil, criminal, family | First |
| Sąd Okręgowy | Civil, criminal, commercial | First/Second |
| Sąd Apelacyjny | All civil and criminal | Second |
| Sąd Najwyższy | Cassation, legal interpretation | Final |
| WSA | Administrative cases | First |
| NSA | Administrative cassation | Final |
| Trybunał Konstytucyjny | Constitutional review | Special |

### Verification Status

✅ **NEEDS UPDATE** - Current schema uses English terms and is incomplete

### Recommended Action

1. Use Polish terminology as primary (matches document language)
2. Expand to include all 8 types of courts
3. Include administrative courts (crucial for tax interpretations)
4. Provide English translations as metadata

---
_Generated: 2025-10-11_
_Source: Web search validation for JuDDGES extraction schema_
