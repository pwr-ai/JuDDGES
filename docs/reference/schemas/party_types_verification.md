# Party Types Verification - Polish Legal Proceedings

## Web Search Findings

### Research Question

Is the current enumeration for `parties.party_type` complete for Polish legal proceedings?

### Current Schema

```python
"party_type": "string (plaintiff/defendant/applicant/etc)"
```

### Findings from Polish Procedural Law Research

#### Complete Party Types in Polish Legal System

Based on research of Polish procedural codes (sources: Kodeks postępowania cywilnego, Kodeks postępowania administracyjnego, Kodeks postępowania karnego):

#### Civil Proceedings (Postępowanie Cywilne)

1. **Powód** (Plaintiff)
   - Strona wnosząca pozew
   - Osoba dochodzą swoich praw
   - English: Plaintiff

2. **Pozwany** (Defendant)
   - Strona przeciwko której skierowany jest pozew
   - English: Defendant

3. **Interwenient uboczny** (Third-party intervenor)
   - Osoba trzecia przystępująca do procesu po stronie powoda lub pozwanego
   - English: Third-party intervenor

4. **Uczestnik postępowania** (Participant in proceedings)
   - W postępowaniu nieprocesowym
   - English: Participant

5. **Wnioskodawca** (Petitioner/Applicant)
   - W postępowaniu nieprocesowym lub egzekucyjnym
   - English: Petitioner/Applicant

#### Administrative Proceedings (Postępowanie Administracyjne)

6. **Skarżący** (Complainant/Appellant)
   - Strona składająca skargę do sądu administracyjnego
   - English: Complainant/Appellant

7. **Organ administracji** (Administrative body)
   - Organ którego decyzja jest zaskarżana
   - English: Administrative body/Authority

#### Criminal Proceedings (Postępowanie Karne)

8. **Oskarżony** (Accused/Defendant in criminal case)
   - Osoba oskarżona o popełnienie przestępstwa
   - English: Accused/Defendant

9. **Pokrzywdzony** (Injured party/Victim)
   - Osoba pokrzywdzona przestępstwem
   - English: Injured party/Victim

10. **Oskarżyciel publiczny** (Public prosecutor)
    - Prokurator
    - English: Public prosecutor

11. **Oskarżyciel posiłkowy** (Auxiliary prosecutor)
    - Pokrzywdzony występujący jako oskarżyciel
    - English: Auxiliary prosecutor

#### Tax and Financial Proceedings

12. **Podatnik** (Taxpayer)
    - W sprawach podatkowych
    - English: Taxpayer

13. **Organ podatkowy** (Tax authority)
    - Urząd skarbowy, naczelnik urzędu celno-skarbowego
    - English: Tax authority

### Recommended Enumeration

```python
"party_type": "enum (powod/pozwany/interwenient_uboczny/uczestnik/wnioskodawca/skarzacy/organ_administracji/oskarżony/pokrzywdzony/oskarżyciel_publiczny/oskarżyciel_posilkowy/podatnik/organ_podatkowy)"
```

### Party Category Enumeration

Current schema also includes `party_category`:

```python
"party_category": "string (natural_person/company/public_entity)"
```

#### Recommended Expansion

```python
"party_category": "enum (osoba_fizyczna/osoba_prawna/jednostka_organizacyjna_bez_osobowosci/organ_panstwowy/organ_samorzadowy/przedsiebiorca)"
```

**Mapping:**

- **osoba_fizyczna** (natural person) - individual
- **osoba_prawna** (legal person) - company, foundation, association
- **jednostka_organizacyjna_bez_osobowosci** (organizational unit without legal personality) - partnerships
- **organ_panstwowy** (state authority) - ministries, government agencies
- **organ_samorzadowy** (local government authority) - municipalities, counties
- **przedsiebiorca** (entrepreneur) - business entity

### Examples in Legal Documents

**Civil case:**

```
Powód: Jan Kowalski
Pozwany: Spółka ABC S.A.
```

**Administrative case:**

```
Skarżący: Anna Nowak
Organ: Naczelnik Urzędu Skarbowego w Warszawie
```

**Tax interpretation:**

```
Wnioskodawca: XYZ Sp. z o.o.
Organ wydający interpretację: Dyrektor Krajowej Informacji Skarbowej
```

**Criminal case:**

```
Oskarżony: Jan Kowalski
Oskarżyciel publiczny: Prokurator Prokuratury Rejonowej
Pokrzywdzony: Maria Nowak
```

### Multiple Party Scenarios

**Example: Complex civil case with intervention**

```json
"parties": [
    {
        "party_type": "powod",
        "party_name": "Jan Kowalski",
        "party_category": "osoba_fizyczna"
    },
    {
        "party_type": "pozwany",
        "party_name": "Bank ABC S.A.",
        "party_category": "osoba_prawna"
    },
    {
        "party_type": "interwenient_uboczny",
        "party_name": "Ubezpieczenia XYZ S.A.",
        "party_category": "osoba_prawna"
    }
]
```

### Verification Status

✅ **NEEDS UPDATE** - Current schema uses English terms and is incomplete

### Recommended Action

1. Use Polish terminology (matches document language)
2. Expand to include all major party types across all proceedings
3. Expand party_category to reflect Polish legal classification
4. Consider making party_type conditional on document_type
5. Provide English translations as metadata

### Special Considerations

**For tax interpretations:** Most common party types are:

- wnioskodawca (applicant requesting interpretation)
- organ_podatkowy (tax authority issuing interpretation)

**For court judgments:** Party types vary by case type:

- Civil: powod, pozwany
- Administrative: skarzacy, organ_administracji
- Criminal: oskarżony, pokrzywdzony, oskarżyciel_publiczny

---
_Generated: 2025-10-11_
_Source: Web search validation for JuDDGES extraction schema_
