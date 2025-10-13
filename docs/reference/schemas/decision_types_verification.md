# Decision Types Verification - Polish Court System

## Web Search Findings

### Research Question

Is the current enumeration for `outcome.decision_type` complete for Polish court decisions?

### Current Schema

```python
"decision_type": "string (uwzględniono/oddalono/uchylono/pozytywne/negatywne)"
```

### Findings from Polish Court System Research

#### Complete Decision Types in Polish Courts

Based on research of Polish procedural law (sources: kodeks postępowania cywilnego, kodeks postępowania administracyjnego):

1. **Uwzględniono w całości** (Granted in full)
   - Pozew/skarga została uwzględniona w pełnym zakresie
   - Sąd przyznał rację stronie wnoszącej

2. **Uwzględniono w części** (Granted in part)
   - Pozew/skarga uwzględniona częściowo
   - Częściowe przyznanie racji obu stronom

3. **Oddalono** (Dismissed)
   - Pozew/skarga odrzucona jako bezzasadna
   - Strona wnosząca przegrała sprawę

4. **Umorzono** (Discontinued)
   - Postępowanie zakończone bez merytorycznego rozstrzygnięcia
   - Przyczyny proceduralne (np. cofnięcie pozwu, śmierć strony)

5. **Uchylono** (Overturned/Annulled)
   - W postępowaniu odwoławczym - uchylenie wyroku sądu niższej instancji
   - W postępowaniu kasacyjnym

6. **Uchylono i przekazano do ponownego rozpoznania** (Remanded)
   - Sąd wyższej instancji uchyla wyrok i przekazuje sprawę do ponownego rozpatrzenia

7. **Zmieniono** (Modified)
   - Wyrok sądu niższej instancji został zmieniony przez sąd odwoławczy
   - Częściowa lub całkowita zmiana rozstrzygnięcia

8. **Utrzymano w mocy** (Upheld)
   - Sąd odwoławczy potwierdził wyrok sądu niższej instancji
   - Brak podstaw do zmiany lub uchylenia

9. **Odrzucono** (Rejected on procedural grounds)
   - Pozew/skarga odrzucona z przyczyn formalnych
   - Np. brak właściwości sądu, wniesienie po terminie

### Tax Interpretation Specific Decision Types

For tax interpretations (interpretacje podatkowe):

1. **Stanowisko prawidłowe** (Position correct)
   - Organy podatkowe zgadzają się z interpretacją podatnika

2. **Stanowisko nieprawidłowe** (Position incorrect)
   - Organy podatkowe nie zgadzają się z interpretacją podatnika

3. **Stanowisko częściowo prawidłowe** (Position partially correct)
   - Częściowa zgoda z przedstawionym stanowiskiem

### Recommended Enumeration

**For Judgments (wyroki sądowe):**

```python
"decision_type": "enum (uwzgledniono_w_calosci/uwzgledniono_w_czesci/oddalono/umorzono/uchylono/uchylono_i_przekazano/zmieniono/utrzymano_w_mocy/odrzucono)"
```

**For Tax Interpretations:**

```python
"decision_type": "enum (stanowisko_prawidlowe/stanowisko_nieprawidlowe/stanowisko_czesciowo_prawidlowe)"
```

### Examples in Legal Documents

**Uwzględniono w całości:**

- "Sąd uwzględnia powództwo w całości i zasądza od pozwanego..."

**Oddalono:**

- "Sąd oddala powództwo jako bezzasadne"

**Uchylono i przekazano:**

- "Sąd Apelacyjny uchyla zaskarżony wyrok i przekazuje sprawę do ponownego rozpoznania Sądowi Okręgowemu"

**Utrzymano w mocy:**

- "Sąd Apelacyjny oddala apelację i utrzymuje w mocy wyrok Sądu Okręgowego"

### Verification Status

✅ **NEEDS UPDATE** - Current schema is incomplete and uses simplified terms

### Recommended Action

1. Expand enumeration to include all 9 types of court decisions
2. Add separate enumeration for tax interpretations
3. Use underscore-separated format for compound terms
4. Consider making decision_type conditional on document_type

---
_Generated: 2025-10-11_
_Source: Web search validation for JuDDGES extraction schema_
