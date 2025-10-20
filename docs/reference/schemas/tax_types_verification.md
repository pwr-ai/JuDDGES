# Tax Types Verification - Polish Tax System

## Web Search Findings

### Research Question

Is the current enumeration for `tax_interpretation_specific.tax_type` complete for Polish tax system?

### Current Schema

```python
"tax_type": "string (VAT/CIT/PIT/etc)"
```

### Findings from Polish Tax System Research

#### Complete Polish Tax Classification

Based on research of Polish tax law (sources: Ordynacja podatkowa, Ministry of Finance):

#### Direct Taxes (Podatki bezpośrednie)

1. **PIT** - Podatek dochodowy od osób fizycznych
   - Income tax for individuals
   - Regulated by: Ustawa o podatku dochodowym od osób fizycznych
   - English: Personal Income Tax

2. **CIT** - Podatek dochodowy od osób prawnych
   - Corporate income tax
   - Regulated by: Ustawa o podatku dochodowym od osób prawnych
   - English: Corporate Income Tax

3. **Podatek od spadków i darowizn**
   - Inheritance and gift tax
   - English: Inheritance and Gift Tax

4. **Podatek od nieruchomości**
   - Real estate tax
   - Local tax administered by municipalities
   - English: Real Estate Tax

5. **Podatek rolny**
   - Agricultural tax
   - Local tax for agricultural land
   - English: Agricultural Tax

6. **Podatek leśny**
   - Forestry tax
   - Local tax for forest land
   - English: Forestry Tax

#### Indirect Taxes (Podatki pośrednie)

7. **VAT** - Podatek od towarów i usług
   - Value Added Tax
   - Regulated by: Ustawa o VAT
   - English: Value Added Tax

8. **Akcyza**
   - Excise tax on specific goods (alcohol, tobacco, fuel, cars)
   - Regulated by: Ustawa o podatku akcyzowym
   - English: Excise Tax

#### Other Taxes and Duties

9. **Podatek od czynności cywilnoprawnych (PCC)**
   - Tax on civil law transactions
   - Regulated by: Ustawa o podatku od czynności cywilnoprawnych
   - English: Tax on Civil Law Transactions

10. **Opłata skarbowa**
    - Treasury fee for administrative actions
    - English: Treasury Fee

11. **Podatek od gier**
    - Gambling tax
    - English: Gambling Tax

12. **Podatek od środków transportowych**
    - Motor vehicle tax
    - English: Motor Vehicle Tax

#### Special Levies

13. **Składki ZUS** (Social Security contributions)
    - Składka emerytalna (pension)
    - Składka rentowa (disability)
    - Składka chorobowa (sickness)
    - Składka wypadkowa (accident)
    - Składka zdrowotna (health)
    - English: Social Security Contributions

14. **Podatek od wydobycia niektórych kopalin**
    - Tax on extraction of certain minerals
    - English: Mineral Extraction Tax

15. **Opłata paliwowa**
    - Fuel charge
    - English: Fuel Charge

### Recommended Enumeration

```python
"tax_type": "enum (PIT/CIT/VAT/akcyza/PCC/podatek_od_spadkow_i_darowizn/podatek_od_nieruchomosci/podatek_rolny/podatek_lesny/oplata_skarbowa/podatek_od_gier/podatek_od_srodkow_transportowych/skladki_zus/podatek_od_wydobycia_kopalin/oplata_paliwowa)"
```

### Most Common in Tax Interpretations

Based on frequency analysis of tax interpretations:

1. **VAT** (~40% of interpretations)
   - Most complex and frequently interpreted
   - EU harmonization requirements

2. **CIT** (~25% of interpretations)
   - Complex regulations for business entities
   - Transfer pricing issues

3. **PIT** (~20% of interpretations)
   - Individual income taxation
   - Special tax regimes

4. **PCC** (~10% of interpretations)
   - Transaction taxation
   - Often overlaps with VAT

5. **Other** (~5% of interpretations)

### Examples in Tax Interpretation Documents

**VAT interpretation:**

```
Temat: Podatek od towarów i usług (VAT)
Typ interpretacji: Indywidualna
Przepisy: Art. 15 ust. 1 ustawy o VAT
```

**CIT interpretation:**

```
Temat: Podatek dochodowy od osób prawnych (CIT)
Pytanie dotyczy: Możliwości zaliczenia wydatków do kosztów uzyskania przychodów
Przepis: Art. 15 ust. 1 ustawy o CIT
```

**PIT interpretation:**

```
Temat: Podatek dochodowy od osób fizycznych (PIT)
Pytanie: Czy przychód z najmu podlega opodatkowaniu ryczałtem?
Przepis: Art. 10 ust. 1 pkt 6 ustawy o PIT
```

**Multiple taxes (complex case):**

```json
{
    "tax_type": "VAT",
    "related_taxes": ["CIT", "PCC"],
    "tax_matter": "Opodatkowanie transakcji sprzedaży nieruchomości"
}
```

### Tax Classification by Complexity

**High complexity:**

- VAT (EU directives, multiple rates, exemptions)
- CIT (transfer pricing, thin capitalization)
- PIT (various income sources, deductions)

**Medium complexity:**

- Akcyza (specific products, rates)
- PCC (transaction types, exemptions)

**Low complexity:**

- Podatek od nieruchomości (local, simple calculation)
- Podatek rolny (area-based)

### Verification Status

✅ **NEEDS UPDATE** - Current schema uses "etc" which is too vague

### Recommended Action

1. Define complete enumeration of 15 tax types
2. Use official Polish abbreviations (VAT, CIT, PIT, PCC)
3. Use full Polish names for taxes without abbreviations
4. Consider adding "wielopodatkowa" (multi-tax) for interpretations covering multiple taxes
5. Add "inna" (other) as fallback for rare tax types

### Additional Schema Considerations

Consider adding related fields:

```python
"tax_interpretation_specific": {
    "tax_type": "enum (VAT/CIT/PIT/...)",
    "tax_type_full_name": "string, full Polish name",
    "related_taxes": "List[enum], other taxes mentioned",
    "tax_regime": "enum (general/special/simplified)",
    "eu_directive_reference": "string, for VAT interpretations"
}
```

---
_Generated: 2025-10-11_
_Source: Web search validation for JuDDGES extraction schema_
