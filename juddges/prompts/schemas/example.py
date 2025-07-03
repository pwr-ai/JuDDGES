from textwrap import dedent

EXAMPLE_SCHEMA = dedent("""
    verdict_date: date as ISO 8601
    verdict: string, text representing verdict of the judgment
    verdict_summary: string, short summary of the verdict
    verdict_id: string
    court: string
    parties: string
    appeal_against: string
    first_trial: boolean
    drug_offence: boolean
    child_offence: boolean
    offence_seriousness: boolean
    verdict_tags: List[string]
""").strip()
