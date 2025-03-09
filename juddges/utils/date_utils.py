from datetime import datetime

import pytz


def convert_date_to_rfc3339(date_str: str) -> str:
    """
    Convert date strings like '2012-11-27 14:45:14.0 CET' to RFC3339 format.

    Args:
        date_str: Date string in format 'YYYY-MM-DD HH:MM:SS.f TZ'

    Returns:
        str: Date string in RFC3339 format

    Examples:
        >>> convert_date_to_rfc3339('2012-11-27 14:45:14.0 CET')
        '2012-11-27T14:45:14.000000+01:00'
    """
    if not date_str:
        return None

    # Map timezone abbreviations to proper timezone names
    tz_map = {"CET": "Europe/Paris", "CEST": "Europe/Paris"}

    try:
        # Split date string and timezone
        date_part, tz_abbr = date_str.rsplit(" ", 1)

        # Parse the date part
        dt = datetime.strptime(date_part, "%Y-%m-%d %H:%M:%S.%f")

        # Get the proper timezone
        timezone = pytz.timezone(tz_map[tz_abbr])

        # Localize the datetime and convert to RFC3339
        dt_with_tz = timezone.localize(dt)
        return dt_with_tz.isoformat()
    except (ValueError, KeyError) as e:
        raise ValueError(
            f"Failed to convert date '{date_str}' to RFC3339 format: {str(e)}"
        )


def process_judgement_dates(judgement: dict) -> dict:
    """
    Process a judgement dictionary to convert all date fields to RFC3339.

    Args:
        judgement: Dictionary containing judgement data

    Returns:
        dict: Processed judgement with dates in RFC3339 format
    """
    date_fields = ["publication_date", "judgement_date", "last_update"]

    for field in date_fields:
        if field in judgement and judgement[field]:
            judgement[field] = convert_date_to_rfc3339(judgement[field])

    return judgement
