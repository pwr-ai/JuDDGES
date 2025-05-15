from datetime import datetime

from dateutil.parser import ParserError
from dateutil.parser import parse as dt_parse
from dateutil.tz import gettz

TZ_MAP = {"CET": gettz("Europe/Warsaw"), "CEST": gettz("Europe/Warsaw")}


def convert_date_to_rfc3339(date: str | datetime | None) -> str | None:
    """
    Convert date strings like '2012-11-27 14:45:14.0 CET' to RFC3339 format.

    Args:
        date_str: Date string in format 'YYYY-MM-DD HH:MM:SS.f TZ'

    Returns:
        str: Date string in RFC3339 format

    Examples:
        >>> convert_date_to_rfc3339('2012-11-27 14:45:14.0 CET')
        '2012-11-27T14:45:14+01:00'
    """
    if not date:
        return None

    if isinstance(date, str):
        try:
            date = dt_parse(date, tzinfos=TZ_MAP)
        except (ParserError, OverflowError, TypeError) as e:
            raise ValueError(f"Failed to convert date '{date}' to RFC3339 format: {str(e)}")
    elif not isinstance(date, datetime):
        raise TypeError(f"Invalid date type: {type(date)}")

    if not date.tzinfo:
        date = date.replace(tzinfo=TZ_MAP["CET"])

    return date.isoformat()


def process_judgment_dates(judgment: dict) -> dict:
    """
    Process a judgment dictionary to convert all date fields to RFC3339.

    Args:
        judgment: Dictionary containing judgment data

    Returns:
        dict: Processed judgment with dates in RFC3339 format
    """
    date_fields = ["publication_date", "judgment_date", "last_update"]

    for field in date_fields:
        if field in judgment and judgment[field]:
            judgment[field] = convert_date_to_rfc3339(judgment[field])

    return judgment
