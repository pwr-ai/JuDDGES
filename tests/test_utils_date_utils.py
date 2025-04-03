import pytest

from juddges.utils.date_utils import convert_date_to_rfc3339


def test_valid_date_with_cet_timezone():
    """Test conversion of a valid date string with CET timezone"""
    date_str = "2012-11-27 14:45:14.0 CET"
    expected = "2012-11-27T14:45:14+01:00"
    assert convert_date_to_rfc3339(date_str) == expected


def test_valid_date_with_cest_timezone():
    """Test conversion of a valid date string with CEST timezone"""
    date_str = "2012-06-27 14:45:14.0 CEST"
    expected = "2012-06-27T14:45:14+02:00"
    assert convert_date_to_rfc3339(date_str) == expected


def test_valid_date_without_timezone_defaults_to_cet():
    """Test conversion of a valid date string without timezone"""
    date_str = "2012-11-27 14:45:14"
    expected = "2012-11-27T14:45:14+01:00"
    assert convert_date_to_rfc3339(date_str) == expected


def test_date_only():
    """Test conversion of a valid date string without time"""
    date_str = "2012-11-27"
    expected = "2012-11-27T00:00:00+01:00"
    assert convert_date_to_rfc3339(date_str) == expected


@pytest.mark.parametrize(
    "date_str",
    [
        "",
        None,
    ],
)
def test_empty_date(date_str):
    """Test handling of empty date input"""
    assert convert_date_to_rfc3339(date_str) is None


@pytest.mark.parametrize(
    "invalid_date",
    [
        "not a date",
        "2012-13-45",  # invalid month and day
    ],
)
def test_invalid_date_format(invalid_date):
    """Test handling of invalid date format"""
    with pytest.raises(ValueError) as exc_info:
        convert_date_to_rfc3339(invalid_date)
    assert "Failed to convert date" in str(exc_info.value)
