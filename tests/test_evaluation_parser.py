from unittest import TestCase

from juddges.evaluation.parse import EMPTY_ANSWER, _parse_item


class TestParseItem(TestCase):
    def test_invalid_yaml(self):
        invalid_yaml = "it's not a yaml"
        self.assertIsNone(_parse_item(invalid_yaml))

    def test_empty_yaml(self):
        empty_yaml = "```yaml\n```"
        self.assertIsNone(_parse_item(empty_yaml))

    def test_yaml_is_list_instead_of_dict(self):
        invalid_list_yaml = """
        ```yaml
            - first
            - second
            - third
        ```
        """
        self.assertIsNone(_parse_item(invalid_list_yaml))

    def test_yaml_with_preceeding_text(self):
        yaml_with_text = """
        Here is the extracted information in YAML format:
        ```yaml
            name: John
        ```
        Text after YAML
        """
        target_output = {"name": "John"}
        self.assertDictEqual(_parse_item(yaml_with_text), target_output)

    def test_several_yamls(self):
        yaml_with_text = """
        Here is the extracted information in YAML format:
        ```yaml
            name: John
        ```
        Here is alternative:
        ```yaml
            name: Jack
        ```
        """
        target_output = {"name": "John"}
        _parse_item(yaml_with_text)
        self.assertDictEqual(_parse_item(yaml_with_text), target_output)

    def test_list_field_casted_to_dict_and_sorted(self):
        valid_list_yaml = """
        ```yaml
            judge:
                - c
                - b
                - a
        ```
        """
        target_output = {"judge": "a, b, c"}
        self.assertDictEqual(_parse_item(valid_list_yaml), target_output)

    def test_yaml_parsing_without_yaml_header(self):
        valid_list_yaml = """
        judge:
            - c
            - b
            - a
        """
        target_output = {"judge": "a, b, c"}
        self.assertDictEqual(_parse_item(valid_list_yaml), target_output)

    def test_date_field_format(self):
        date_yaml = """
        ```yaml
            date: 2024-08-13T10:00:00
        ```
        """
        target_output = {"date": "2024-08-13"}
        self.assertDictEqual(_parse_item(date_yaml), target_output)

    def test_null_field(self):
        null_yaml = """
        ```yaml
            field: null
        ```
        """
        target_output = {"field": EMPTY_ANSWER}
        self.assertDictEqual(_parse_item(null_yaml), target_output)

    def test_string_output(self):
        yaml = """
        ```yaml
            judge: a
            id_number: 10
            no_date: 2024-10
        ```
        """
        target_output = {"judge": "a", "id_number": "10", "no_date": "2024-10"}
        self.assertDictEqual(_parse_item(yaml), target_output)

    def test_multiple_fields(self):
        yaml = """
        ```yaml
            judge: a
            date: 2024-08-13T10:00:00
            participants:
                - a
                - d
                - b
            other: null
        ```
        """
        target_output = {
            "judge": "a",
            "date": "2024-08-13",
            "participants": "a, b, d",
            "other": EMPTY_ANSWER,
        }
        self.assertDictEqual(_parse_item(yaml), target_output)
