import re
import yaml

yaml_pattern: re.Pattern = re.compile(r"^```(?:ya?ml)?(?P<yaml>[^`]*)", re.MULTILINE | re.DOTALL)


def parse_yaml(text: str):
    """YAML parser taken from langchaing.
    Credit: https://github.com/langchain-ai/langchain.
    """
    match = re.search(yaml_pattern, text.strip())
    yaml_str = ""
    if match:
        yaml_str = match.group("yaml")
    else:
        yaml_str = text

    return yaml.safe_load(yaml_str)
