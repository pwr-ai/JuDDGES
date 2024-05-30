from enum import Enum


class VersionBump(str, Enum):
    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"


def bump_version(version: str, bump: VersionBump) -> str:
    major, minor, patch = map(int, version.split("."))
    if bump == VersionBump.MAJOR:
        major += 1
        minor = 0
        patch = 0
    elif bump == VersionBump.MINOR:
        minor += 1
        patch = 0
    elif bump == VersionBump.PATCH:
        patch += 1
    return f"{major}.{minor}.{patch}"
