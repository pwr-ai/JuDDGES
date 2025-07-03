"""
Smart column mapping using semantic similarity and pattern matching.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class MappingSuggestion:
    """A suggested column mapping with confidence score."""

    source_column: str
    target_field: str
    confidence: float
    reason: str


class SmartColumnMapper:
    """Intelligent column mapping using semantic similarity and patterns."""

    # Semantic mappings with variations and synonyms
    SEMANTIC_MAPPINGS = {
        "document_id": [
            "id",
            "document_id",
            "case_id",
            "judgment_id",
            "doc_id",
            "document_identifier",
            "case_identifier",
            "unique_id",
            "record_id",
            "file_id",
            "reference_id",
        ],
        "full_text": [
            "content",
            "text",
            "full_text",
            "body",
            "judgment_text",
            "document_text",
            "content_text",
            "main_text",
            "decision_text",
            "ruling_text",
            "opinion_text",
        ],
        "title": [
            "title",
            "name",
            "case_name",
            "heading",
            "subject",
            "caption",
            "case_title",
            "document_title",
            "matter_title",
        ],
        "summary": [
            "summary",
            "abstract",
            "excerpt",
            "synopsis",
            "overview",
            "brief",
            "description",
            "headnote",
            "syllabus",
        ],
        "thesis": [
            "thesis",
            "main_point",
            "key_point",
            "holding",
            "ratio",
            "principle",
            "legal_principle",
            "decision_point",
        ],
        "date_issued": [
            "date",
            "issued_date",
            "judgment_date",
            "decision_date",
            "ruling_date",
            "date_decided",
            "hearing_date",
            "publication_date",
            "created_date",
        ],
        "document_number": [
            "case_number",
            "docket_number",
            "file_number",
            "reference_number",
            "citation",
            "case_citation",
            "neutral_citation",
            "document_number",
        ],
        "court_name": [
            "court",
            "court_name",
            "tribunal",
            "issuing_court",
            "jurisdiction",
            "forum",
            "venue",
            "court_type",
            "issuing_authority",
        ],
        "judges": [
            "judges",
            "judge_names",
            "panel",
            "justices",
            "magistrates",
            "presiding_judge",
            "bench",
            "tribunal_members",
        ],
        "language": ["lang", "language", "locale", "language_code"],
        "country": ["country", "jurisdiction", "nation", "state", "territory"],
        "keywords": [
            "keywords",
            "tags",
            "categories",
            "subjects",
            "topics",
            "legal_areas",
            "practice_areas",
            "classifications",
        ],
        "legal_bases": [
            "legal_bases",
            "statutes",
            "regulations",
            "laws",
            "legal_references",
            "cited_law",
            "legal_provisions",
            "authorities",
        ],
    }

    # Patterns for detecting field types
    DATE_PATTERNS = [r"date", r"time", r"issued", r"created", r"modified", r"updated"]

    TEXT_PATTERNS = [r"text", r"content", r"body", r"description", r"summary", r"abstract"]

    ID_PATTERNS = [r"id", r"identifier", r"number", r"ref", r"key"]

    ARRAY_PATTERNS = [r"tags", r"keywords", r"categories", r"judges", r"authors", r"parties"]

    def __init__(self):
        """Initialize the smart mapper."""
        self._build_reverse_mapping()

    def _build_reverse_mapping(self) -> None:
        """Build reverse mapping for faster lookups."""
        self.reverse_mapping = {}
        for target_field, source_variations in self.SEMANTIC_MAPPINGS.items():
            for variation in source_variations:
                self.reverse_mapping[variation.lower()] = target_field

    def suggest_mapping(
        self, dataset_columns: List[str], required_fields: Optional[List[str]] = None
    ) -> Dict[str, MappingSuggestion]:
        """Suggest column mappings based on semantic similarity."""
        suggestions = {}
        used_targets = set()

        # Sort columns by priority (exact matches first)
        prioritized_columns = self._prioritize_columns(dataset_columns)

        for column in prioritized_columns:
            suggestion = self._find_best_match(column, used_targets)
            if suggestion and suggestion.confidence > 0.3:  # Minimum confidence threshold
                suggestions[column] = suggestion
                used_targets.add(suggestion.target_field)

        # Check for missing required fields
        if required_fields:
            missing = self._find_missing_required_fields(
                suggestions, required_fields, dataset_columns, used_targets
            )
            suggestions.update(missing)

        return suggestions

    def _prioritize_columns(self, columns: List[str]) -> List[str]:
        """Prioritize columns for mapping (exact matches first)."""
        exact_matches = []
        partial_matches = []
        others = []

        for col in columns:
            col_lower = col.lower()
            if col_lower in self.reverse_mapping:
                exact_matches.append(col)
            elif any(
                pattern in col_lower
                for patterns in self.SEMANTIC_MAPPINGS.values()
                for pattern in patterns
            ):
                partial_matches.append(col)
            else:
                others.append(col)

        return exact_matches + partial_matches + others

    def _find_best_match(self, column: str, used_targets: Set[str]) -> Optional[MappingSuggestion]:
        """Find the best target field match for a column."""
        col_lower = column.lower()

        # Check for exact match
        if col_lower in self.reverse_mapping:
            target = self.reverse_mapping[col_lower]
            if target not in used_targets:
                return MappingSuggestion(
                    source_column=column,
                    target_field=target,
                    confidence=1.0,
                    reason="Exact semantic match",
                )

        # Check for partial matches
        best_match = None
        best_score = 0.0

        for target_field, variations in self.SEMANTIC_MAPPINGS.items():
            if target_field in used_targets:
                continue

            score = self._calculate_similarity_score(col_lower, variations)
            if score > best_score:
                best_score = score
                best_match = MappingSuggestion(
                    source_column=column,
                    target_field=target_field,
                    confidence=score,
                    reason=f"Partial match with {target_field} variations",
                )

        # Check pattern-based matching
        if best_score < 0.7:  # Only if we don't have a good semantic match
            pattern_match = self._pattern_based_match(column, used_targets)
            if pattern_match and pattern_match.confidence > best_score:
                best_match = pattern_match

        return best_match

    def _calculate_similarity_score(self, column: str, variations: List[str]) -> float:
        """Calculate similarity score between column and target variations."""
        max_score = 0.0

        for variation in variations:
            # Exact match
            if column == variation.lower():
                return 1.0

            # Substring match
            if variation.lower() in column or column in variation.lower():
                score = min(len(variation), len(column)) / max(len(variation), len(column))
                max_score = max(max_score, score * 0.8)

            # Word boundary match
            if re.search(rf"\b{re.escape(variation.lower())}\b", column):
                max_score = max(max_score, 0.7)

            # Partial word match
            common_chars = set(column) & set(variation.lower())
            if len(common_chars) > 3:
                char_score = len(common_chars) / max(len(column), len(variation))
                max_score = max(max_score, char_score * 0.4)

        return max_score

    def _pattern_based_match(
        self, column: str, used_targets: Set[str]
    ) -> Optional[MappingSuggestion]:
        """Match based on common patterns in field names."""
        col_lower = column.lower()

        # Date field patterns
        if any(re.search(pattern, col_lower) for pattern in self.DATE_PATTERNS):
            if "date_issued" not in used_targets:
                return MappingSuggestion(
                    source_column=column,
                    target_field="date_issued",
                    confidence=0.6,
                    reason="Date pattern detected",
                )

        # Text field patterns
        if any(re.search(pattern, col_lower) for pattern in self.TEXT_PATTERNS):
            if "full_text" not in used_targets:
                return MappingSuggestion(
                    source_column=column,
                    target_field="full_text",
                    confidence=0.5,
                    reason="Text pattern detected",
                )

        # ID field patterns
        if any(re.search(pattern, col_lower) for pattern in self.ID_PATTERNS):
            if "document_id" not in used_targets:
                return MappingSuggestion(
                    source_column=column,
                    target_field="document_id",
                    confidence=0.5,
                    reason="ID pattern detected",
                )

        return None

    def _find_missing_required_fields(
        self,
        suggestions: Dict[str, MappingSuggestion],
        required_fields: List[str],
        dataset_columns: List[str],
        used_targets: Set[str],
    ) -> Dict[str, MappingSuggestion]:
        """Find mappings for missing required fields."""
        missing_suggestions = {}
        mapped_targets = {sugg.target_field for sugg in suggestions.values()}

        for required_field in required_fields:
            if required_field not in mapped_targets:
                # Try to find any unmapped column that could work
                for col in dataset_columns:
                    if col not in suggestions:
                        # Use lower confidence for required field fallbacks
                        missing_suggestions[col] = MappingSuggestion(
                            source_column=col,
                            target_field=required_field,
                            confidence=0.2,
                            reason=f"Fallback for required field {required_field}",
                        )
                        break

        return missing_suggestions

    def validate_mapping(
        self, mapping: Dict[str, str], required_fields: List[str]
    ) -> Tuple[bool, List[str]]:
        """Validate that required fields are mapped."""
        mapped_targets = set(mapping.values())
        missing = [field for field in required_fields if field not in mapped_targets]
        is_valid = len(missing) == 0
        return is_valid, missing

    def suggest_field_types(self, column_name: str, sample_values: List[any]) -> Dict[str, any]:
        """Suggest field configuration based on column name and sample values."""
        col_lower = column_name.lower()
        suggestions = {
            "is_text_field": False,
            "is_date_field": False,
            "is_array_field": False,
            "is_json_field": False,
            "should_vectorize": False,
        }

        # Check sample values
        if sample_values:
            first_non_null = next((v for v in sample_values if v is not None), None)

            if isinstance(first_non_null, list):
                suggestions["is_array_field"] = True
            elif isinstance(first_non_null, dict):
                suggestions["is_json_field"] = True
            elif isinstance(first_non_null, str):
                # Check if it looks like a date
                if any(re.search(pattern, col_lower) for pattern in self.DATE_PATTERNS):
                    suggestions["is_date_field"] = True
                # Check if it's a substantial text field
                elif len(first_non_null) > 100 or any(
                    re.search(pattern, col_lower) for pattern in self.TEXT_PATTERNS
                ):
                    suggestions["is_text_field"] = True
                    suggestions["should_vectorize"] = True

        # Pattern-based suggestions
        if any(re.search(pattern, col_lower) for pattern in self.DATE_PATTERNS):
            suggestions["is_date_field"] = True

        if any(re.search(pattern, col_lower) for pattern in self.ARRAY_PATTERNS):
            suggestions["is_array_field"] = True

        if col_lower in ["full_text", "content", "summary", "abstract", "thesis"]:
            suggestions["should_vectorize"] = True

        return suggestions
