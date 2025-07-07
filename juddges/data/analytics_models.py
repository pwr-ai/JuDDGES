"""
Data models for collection analytics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class AggregationType(Enum):
    """Types of aggregations available."""

    DOCUMENT_TYPE = "document_type"
    COUNTRY = "country"
    LANGUAGE = "language"
    COURT = "court_name"
    PROCESSING_STATUS = "processing_status"
    ISSUING_BODY = "issuing_body"


@dataclass
class AggregationItem:
    """Single aggregation result item."""

    value: str
    count: int
    percentage: float = 0.0


@dataclass
class AggregationResult:
    """Result of a single aggregation query."""

    aggregation_type: AggregationType
    items: List[AggregationItem]
    total_count: int
    unique_values: int
    generated_at: datetime = field(default_factory=datetime.now)

    def get_top_items(self, limit: int = 10) -> List[AggregationItem]:
        """Get top N items by count."""
        return sorted(self.items, key=lambda x: x.count, reverse=True)[:limit]

    def get_item_by_value(self, value: str) -> Optional[AggregationItem]:
        """Get specific item by value."""
        for item in self.items:
            if item.value == value:
                return item
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "aggregation_type": self.aggregation_type.value,
            "items": [
                {"value": item.value, "count": item.count, "percentage": item.percentage}
                for item in self.items
            ],
            "total_count": self.total_count,
            "unique_values": self.unique_values,
            "generated_at": self.generated_at.isoformat(),
        }


@dataclass
class DateRangeStats:
    """Statistics about document date ranges."""

    earliest_date: Optional[str]
    latest_date: Optional[str]
    total_with_dates: int
    sample_dates: List[str] = field(default_factory=list)
    date_distribution: Dict[str, int] = field(default_factory=dict)  # Year -> count

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "earliest_date": self.earliest_date,
            "latest_date": self.latest_date,
            "total_with_dates": self.total_with_dates,
            "sample_dates": self.sample_dates,
            "date_distribution": self.date_distribution,
        }


@dataclass
class CollectionAnalytics:
    """Comprehensive collection analytics data model."""

    total_documents: int
    document_types: AggregationResult
    countries: AggregationResult
    languages: AggregationResult
    courts: AggregationResult
    processing_status: AggregationResult
    issuing_bodies: Optional[AggregationResult] = None
    date_range: Optional[DateRangeStats] = None
    collection_name: str = "LegalDocuments"
    generated_at: datetime = field(default_factory=datetime.now)

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get high-level summary statistics."""
        return {
            "total_documents": self.total_documents,
            "unique_document_types": self.document_types.unique_values,
            "unique_countries": self.countries.unique_values,
            "unique_languages": self.languages.unique_values,
            "unique_courts": self.courts.unique_values,
            "most_common_type": self.document_types.get_top_items(1)[0]
            if self.document_types.items
            else None,
            "most_common_country": self.countries.get_top_items(1)[0]
            if self.countries.items
            else None,
            "most_common_language": self.languages.get_top_items(1)[0]
            if self.languages.items
            else None,
        }

    def get_distribution_by_type(
        self, aggregation_type: AggregationType
    ) -> Optional[AggregationResult]:
        """Get distribution for specific aggregation type."""
        mapping = {
            AggregationType.DOCUMENT_TYPE: self.document_types,
            AggregationType.COUNTRY: self.countries,
            AggregationType.LANGUAGE: self.languages,
            AggregationType.COURT: self.courts,
            AggregationType.PROCESSING_STATUS: self.processing_status,
            AggregationType.ISSUING_BODY: self.issuing_bodies,
        }
        return mapping.get(aggregation_type)

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire analytics to dictionary."""
        result = {
            "total_documents": self.total_documents,
            "collection_name": self.collection_name,
            "generated_at": self.generated_at.isoformat(),
            "aggregations": {
                "document_types": self.document_types.to_dict(),
                "countries": self.countries.to_dict(),
                "languages": self.languages.to_dict(),
                "courts": self.courts.to_dict(),
                "processing_status": self.processing_status.to_dict(),
            },
            "summary": self.get_summary_stats(),
        }

        if self.issuing_bodies:
            result["aggregations"]["issuing_bodies"] = self.issuing_bodies.to_dict()

        if self.date_range:
            result["date_range"] = self.date_range.to_dict()

        return result


@dataclass
class ParallelAggregationRequest:
    """Request model for parallel aggregations."""

    aggregation_types: List[AggregationType]
    include_date_range: bool = False
    top_n_limit: Optional[int] = None
    filter_empty_values: bool = True


@dataclass
class ParallelAggregationResponse:
    """Response model for parallel aggregations."""

    results: Dict[str, AggregationResult]
    total_documents: int
    execution_time_ms: float
    date_range: Optional[DateRangeStats] = None

    def get_aggregation(self, aggregation_type: AggregationType) -> Optional[AggregationResult]:
        """Get specific aggregation result."""
        return self.results.get(aggregation_type.value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "total_documents": self.total_documents,
            "execution_time_ms": self.execution_time_ms,
            "results": {key: aggregation.to_dict() for key, aggregation in self.results.items()},
        }

        if self.date_range:
            result["date_range"] = self.date_range.to_dict()

        return result
