#!/usr/bin/env python3
"""
Example script demonstrating model-based collection analytics.

This script shows how to use the data models to get structured analytics
without console logging, perfect for API endpoints or data processing.
"""

import json
from datetime import datetime
from typing import Dict

from dotenv import load_dotenv

from juddges.data.analytics_models import (
    AggregationItem,
    AggregationResult,
    AggregationType,
)
from juddges.data.stream_ingester import StreamingIngester

load_dotenv(".env", override=True)


def dict_to_aggregation_result(
    stats_dict: Dict[str, int], aggregation_type: AggregationType
) -> AggregationResult:
    """Convert dictionary stats to AggregationResult model."""
    if not stats_dict:
        return AggregationResult(
            aggregation_type=aggregation_type,
            items=[],
            total_count=0,
            unique_values=0,
        )

    total_count = sum(stats_dict.values())
    items = []

    for value, count in stats_dict.items():
        percentage = (count / total_count * 100) if total_count > 0 else 0.0
        items.append(AggregationItem(value=value, count=count, percentage=percentage))

    return AggregationResult(
        aggregation_type=aggregation_type,
        items=items,
        total_count=total_count,
        unique_values=len(stats_dict),
    )


def example_basic_aggregations():
    """Example of basic individual aggregations."""
    print("=== Basic Aggregations ===")

    with StreamingIngester() as ingester:
        # Get raw dictionary stats and convert to models
        doc_types_dict = ingester.get_document_type_stats()
        countries_dict = ingester.get_country_stats()

        # Convert to structured models
        doc_types = dict_to_aggregation_result(doc_types_dict, AggregationType.DOCUMENT_TYPE)
        countries = dict_to_aggregation_result(countries_dict, AggregationType.COUNTRY)

        # Access structured data
        print(f"Document Types Found: {doc_types.unique_values}")
        print(f"Total Documents: {doc_types.total_count}")

        # Get top items
        top_doc_types = doc_types.get_top_items(3)
        for item in top_doc_types:
            print(f"  - {item.value}: {item.count} ({item.percentage:.1f}%)")

        print(f"\nCountries Found: {countries.unique_values}")
        top_countries = countries.get_top_items(3)
        for item in top_countries:
            print(f"  - {item.value}: {item.count} ({item.percentage:.1f}%)")


def example_api_style_usage():
    """Example showing how this would be used in an API endpoint."""
    print("\n=== API-Style Usage ===")

    def get_collection_stats_api():
        """Simulated API endpoint function."""
        try:
            with StreamingIngester() as ingester:
                # Get raw stats
                doc_types_dict = ingester.get_document_type_stats()
                countries_dict = ingester.get_country_stats()

                # Convert to models
                doc_types = dict_to_aggregation_result(
                    doc_types_dict, AggregationType.DOCUMENT_TYPE
                )
                countries = dict_to_aggregation_result(countries_dict, AggregationType.COUNTRY)

                total_documents = doc_types.total_count

                return {
                    "success": True,
                    "data": {
                        "total_documents": total_documents,
                        "document_types": [
                            {"type": item.value, "count": item.count, "percentage": item.percentage}
                            for item in doc_types.get_top_items(5)
                        ],
                        "countries": [
                            {
                                "country": item.value,
                                "count": item.count,
                                "percentage": item.percentage,
                            }
                            for item in countries.get_top_items(5)
                        ],
                    },
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # Simulate API call
    api_response = get_collection_stats_api()
    print("API Response:")
    print(json.dumps(api_response, indent=2, ensure_ascii=False))


def example_json_export():
    """Example of exporting analytics as JSON."""
    print("\n=== JSON Export ===")

    with StreamingIngester() as ingester:
        # Get and convert stats
        doc_types_dict = ingester.get_document_type_stats()
        countries_dict = ingester.get_country_stats()

        doc_types = dict_to_aggregation_result(doc_types_dict, AggregationType.DOCUMENT_TYPE)
        countries = dict_to_aggregation_result(countries_dict, AggregationType.COUNTRY)

        # Create analytics summary
        analytics_data = {
            "total_documents": doc_types.total_count,
            "generated_at": datetime.now().isoformat(),
            "aggregations": {
                "document_types": doc_types.to_dict(),
                "countries": countries.to_dict(),
            },
            "summary": {
                "unique_document_types": doc_types.unique_values,
                "unique_countries": countries.unique_values,
                "most_common_type": doc_types.get_top_items(1)[0].value
                if doc_types.items
                else None,
                "most_common_country": countries.get_top_items(1)[0].value
                if countries.items
                else None,
            },
        }

        # Pretty print JSON
        print("Analytics as JSON (first 500 chars):")
        json_str = json.dumps(analytics_data, indent=2, ensure_ascii=False)
        print(json_str[:500] + "..." if len(json_str) > 500 else json_str)


def main():
    """Run all examples."""
    print("🔍 Model-Based Analytics Examples")
    print("=" * 50)

    try:
        example_basic_aggregations()
        # example_api_style_usage()
        # example_json_export()

        print("\n✅ All examples completed successfully!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Make sure:")
        print("1. Weaviate is running: docker-compose up -d")
        print("2. You have documents ingested")
        print("3. Check connection settings")


if __name__ == "__main__":
    main()
