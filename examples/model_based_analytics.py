#!/usr/bin/env python3
"""
Example script demonstrating model-based collection analytics.

This script shows how to use the data models to get structured analytics
without console logging, perfect for API endpoints or data processing.
"""

import json

from juddges.data.analytics_models import (
    AggregationType,
    ParallelAggregationRequest,
)
from juddges.data.stream_ingester import StreamingIngester


def example_basic_aggregations():
    """Example of basic individual aggregations."""
    print("=== Basic Aggregations ===")

    with StreamingIngester() as ingester:
        # Get individual aggregations as models
        doc_types = ingester.get_document_type_stats()
        countries = ingester.get_country_stats()

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


def example_typescript_style():
    """Example equivalent to TypeScript Promise.all() aggregations."""
    print("\n=== TypeScript-Style Parallel Aggregations ===")

    with StreamingIngester() as ingester:
        # TypeScript equivalent: const [docTypes, countries] = await Promise.all([...])
        response = ingester.get_typescript_style_aggregations()

        print(f"Execution time: {response.execution_time_ms:.2f}ms")
        print(f"Total documents: {response.total_documents:,}")

        # Access specific aggregations
        doc_types = response.get_aggregation(AggregationType.DOCUMENT_TYPE)
        countries = response.get_aggregation(AggregationType.COUNTRY)

        if doc_types:
            print(f"\nDocument Types: {doc_types.unique_values} types")
            for item in doc_types.get_top_items(2):
                print(f"  - {item.value}: {item.count:,}")

        if countries:
            print(f"\nCountries: {countries.unique_values} countries")
            for item in countries.get_top_items(2):
                print(f"  - {item.value}: {item.count:,}")


def example_custom_parallel_request():
    """Example of custom parallel aggregation request."""
    print("\n=== Custom Parallel Request ===")

    with StreamingIngester() as ingester:
        # Custom request - only specific aggregations, top 5 items each
        request = ParallelAggregationRequest(
            aggregation_types=[
                AggregationType.DOCUMENT_TYPE,
                AggregationType.LANGUAGE,
                AggregationType.COURT,
            ],
            include_date_range=True,
            top_n_limit=5,
            filter_empty_values=True,
        )

        response = ingester.get_parallel_aggregations(request)

        print(f"Custom aggregation completed in {response.execution_time_ms:.2f}ms")

        for agg_type, result in response.results.items():
            print(f"\n{agg_type.replace('_', ' ').title()}:")
            for item in result.items:
                print(f"  - {item.value}: {item.count:,}")


def example_comprehensive_analytics():
    """Example of comprehensive analytics model."""
    print("\n=== Comprehensive Analytics ===")

    with StreamingIngester() as ingester:
        analytics = ingester.get_comprehensive_analytics()

        # Get summary statistics
        summary = analytics.get_summary_stats()
        print(f"Total documents: {summary['total_documents']:,}")
        print(f"Unique document types: {summary['unique_document_types']}")
        print(f"Unique countries: {summary['unique_countries']}")

        # Most common items
        if summary["most_common_type"]:
            item = summary["most_common_type"]
            print(f"Most common type: {item.value} ({item.count:,} docs)")

        if summary["most_common_country"]:
            item = summary["most_common_country"]
            print(f"Most common country: {item.value} ({item.count:,} docs)")

        # Date range information
        if analytics.date_range and analytics.date_range.total_with_dates > 0:
            dr = analytics.date_range
            print(f"\nDate range: {dr.earliest_date} to {dr.latest_date}")
            print(f"Documents with dates: {dr.total_with_dates:,}")

            # Year distribution
            if dr.date_distribution:
                print("Top years:")
                sorted_years = sorted(
                    dr.date_distribution.items(), key=lambda x: x[1], reverse=True
                )
                for year, count in sorted_years[:3]:
                    print(f"  - {year}: {count:,} documents")


def example_json_export():
    """Example of exporting analytics as JSON."""
    print("\n=== JSON Export ===")

    with StreamingIngester() as ingester:
        analytics = ingester.get_comprehensive_analytics()

        # Convert to dictionary for JSON serialization
        data = analytics.to_dict()

        # Pretty print JSON
        print("Analytics as JSON (first 500 chars):")
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        print(json_str[:500] + "..." if len(json_str) > 500 else json_str)

        # You could save this to a file or send via API
        # with open('analytics.json', 'w') as f:
        #     json.dump(data, f, indent=2, ensure_ascii=False)


def example_api_style_usage():
    """Example showing how this would be used in an API endpoint."""
    print("\n=== API-Style Usage ===")

    def get_collection_stats_api():
        """Simulated API endpoint function."""
        try:
            with StreamingIngester() as ingester:
                # Quick stats for dashboard
                response = ingester.get_typescript_style_aggregations()

                return {
                    "success": True,
                    "data": {
                        "total_documents": response.total_documents,
                        "execution_time_ms": response.execution_time_ms,
                        "document_types": [
                            {"type": item.value, "count": item.count, "percentage": item.percentage}
                            for item in response.get_aggregation(
                                AggregationType.DOCUMENT_TYPE
                            ).get_top_items(5)
                        ]
                        if response.get_aggregation(AggregationType.DOCUMENT_TYPE)
                        else [],
                        "countries": [
                            {
                                "country": item.value,
                                "count": item.count,
                                "percentage": item.percentage,
                            }
                            for item in response.get_aggregation(
                                AggregationType.COUNTRY
                            ).get_top_items(5)
                        ]
                        if response.get_aggregation(AggregationType.COUNTRY)
                        else [],
                    },
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # Simulate API call
    api_response = get_collection_stats_api()
    print("API Response:")
    print(json.dumps(api_response, indent=2, ensure_ascii=False))


def main():
    """Run all examples."""
    print("🔍 Model-Based Analytics Examples")
    print("=" * 50)

    try:
        example_basic_aggregations()
        example_typescript_style()
        example_custom_parallel_request()
        example_comprehensive_analytics()
        example_json_export()
        example_api_style_usage()

        print("\n✅ All examples completed successfully!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Make sure:")
        print("1. Weaviate is running: docker-compose up -d")
        print("2. You have documents ingested")
        print("3. Check connection settings")


if __name__ == "__main__":
    main()
