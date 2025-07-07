#!/usr/bin/env python3
"""
Example showing how to format analytics data for REST API responses.

This demonstrates the exact format you'd return from endpoints similar to your TypeScript queries.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from juddges.data.stream_ingester import StreamingIngester
from juddges.data.analytics_models import AggregationType, ParallelAggregationRequest


def format_for_typescript_client(ingester: StreamingIngester) -> Dict[str, Any]:
    """
    Format data exactly as your TypeScript client expects.
    
    Equivalent to TypeScript:
    const [documentTypeStats, countryStats] = await Promise.all([...])
    """
    response = ingester.get_typescript_style_aggregations()
    
    # Format document type stats
    document_type_stats = response.get_aggregation(AggregationType.DOCUMENT_TYPE)
    document_types_formatted = {
        "groups": [
            {
                "groupedBy": {"value": item.value},
                "totalCount": item.count,
                "percentage": round(item.percentage, 1)
            }
            for item in document_type_stats.items
        ] if document_type_stats else []
    }
    
    # Format country stats
    country_stats = response.get_aggregation(AggregationType.COUNTRY)
    countries_formatted = {
        "groups": [
            {
                "groupedBy": {"value": item.value},
                "totalCount": item.count,
                "percentage": round(item.percentage, 1)
            }
            for item in country_stats.items
        ] if country_stats else []
    }
    
    return {
        "documentTypeStats": document_types_formatted,
        "countryStats": countries_formatted,
        "metadata": {
            "totalDocuments": response.total_documents,
            "executionTimeMs": round(response.execution_time_ms, 2),
            "generatedAt": response.date_range.generated_at.isoformat() if response.date_range else None
        }
    }


def create_dashboard_api_response(ingester: StreamingIngester) -> Dict[str, Any]:
    """Create a comprehensive dashboard API response."""
    analytics = ingester.get_comprehensive_analytics()
    
    return {
        "success": True,
        "data": {
            "overview": {
                "totalDocuments": analytics.total_documents,
                "documentTypes": analytics.document_types.unique_values,
                "countries": analytics.countries.unique_values,
                "languages": analytics.languages.unique_values,
                "courts": analytics.courts.unique_values,
                "generatedAt": analytics.generated_at.isoformat()
            },
            "distributions": {
                "documentTypes": [
                    {
                        "name": item.value,
                        "value": item.count,
                        "percentage": round(item.percentage, 1)
                    }
                    for item in analytics.document_types.get_top_items(10)
                ],
                "countries": [
                    {
                        "name": item.value,
                        "value": item.count,
                        "percentage": round(item.percentage, 1)
                    }
                    for item in analytics.countries.get_top_items(10)
                ],
                "languages": [
                    {
                        "name": item.value,
                        "value": item.count,
                        "percentage": round(item.percentage, 1)
                    }
                    for item in analytics.languages.get_top_items()
                ],
                "topCourts": [
                    {
                        "name": item.value,
                        "value": item.count,
                        "percentage": round(item.percentage, 1)
                    }
                    for item in analytics.courts.get_top_items(5)
                ]
            },
            "dateRange": {
                "earliest": analytics.date_range.earliest_date,
                "latest": analytics.date_range.latest_date,
                "documentsWithDates": analytics.date_range.total_with_dates,
                "yearDistribution": analytics.date_range.date_distribution
            } if analytics.date_range else None
        }
    }


def create_chart_data_format(ingester: StreamingIngester) -> Dict[str, Any]:
    """Create data formatted for chart libraries (Chart.js, D3, etc.)."""
    analytics = ingester.get_comprehensive_analytics()
    
    # Pie chart data for document types
    doc_types_chart = {
        "labels": [item.value for item in analytics.document_types.get_top_items(8)],
        "datasets": [{
            "data": [item.count for item in analytics.document_types.get_top_items(8)],
            "backgroundColor": [
                "#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0", 
                "#9966FF", "#FF9F40", "#FF6384", "#C9CBCF"
            ]
        }]
    }
    
    # Bar chart data for countries
    top_countries = analytics.countries.get_top_items(10)
    countries_chart = {
        "labels": [item.value for item in top_countries],
        "datasets": [{
            "label": "Documents by Country",
            "data": [item.count for item in top_countries],
            "backgroundColor": "#36A2EB"
        }]
    }
    
    # Time series data (by year)
    year_data = []
    if analytics.date_range and analytics.date_range.date_distribution:
        sorted_years = sorted(analytics.date_range.date_distribution.items())
        year_data = {
            "labels": [year for year, _ in sorted_years],
            "datasets": [{
                "label": "Documents by Year",
                "data": [count for _, count in sorted_years],
                "borderColor": "#4BC0C0",
                "fill": False
            }]
        }
    
    return {
        "pieCharts": {
            "documentTypes": doc_types_chart
        },
        "barCharts": {
            "countries": countries_chart
        },
        "lineCharts": {
            "documentsByYear": year_data
        }
    }


def create_minimal_api_response(ingester: StreamingIngester) -> Dict[str, Any]:
    """Create a minimal API response for lightweight requests."""
    request = ParallelAggregationRequest(
        aggregation_types=[AggregationType.DOCUMENT_TYPE, AggregationType.COUNTRY],
        top_n_limit=5,
        filter_empty_values=True
    )
    response = ingester.get_parallel_aggregations(request)
    
    return {
        "totalDocuments": response.total_documents,
        "executionTime": f"{response.execution_time_ms:.0f}ms",
        "topDocumentTypes": [
            {"type": item.value, "count": item.count}
            for item in response.get_aggregation(AggregationType.DOCUMENT_TYPE).get_top_items(3)
        ] if response.get_aggregation(AggregationType.DOCUMENT_TYPE) else [],
        "topCountries": [
            {"country": item.value, "count": item.count}
            for item in response.get_aggregation(AggregationType.COUNTRY).get_top_items(3)
        ] if response.get_aggregation(AggregationType.COUNTRY) else []
    }


def main():
    """Demonstrate different API response formats."""
    print("🚀 API Response Format Examples")
    print("=" * 40)
    
    try:
        with StreamingIngester() as ingester:
            
            print("1. TypeScript Client Format:")
            ts_format = format_for_typescript_client(ingester)
            print(json.dumps(ts_format, indent=2)[:500] + "...")
            
            print("\n2. Dashboard API Format:")
            dashboard_format = create_dashboard_api_response(ingester)
            print(json.dumps(dashboard_format, indent=2)[:500] + "...")
            
            print("\n3. Chart Data Format:")
            chart_format = create_chart_data_format(ingester)
            print(json.dumps(chart_format, indent=2)[:500] + "...")
            
            print("\n4. Minimal API Format:")
            minimal_format = create_minimal_api_response(ingester)
            print(json.dumps(minimal_format, indent=2))
            
            print("\n✅ All API formats generated successfully!")
            
            # Example of saving to files for documentation
            formats = {
                "typescript_client.json": ts_format,
                "dashboard_api.json": dashboard_format,
                "chart_data.json": chart_format,
                "minimal_api.json": minimal_format
            }
            
            print(f"\n📁 Generated {len(formats)} API response examples")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure Weaviate is running and has data")


if __name__ == "__main__":
    main()