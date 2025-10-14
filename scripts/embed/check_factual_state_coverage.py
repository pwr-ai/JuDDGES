"""Check how many documents in Weaviate have non-empty factual_state property."""

from loguru import logger
from rich.console import Console

from juddges.data.documents_weaviate_db import WeaviateLegalDocumentsDatabase

console = Console()


def check_factual_state_coverage(db: WeaviateLegalDocumentsDatabase) -> dict:
    """Check factual_state field coverage in the LegalDocuments collection.

    Args:
        db: Weaviate database connection

    Returns:
        Dict with coverage statistics for factual_state field
    """
    collection = db.legal_documents_collection

    # Get total count
    logger.info("Getting total document count...")
    total_response = collection.aggregate.over_all(total_count=True)
    total_count = total_response.total_count

    logger.info(f"Total documents in collection: {total_count:,}")

    # Query documents with factual_state field
    # We'll need to iterate through all documents to check this
    logger.info("Fetching all documents to check factual_state field...")

    populated_count = 0
    empty_count = 0
    batch_size = 1000
    offset = 0

    while offset < total_count:
        try:
            response = collection.query.fetch_objects(
                limit=batch_size,
                offset=offset,
                return_properties=["document_id", "factual_state"]
            )

            for obj in response.objects:
                factual_state = obj.properties.get("factual_state")

                # Check if factual_state is populated (non-null and non-empty string)
                if factual_state and isinstance(factual_state, str) and factual_state.strip():
                    populated_count += 1
                else:
                    empty_count += 1

            offset += batch_size
            logger.info(f"Processed {offset:,} / {total_count:,} documents...")

        except Exception as e:
            logger.error(f"Error fetching documents at offset {offset}: {e}")
            break

    total_checked = populated_count + empty_count
    coverage_percentage = (populated_count / total_checked * 100) if total_checked > 0 else 0.0

    return {
        "total_documents": total_count,
        "documents_checked": total_checked,
        "populated_count": populated_count,
        "empty_count": empty_count,
        "coverage_percentage": round(coverage_percentage, 2),
    }


def main():
    """Main execution function."""
    logger.info("Connecting to Weaviate...")

    with WeaviateLegalDocumentsDatabase() as db:
        try:
            stats = check_factual_state_coverage(db)

            # Print results
            console.print("\n[bold cyan]Factual State Field Coverage[/bold cyan]\n")
            console.print(f"[bold]Total documents in collection:[/bold] {stats['total_documents']:,}")
            console.print(f"[bold]Documents checked:[/bold] {stats['documents_checked']:,}")
            console.print(f"[green]Documents with factual_state:[/green] {stats['populated_count']:,}")
            console.print(f"[red]Documents without factual_state:[/red] {stats['empty_count']:,}")
            console.print(f"[yellow]Coverage percentage:[/yellow] {stats['coverage_percentage']:.2f}%")

        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            raise


if __name__ == "__main__":
    main()
