#!/usr/bin/env python3
"""
Native Weaviate backup using the filesystem backup module.

Preserves data + vectors + HNSW indices. Recovery takes minutes, not hours.

Usage:
    python backup_native.py                          # Auto-named backup
    python backup_native.py --backup-id my-backup    # Custom name
    python backup_native.py --list                   # List existing backups
    python backup_native.py --restore backup-20260324-120000  # Restore

Requires:
    - ENABLE_MODULES includes 'backup-filesystem' in weaviate .env
    - BACKUP_FILESYSTEM_PATH set and mounted in docker-compose
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import weaviate
from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.table import Table

console = Console()

# Load environment
load_dotenv(Path(__file__).parent / ".env", override=False)
load_dotenv(Path(__file__).parent.parent / ".env", override=False)

WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", "8084"))
# Docker maps host 8085 -> container 50051; env may have the container port
WEAVIATE_GRPC_PORT = 8085 if WEAVIATE_HOST in ("localhost", "127.0.0.1") else int(os.getenv("WEAVIATE_GRPC_PORT", "8085"))
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")
BACKUP_PATH = Path("/mnt/readynas/datasets/legal-ai-weaviate/native-backups")


def get_client() -> weaviate.WeaviateClient:
    """Create Weaviate client."""
    auth = weaviate.auth.AuthApiKey(api_key=WEAVIATE_API_KEY) if WEAVIATE_API_KEY else None
    client = weaviate.connect_to_custom(
        http_host=WEAVIATE_HOST,
        http_port=WEAVIATE_PORT,
        http_secure=False,
        grpc_host=WEAVIATE_HOST,
        grpc_port=WEAVIATE_GRPC_PORT,
        grpc_secure=False,
        auth_credentials=auth,
        skip_init_checks=False,
    )
    return client


def create_backup(client: weaviate.WeaviateClient, backup_id: str) -> None:
    """Create a native filesystem backup."""
    console.print(f"\n[bold cyan]Creating backup:[/bold cyan] {backup_id}")
    console.print(f"Backend: filesystem -> {BACKUP_PATH}")

    # Show collection sizes before backup
    for name in ["LegalDocuments", "DocumentChunks"]:
        try:
            collection = client.collections.get(name)
            count = collection.aggregate.over_all(total_count=True).total_count or 0
            console.print(f"  {name}: {count:,} objects")
        except Exception as e:
            console.print(f"  {name}: [yellow]skipped ({e})[/yellow]")

    console.print("\n[dim]Backup in progress (this may take 30-60 minutes)...[/dim]")

    result = client.backup.create(
        backup_id=backup_id,
        backend="filesystem",
        wait_for_completion=True,
    )

    console.print(f"\n[bold green]Backup completed![/bold green]")
    console.print(f"  Backup ID: {backup_id}")
    console.print(f"  Status: {result.status}")

    # Show size on disk
    backup_dir = BACKUP_PATH / backup_id
    if backup_dir.exists():
        size_bytes = sum(f.stat().st_size for f in backup_dir.rglob("*") if f.is_file())
        size_gb = size_bytes / (1024**3)
        console.print(f"  Size on disk: {size_gb:.2f} GB")
        console.print(f"  Path: {backup_dir}")


def restore_backup(client: weaviate.WeaviateClient, backup_id: str) -> None:
    """Restore from a native filesystem backup."""
    backup_dir = BACKUP_PATH / backup_id
    if not backup_dir.exists():
        console.print(f"[red]Backup not found:[/red] {backup_dir}")
        sys.exit(1)

    console.print(f"\n[bold yellow]Restoring backup:[/bold yellow] {backup_id}")
    console.print(f"[yellow]WARNING: This will overwrite existing collections![/yellow]")
    console.print(f"Path: {backup_dir}")

    confirm = input("\nType 'yes' to confirm restore: ")
    if confirm.strip().lower() != "yes":
        console.print("[dim]Restore cancelled.[/dim]")
        return

    console.print("\n[dim]Restore in progress...[/dim]")

    result = client.backup.restore(
        backup_id=backup_id,
        backend="filesystem",
        wait_for_completion=True,
    )

    console.print(f"\n[bold green]Restore completed![/bold green]")
    console.print(f"  Status: {result.status}")


def list_backups() -> None:
    """List existing backups on the filesystem."""
    if not BACKUP_PATH.exists():
        console.print(f"[yellow]Backup directory does not exist:[/yellow] {BACKUP_PATH}")
        return

    backups = sorted(
        [d for d in BACKUP_PATH.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )

    if not backups:
        console.print("[yellow]No backups found.[/yellow]")
        return

    table = Table(title="Native Weaviate Backups")
    table.add_column("Backup ID", style="cyan")
    table.add_column("Date", style="green")
    table.add_column("Size", justify="right")

    for backup_dir in backups:
        mtime = datetime.fromtimestamp(backup_dir.stat().st_mtime)
        size_bytes = sum(f.stat().st_size for f in backup_dir.rglob("*") if f.is_file())
        size_gb = size_bytes / (1024**3)
        table.add_row(
            backup_dir.name,
            mtime.strftime("%Y-%m-%d %H:%M"),
            f"{size_gb:.2f} GB",
        )

    console.print(table)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Native Weaviate backup/restore via filesystem module",
    )
    parser.add_argument(
        "--backup-id",
        type=str,
        help="Backup identifier (default: auto-generated timestamp)",
    )
    parser.add_argument(
        "--restore",
        type=str,
        metavar="BACKUP_ID",
        help="Restore from a specific backup ID",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List existing backups",
    )

    args = parser.parse_args()

    if args.list:
        list_backups()
        return

    client = None
    try:
        client = get_client()
        console.print("[green]Connected to Weaviate[/green]")

        if args.restore:
            restore_backup(client, args.restore)
        else:
            backup_id = args.backup_id or f"backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            create_backup(client, backup_id)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        logger.exception("Backup operation failed")
        sys.exit(1)
    finally:
        if client:
            client.close()


if __name__ == "__main__":
    main()
