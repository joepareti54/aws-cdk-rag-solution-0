#!/usr/bin/env python3
"""
Script to bulk upload PDF documents to the RAG system.
Uploads to a FLAT S3 structure (all files in uploads/ folder).

Usage:
    python upload_documents.py /path/to/datasets
    python upload_documents.py /path/to/datasets --dry-run
    python upload_documents.py /path/to/datasets --year 2024
    python upload_documents.py /path/to/datasets --from-year 2023 --to-year 2025
"""

import os
import sys
import boto3
import argparse
from pathlib import Path
from collections import defaultdict
import time
import uuid


VALID_YEARS = [str(year) for year in range(2020, 2027)]

MONTH_NAMES = {
    'january': '01', 'jan': '01', '01': '01',
    'february': '02', 'feb': '02', '02': '02',
    'march': '03', 'mar': '03', '03': '03',
    'april': '04', 'apr': '04', '04': '04',
    'may': '05', '05': '05',
    'june': '06', 'jun': '06', '06': '06',
    'july': '07', 'jul': '07', '07': '07',
    'august': '08', 'aug': '08', '08': '08',
    'september': '09', 'sep': '09', 'sept': '09', '09': '09',
    'october': '10', 'oct': '10', '10': '10',
    'november': '11', 'nov': '11', '11': '11',
    'december': '12', 'dec': '12', '12': '12',
}

MONTH_DISPLAY = {
    '01': 'january', '02': 'february', '03': 'march', '04': 'april',
    '05': 'may', '06': 'june', '07': 'july', '08': 'august',
    '09': 'september', '10': 'october', '11': 'november', '12': 'december',
}


def sanitize_metadata(text):
    """Convert text to ASCII-safe string for S3 metadata."""
    return text.encode('ascii', 'replace').decode('ascii')


def get_bucket_name():
    """Get the documents bucket name from CloudFormation outputs or environment."""
    # Check environment variable first
    bucket = os.environ.get('DOCUMENTS_BUCKET')
    if bucket:
        return bucket

    cf_client = boto3.client("cloudformation")
    try:
        response = cf_client.describe_stacks(StackName="RAGStack")
        outputs = response["Stacks"][0]["Outputs"]
        for output in outputs:
            if output["OutputKey"] == "BucketName":
                return output["OutputValue"]
        raise ValueError("BucketName not found in stack outputs")
    except Exception as e:
        print(f"Error getting bucket name: {e}")
        print("Make sure the RAGStack is deployed or set DOCUMENTS_BUCKET env var.")
        sys.exit(1)


def parse_year_month(filepath):
    """Extract year and month from file path."""
    parts = filepath.parts
    year = None
    month_num = None
    month_name = None

    for part in parts:
        part_clean = part.strip()
        if part_clean in VALID_YEARS:
            year = part_clean
        part_lower = part_clean.lower()
        if part_lower in MONTH_NAMES:
            month_num = MONTH_NAMES[part_lower]
            month_name = MONTH_DISPLAY[month_num]

    return year, month_num, month_name


def discover_files(base_path, from_year=2020, to_year=2026):
    """Discover all PDF files organized by year/month."""
    files_by_period = defaultdict(list)
    unorganized_files = []
    skipped_files = []

    all_pdfs = list(base_path.glob("**/*.pdf"))

    for pdf_path in all_pdfs:
        year, month_num, month_name = parse_year_month(pdf_path)

        if year and month_num:
            year_int = int(year)
            if year_int < from_year or year_int > to_year:
                skipped_files.append((pdf_path, year, month_name, "outside year range"))
                continue
            files_by_period[(year, month_num, month_name)].append(pdf_path)
        else:
            unorganized_files.append(pdf_path)

    return dict(files_by_period), unorganized_files, skipped_files


def upload_file(s3_client, bucket, filepath, year=None, month=None, month_name=None):
    """Upload a single PDF file to S3 with FLAT structure."""
    file_uuid = str(uuid.uuid4())[:8]
    key = f"uploads/{file_uuid}_{filepath.name}"

    metadata = {
        "original-filename": sanitize_metadata(filepath.name),
        "original-path": sanitize_metadata(str(filepath)),
        "upload-id": file_uuid,
    }

    if year:
        metadata["source-year"] = year
    if month:
        metadata["source-month"] = month
    if month_name:
        metadata["source-month-name"] = month_name

    try:
        file_size = filepath.stat().st_size
        s3_client.upload_file(
            str(filepath),
            bucket,
            key,
            ExtraArgs={"ContentType": "application/pdf", "Metadata": metadata}
        )
        return {"file": filepath.name, "status": "success", "size": file_size, "year": year or "N/A"}
    except Exception as e:
        return {"file": filepath.name, "status": "error", "error": str(e), "year": year or "N/A"}


def format_size(size_bytes):
    """Format file size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def print_discovery_summary(files_by_period, unorganized, skipped, from_year, to_year):
    """Print summary of discovered files."""
    print("\n" + "=" * 70)
    print(f"DISCOVERY SUMMARY (Years {from_year}-{to_year})")
    print("=" * 70)

    total_size = 0
    total_files = 0

    if files_by_period:
        sorted_periods = sorted(files_by_period.keys(), key=lambda x: (x[0], x[1]))
        current_year = None
        year_file_count = 0
        year_size = 0

        for year, month_num, month_name in sorted_periods:
            if year != current_year:
                if current_year is not None:
                    print(f"   └── Year total: {year_file_count} file(s) ({format_size(year_size)})\n")
                print(f"📁 {year}/")
                current_year = year
                year_file_count = 0
                year_size = 0

            files = files_by_period[(year, month_num, month_name)]
            count = len(files)
            size = sum(f.stat().st_size for f in files)
            total_size += size
            total_files += count
            year_file_count += count
            year_size += size
            print(f"   ├── {month_name:<12} {count:>4} file(s) ({format_size(size):>10})")

        if current_year is not None:
            print(f"   └── Year total: {year_file_count} file(s) ({format_size(year_size)})")

        print(f"\n📊 Total organized files: {total_files} ({format_size(total_size)})")

    if unorganized:
        unorg_size = sum(f.stat().st_size for f in unorganized)
        print(f"\n⚠️  Unorganized files: {len(unorganized)} ({format_size(unorg_size)})")
        for f in unorganized[:5]:
            print(f"   - {f.name}")
        if len(unorganized) > 5:
            print(f"   ... and {len(unorganized) - 5} more")

    print(f"\n💾 Total size to upload: {format_size(total_size)}")
    print("=" * 70)

    return total_files, total_size


def upload_documents(path, dry_run=False, from_year=2020, to_year=2026, include_unorganized=False):
    """Upload PDF documents from year/month directory structure to FLAT S3 structure."""
    base_path = Path(path)

    if not base_path.exists():
        print(f"Error: Path does not exist: {base_path}")
        sys.exit(1)

    if not base_path.is_dir():
        print(f"Error: Path is not a directory: {base_path}")
        sys.exit(1)

    print(f"🔍 Scanning directory: {base_path}")
    print(f"   Looking for: years {from_year}-{to_year}, all 12 months")

    files_by_period, unorganized_files, skipped_files = discover_files(base_path, from_year, to_year)

    if not files_by_period and not unorganized_files:
        print("\nNo PDF files found matching criteria.")
        return

    total_files, total_size = print_discovery_summary(
        files_by_period, unorganized_files, skipped_files, from_year, to_year
    )

    if include_unorganized:
        total_files += len(unorganized_files)

    if total_files == 0:
        print("\nNo files to upload with current filters.")
        return

    if dry_run:
        print(f"\n🔸 DRY RUN - Would upload {total_files} file(s)")
        return

    print(f"\n📤 Ready to upload {total_files} file(s)")
    response = input("Proceed? [y/N]: ").strip().lower()
    if response != 'y':
        print("Upload cancelled.")
        return

    bucket = get_bucket_name()
    print(f"\n☁️  Uploading to: s3://{bucket}/uploads/")

    s3_client = boto3.client("s3")

    # Build upload list
    upload_tasks = []
    for (year, month_num, month_name), files in files_by_period.items():
        for filepath in files:
            upload_tasks.append({"filepath": filepath, "year": year, "month": month_num, "month_name": month_name})

    if include_unorganized:
        for filepath in unorganized_files:
            upload_tasks.append({"filepath": filepath, "year": None, "month": None, "month_name": None})

    # Upload sequentially
    results = []
    start_time = time.time()
    successful = 0
    failed = 0
    total_uploaded_size = 0

    print(f"\nUploading {len(upload_tasks)} files...\n")

    for i, task in enumerate(upload_tasks, 1):
        result = upload_file(s3_client, bucket, task["filepath"], task["year"], task["month"], task["month_name"])
        results.append(result)

        if result["status"] == "success":
            successful += 1
            total_uploaded_size += result.get("size", 0)
            icon = "✓"
        else:
            failed += 1
            icon = "✗"
            print(f"   {icon} {result['file']}: {result.get('error', 'Unknown error')}")

        progress = (i / len(upload_tasks)) * 100
        print(f"\r[{progress:5.1f}%] {successful} uploaded, {failed} failed", end="", flush=True)

    elapsed = time.time() - start_time

    # Summary
    print("\n\n" + "=" * 70)
    print("UPLOAD COMPLETE")
    print("=" * 70)
    print(f"⏱️  Duration: {elapsed:.1f} seconds")
    print(f"✅ Successful: {successful} ({format_size(total_uploaded_size)})")
    print(f"❌ Failed: {failed}")

    if failed > 0:
        print("\nFailed files:")
        for result in results:
            if result["status"] == "error":
                print(f"   - {result['file']}: {result['error']}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Upload PDF documents to RAG system")
    parser.add_argument("path", help="Path to directory containing year/month structured PDFs")
    parser.add_argument("--dry-run", action="store_true", help="Preview without uploading")
    parser.add_argument("--year", type=int, choices=range(2020, 2027), metavar="YEAR", help="Upload only specific year")
    parser.add_argument("--from-year", type=int, default=2020, metavar="YEAR", help="Start year (default: 2020)")
    parser.add_argument("--to-year", type=int, default=2026, metavar="YEAR", help="End year (default: 2026)")
    parser.add_argument("--include-unorganized", action="store_true", help="Include files not in year/month structure")

    args = parser.parse_args()

    from_year = args.from_year
    to_year = args.to_year

    if args.year:
        from_year = args.year
        to_year = args.year

    if from_year > to_year:
        print("Error: --from-year cannot be greater than --to-year")
        sys.exit(1)

    upload_documents(
        path=args.path,
        dry_run=args.dry_run,
        from_year=from_year,
        to_year=to_year,
        include_unorganized=args.include_unorganized
    )


if __name__ == "__main__":
    main()
