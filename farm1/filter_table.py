#!/usr/bin/env python3
"""
Remove rows from table.dat that appear in status.txt, then renumber.

status.txt format:  <row_number> <status>
table.dat format:   <row_number> <command>

Usage:
    python farm1/filter_table.py
    python farm1/filter_table.py --status farm1/status.txt --table farm1/table.dat
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Filter completed rows from table.dat")
    parser.add_argument("--status", default="farm1/status.txt",
                        help="Path to status.txt (default: farm1/status.txt)")
    parser.add_argument("--table", default="farm1/table.dat",
                        help="Path to table.dat (default: farm1/table.dat)")
    args = parser.parse_args()

    status_path = Path(args.status)
    table_path = Path(args.table)

    # Read row numbers to remove from status.txt
    rows_to_remove = set()
    with open(status_path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if not parts:
                continue
            try:
                rows_to_remove.add(int(parts[0]))
            except ValueError:
                print(f"Warning: could not parse row number on line {lineno} of status.txt: {line!r}")

    print(f"Rows to remove: {len(rows_to_remove)}")

    # Read table.dat and filter out rows in rows_to_remove
    kept_commands = []
    removed_count = 0
    with open(table_path) as f:
        for line in f:
            line = line.rstrip('\n')
            if not line.strip():
                continue
            parts = line.split(' ', 1)
            try:
                row_num = int(parts[0])
            except ValueError:
                print(f"Warning: could not parse row number: {line!r}")
                kept_commands.append(parts[1] if len(parts) > 1 else line)
                continue

            if row_num in rows_to_remove:
                removed_count += 1
            else:
                kept_commands.append(parts[1] if len(parts) > 1 else '')

    print(f"Rows removed:   {removed_count}")
    print(f"Rows remaining: {len(kept_commands)}")

    # Write back with renumbered rows
    with open(table_path, 'w') as f:
        for new_num, cmd in enumerate(kept_commands, 1):
            f.write(f"{new_num} {cmd}\n")

    print(f"Done. {table_path} updated with {len(kept_commands)} rows.")


if __name__ == "__main__":
    main()
