#!/usr/bin/env python3
"""Convert benchmark CSV output to a formatted markdown table."""

import sys
import csv
import io

TIME_COLS = ["orig_btree_us", "csr_us", "csr_par_us", "sprs_us", "magnus_seq_us", "magnus_par_us"]
SPEED_COLS = ["x_csr", "x_csr_par", "x_sprs", "x_magnus_seq", "x_magnus_par"]

HEADERS = [
    "Side", "Nodes", "e/n", "nnz", "Comp",
    "BTree µs", "CSR µs", "CSR‖ µs", "sprs µs", "MAG seq µs", "MAG‖ µs",
    "×CSR", "×CSR‖", "×sprs", "×MAG seq", "×MAG‖",
]

def fmt_int(s):
    return f"{int(s):,}"

def fmt_time(s):
    return f"{int(s):,}"

def fmt_speed(s):
    f = float(s)
    if f >= 100:
        return f"{f:,.1f}"
    return f"{f:.4f}"

def bold_best_speed(speeds):
    """Return list with the best (highest) speedup bolded."""
    vals = []
    for s in speeds:
        try:
            vals.append(float(s))
        except (ValueError, TypeError):
            vals.append(0.0)
    best = max(vals)
    result = []
    for s, v in zip(speeds, vals):
        formatted = fmt_speed(s)
        if v == best and best > 0:
            result.append(f"**{formatted}**")
        else:
            result.append(formatted)
    return result

def main():
    # Read from stdin or file argument
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    # Extract only CSV lines (skip non-CSV output)
    lines = []
    header_line = None
    for line in text.strip().splitlines():
        if line.startswith("side,nodes,"):
            header_line = line
            lines = []  # reset in case of multiple runs
        elif header_line and "," in line and not line.startswith("test "):
            # Heuristic: CSV data lines start with a number
            parts = line.split(",")
            try:
                int(parts[0])
                lines.append(line)
            except ValueError:
                continue

    if not header_line or not lines:
        print("No benchmark CSV data found in input.", file=sys.stderr)
        sys.exit(1)

    reader = csv.DictReader(io.StringIO(header_line + "\n" + "\n".join(lines)))
    rows = list(reader)

    # Build table rows
    table_rows = []
    for r in rows:
        speeds_raw = [r[c] for c in SPEED_COLS]
        speeds_fmt = bold_best_speed(speeds_raw)
        table_rows.append([
            r["side"],
            fmt_int(r["nodes"]),
            r["e_per_n"],
            fmt_int(r["nnz"]),
            r["components"],
            fmt_time(r["orig_btree_us"]),
            fmt_time(r["csr_us"]),
            fmt_time(r["csr_par_us"]),
            fmt_time(r["sprs_us"]),
            fmt_time(r["magnus_seq_us"]),
            fmt_time(r["magnus_par_us"]),
            *speeds_fmt,
        ])

    # Compute column widths
    all_rows = [HEADERS] + table_rows
    widths = [max(len(row[i]) for row in all_rows) for i in range(len(HEADERS))]

    def fmt_row(row):
        cells = [cell.rjust(w) for cell, w in zip(row, widths)]
        return "| " + " | ".join(cells) + " |"

    # Print table
    print(fmt_row(HEADERS))
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for row in table_rows:
        print(fmt_row(row))

if __name__ == "__main__":
    main()
