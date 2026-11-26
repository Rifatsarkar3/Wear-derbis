"""Generate docs/data_dictionary.md from wearpg.schema, the single source of truth for the dataset tables.

Run from the project root with the project interpreter:
    .venv/Scripts/python.exe scripts/gen_data_dictionary.py           # (re)write docs/data_dictionary.md
    .venv/Scripts/python.exe scripts/gen_data_dictionary.py --check   # exit 1 if the file is stale
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from wearpg import schema

OUT = Path(__file__).resolve().parents[1] / "docs" / "data_dictionary.md"

ENUMS = ("MECHANISMS", "SEVERITY_LEVELS", "REGIMES", "ELEMENTS", "MODALITIES", "RIGS", "WEAR_VOLUME_METHODS",
         "LABEL_EVIDENCE", "ADJUDICATION_METHODS", "VIEW_SOURCES", "DEBRIS_SAMPLING", "REFERENCE_RATERS")


def _cell(text) -> str:
    return str(text).replace("|", "\\|")  # a bare '|' would split the Markdown table cell


def _file(table: str) -> str:
    return "views.parquet" if table == "views" else f"{table}.csv"  # mirrors WearDataset.save()


def _key(table: str) -> str:
    return "(`particle_id`, `rater_id`) unique pair" if table == "labels" else f"`{schema.PRIMARY_KEYS[table]}`"


def render() -> str:
    """Markdown text of the data dictionary; deterministic (same schema -> same text)."""
    out = [
        "# Data dictionary",
        "",
        f"Generated from `wearpg.schema` (schema version {schema.SCHEMA_VERSION}) by "
        "`scripts/gen_data_dictionary.py`. Do not edit by hand: change `src/wearpg/schema.py`, then rerun the "
        "script (`--check` exits 1 when this file is stale).",
        "",
        "## Overview (schema module docstring)",
        "",
        "```text",
        schema.__doc__.strip(),
        "```",
        "",
        "- *required* columns must be present and non-missing; `WearDataset.validate()` reports every violation.",
        "- *optional* columns may be omitted from a file; they are added with the missing value on load.",
        f"- 1 mm^3 = {int(schema.UM3_PER_MM3):,} um^3 (`UM3_PER_MM3`); mass [mg] / density [g/cm^3] = volume [mm^3].",
        "",
        "## Files and primary keys",
        "",
        "| table | file | primary key | columns |",
        "|---|---|---|---|",
        *[f"| `{t}` | `{_file(t)}` | {_key(t)} | {len(spec)} |" for t, spec in schema.TABLES.items()],
        "",
        "## Enumerations",
        "",
        "Order matters where an index is used: class index = position in `MECHANISMS`; severity ordinal = position "
        "in `SEVERITY_LEVELS`.",
        "",
        "| name | values |",
        "|---|---|",
        *[f"| `{name}` | {', '.join(f'`{v}`' for v in getattr(schema, name))} |" for name in ENUMS],
        "",
    ]
    for t, spec in schema.TABLES.items():
        out += [f"## `{t}` (`{_file(t)}`, key {_key(t)})", "",
                "| column | dtype | required | description |", "|---|---|---|---|"]
        out += [f"| `{c}` | {dt} | {'yes' if req else 'no'} | {_cell(desc)} |" for c, (dt, req, desc) in spec.items()]
        out.append("")
    out += [
        "## Prediction tables",
        "",
        "Written by the pipeline to `predictions/parts/*.parquet`; key columns `PRED_KEYS` = "
        + ", ".join(f"`{c}`" for c in schema.PRED_KEYS) + ". Not applicable values are NaN (float) or -1 (int).",
        "",
    ]
    for name in ("PARTICLE_PRED_COLUMNS", "SAMPLE_PRED_COLUMNS", "RUN_PRED_COLUMNS"):
        out += [f"- `{name}`: " + ", ".join(f"`{c}`" for c in getattr(schema, name))]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate docs/data_dictionary.md from wearpg.schema.")
    ap.add_argument("--check", action="store_true", help="exit 1 if docs/data_dictionary.md differs from the schema")
    args = ap.parse_args(argv)
    text = render()
    if args.check:
        ok = OUT.exists() and OUT.read_text(encoding="utf-8") == text
        print(f"{OUT.name}: {'up to date' if ok else 'STALE - rerun scripts/gen_data_dictionary.py'}")
        return 0 if ok else 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
