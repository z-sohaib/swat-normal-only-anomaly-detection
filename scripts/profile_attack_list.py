from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile the official SWaT attack-list XLSX.")
    parser.add_argument("--path", default="data/raw/List_of_attacks_Final.xlsx")
    parser.add_argument("--output-dir", default="runs/attack_list_profile")
    args = parser.parse_args()

    rows = read_xlsx_first_sheet(Path(args.path))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    headers = rows[0] if rows else []
    data_rows = [row for row in rows[1:] if any(cell.strip() for cell in row)]
    profile = {
        "path": args.path,
        "header": headers,
        "row_count": len(data_rows),
        "first_rows": data_rows[:8],
        "last_rows": data_rows[-5:],
    }
    (output_dir / "attack_list_profile.json").write_text(
        json.dumps(profile, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "attack_list_raw.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)

    print(json.dumps(profile, indent=2))
    return 0


def read_xlsx_first_sheet(path: Path) -> list[list[str]]:
    with zipfile.ZipFile(path) as archive:
        sheet_path = _first_sheet_path(archive)
        shared_strings = _load_shared_strings(archive)
        with archive.open(sheet_path) as sheet_xml:
            return [values for _, values in _iter_rows(sheet_xml, shared_strings)]


def _first_sheet_path(archive: zipfile.ZipFile) -> str:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    rel_map = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in rels.findall(f"{{{PKG_REL_NS}}}Relationship")
    }
    sheet = workbook.find(f".//{{{MAIN_NS}}}sheet")
    if sheet is None:
        raise ValueError("Workbook does not contain worksheets.")
    rel_id = sheet.attrib[f"{{{REL_NS}}}id"]
    target = rel_map[rel_id].lstrip("/")
    return target if target.startswith("xl/") else f"xl/{target}"


def _load_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    strings: list[str] = []
    with archive.open("xl/sharedStrings.xml") as shared_xml:
        for _, element in ET.iterparse(shared_xml, events=("end",)):
            if element.tag == f"{{{MAIN_NS}}}si":
                strings.append("".join(text.text or "" for text in element.iter(f"{{{MAIN_NS}}}t")))
                element.clear()
    return strings


def _iter_rows(sheet_xml, shared_strings: list[str]):
    for _, row in ET.iterparse(sheet_xml, events=("end",)):
        if row.tag != f"{{{MAIN_NS}}}row":
            continue
        row_index = int(row.attrib.get("r", "0") or 0)
        values: list[str] = []
        for cell in row.findall(f"{{{MAIN_NS}}}c"):
            column_index = _column_index(cell.attrib.get("r", ""))
            while len(values) < column_index:
                values.append("")
            values[column_index - 1] = _cell_value(cell, shared_strings)
        while values and values[-1] == "":
            values.pop()
        yield row_index, values
        row.clear()


def _cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        inline = cell.find(f"{{{MAIN_NS}}}is")
        return "" if inline is None else "".join(text.text or "" for text in inline.iter(f"{{{MAIN_NS}}}t")).strip()
    value = cell.find(f"{{{MAIN_NS}}}v")
    if value is None or value.text is None:
        return ""
    if cell_type == "s":
        index = int(value.text)
        return shared_strings[index].strip() if index < len(shared_strings) else ""
    if cell_type == "b":
        return "TRUE" if value.text == "1" else "FALSE"
    return value.text.strip()


def _column_index(cell_ref: str) -> int:
    letters = re.sub(r"[^A-Z]", "", cell_ref.upper())
    index = 0
    for char in letters:
        index = index * 26 + ord(char) - ord("A") + 1
    return max(index, 1)


if __name__ == "__main__":
    raise SystemExit(main())
