from __future__ import annotations

import argparse
import csv
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert large SWaT XLSX workbooks to CSV using only the Python standard library."
    )
    parser.add_argument("workbooks", nargs="+", type=Path, help="Input .xlsx file(s).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination directory. Defaults to each workbook's directory.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing CSV files.")
    parser.add_argument(
        "--swat-official",
        action="store_true",
        help=(
            "Apply SWaT Dec 2015 workbook cleanup: skip the normal workbook's "
            "process-stage row, strip headers, add the normal label column, "
            "and normalize the 'A ttack' typo."
        ),
    )
    args = parser.parse_args()

    for workbook in args.workbooks:
        output_dir = args.output_dir or workbook.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{workbook.stem}.csv"
        if output_path.exists() and not args.force:
            print(f"SKIP {output_path} already exists; pass --force to overwrite.")
            continue
        rows = convert_xlsx_to_csv(workbook, output_path, swat_official=args.swat_official)
        print(f"WROTE {output_path} ({rows} data rows)")
    return 0


def convert_xlsx_to_csv(workbook_path: Path, output_path: Path, swat_official: bool = False) -> int:
    with zipfile.ZipFile(workbook_path) as archive:
        sheet_path = _first_sheet_path(archive)
        shared_strings = _load_shared_strings(archive)
        header_row = _header_row(workbook_path) if swat_official else 1
        default_label = "Normal" if swat_official and _is_normal_workbook(workbook_path) else None

        with archive.open(sheet_path) as sheet_xml, output_path.open(
            "w", encoding="utf-8", newline=""
        ) as output_file:
            writer = csv.writer(output_file)
            header_width: int | None = None
            data_rows = 0

            for row_index, values in _iter_rows(sheet_xml, shared_strings):
                if row_index < header_row:
                    continue

                values = [value.strip() for value in values]
                if row_index == header_row:
                    values = _trim_trailing_empty(values)
                    if default_label is not None and not _has_label_column(values):
                        values.append("Normal/Attack")
                    header_width = len(values)
                    writer.writerow(values)
                    continue

                if _is_empty_row(values):
                    continue
                if header_width is not None:
                    values = values[:header_width] + [""] * max(0, header_width - len(values))
                if default_label is not None and len(values) == header_width - 1:
                    values.append(default_label)
                elif default_label is not None and len(values) == header_width:
                    values[-1] = values[-1] or default_label
                elif swat_official and values:
                    values[-1] = _normalise_swat_label(values[-1])
                writer.writerow(values)
                data_rows += 1

        return data_rows


def _first_sheet_path(archive: zipfile.ZipFile) -> str:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    rel_map = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in rels.findall(f"{{{PKG_REL_NS}}}Relationship")
    }
    sheet = workbook.find(f".//{{{MAIN_NS}}}sheet")
    if sheet is None:
        raise ValueError("Workbook does not contain a worksheet.")
    rel_id = sheet.attrib[f"{{{REL_NS}}}id"]
    target = rel_map[rel_id].lstrip("/")
    return target if target.startswith("xl/") else f"xl/{target}"


def _load_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []

    strings: list[str] = []
    with archive.open("xl/sharedStrings.xml") as shared_xml:
        for event, element in ET.iterparse(shared_xml, events=("end",)):
            if element.tag == f"{{{MAIN_NS}}}si":
                strings.append("".join(text.text or "" for text in element.iter(f"{{{MAIN_NS}}}t")))
                element.clear()
    return strings


def _iter_rows(sheet_xml, shared_strings: list[str]):
    for event, row in ET.iterparse(sheet_xml, events=("end",)):
        if row.tag != f"{{{MAIN_NS}}}row":
            continue

        row_index = int(row.attrib.get("r", "0") or 0)
        values: list[str] = []
        for cell in row.findall(f"{{{MAIN_NS}}}c"):
            cell_ref = cell.attrib.get("r", "")
            column_index = _column_index(cell_ref)
            while len(values) < column_index:
                values.append("")
            values[column_index - 1] = _cell_value(cell, shared_strings)

        yield row_index, values
        row.clear()


def _cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        inline = cell.find(f"{{{MAIN_NS}}}is")
        return "" if inline is None else "".join(text.text or "" for text in inline.iter(f"{{{MAIN_NS}}}t"))

    value = cell.find(f"{{{MAIN_NS}}}v")
    if value is None or value.text is None:
        return ""

    if cell_type == "s":
        index = int(value.text)
        return shared_strings[index] if index < len(shared_strings) else ""
    if cell_type == "b":
        return "TRUE" if value.text == "1" else "FALSE"
    return value.text


def _column_index(cell_ref: str) -> int:
    letters = re.sub(r"[^A-Z]", "", cell_ref.upper())
    index = 0
    for char in letters:
        index = index * 26 + ord(char) - ord("A") + 1
    return max(index, 1)


def _trim_trailing_empty(values: list[str]) -> list[str]:
    trimmed = list(values)
    while trimmed and trimmed[-1] == "":
        trimmed.pop()
    return trimmed


def _header_row(workbook_path: Path) -> int:
    return 2 if _is_normal_workbook(workbook_path) else 1


def _is_normal_workbook(workbook_path: Path) -> bool:
    return "normal" in workbook_path.stem.lower()


def _has_label_column(columns: list[str]) -> bool:
    return any(column.lower().replace(" ", "") in {"normal/attack", "normalattack"} for column in columns)


def _is_empty_row(values: list[str]) -> bool:
    return all(value == "" for value in values)


def _normalise_swat_label(value: str) -> str:
    normalized = value.strip().lower().replace(" ", "")
    if normalized == "attack":
        return "Attack"
    if normalized == "normal":
        return "Normal"
    return value.strip()


if __name__ == "__main__":
    raise SystemExit(main())
