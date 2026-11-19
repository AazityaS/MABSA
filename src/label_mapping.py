"""Reproducible label mappings extracted from the supplied taxonomy workbook."""

from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile


def read_taxonomy_xlsx(path: str | Path):
    """Read category and code columns without requiring a spreadsheet runtime."""
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
          "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
    with zipfile.ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rel_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rels = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rel_root}
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            strings = ["".join(t.text or "" for t in si.findall(".//m:t", ns))
                       for si in shared.findall("m:si", ns)]
        sheet_target = None
        for sheet in workbook.findall("m:sheets/m:sheet", ns):
            if sheet.attrib["name"] == "TAXONOMY":
                sheet_target = rels[sheet.attrib[f"{{{ns['r']}}}id"]].lstrip("/")
                if not sheet_target.startswith("xl/"):
                    sheet_target = "xl/" + sheet_target
                break
        if sheet_target is None:
            raise ValueError("TAXONOMY worksheet not found")
        sheet = ET.fromstring(archive.read(sheet_target))
    records = []
    for row in sheet.findall(".//m:sheetData/m:row", ns)[1:]:
        values = {}
        for cell in row.findall("m:c", ns):
            col = "".join(ch for ch in cell.attrib["r"] if ch.isalpha())
            value = cell.find("m:v", ns)
            text = value.text if value is not None else ""
            if cell.attrib.get("t") == "s" and text:
                text = strings[int(text)]
            values[col] = text
        category, code = values.get("A", ""), values.get("B", "")
        if category and code:
            records.append({"aspect_category": category, "taxonomy_code": code,
                            "subcategory": values.get("C", "")})
    return records


def build_label_mappings(taxonomy_path: str | Path):
    taxonomy = read_taxonomy_xlsx(taxonomy_path)
    categories = sorted({row["aspect_category"] for row in taxonomy})
    codes = sorted({row["taxonomy_code"] for row in taxonomy})
    code_to_category = {row["taxonomy_code"]: row["aspect_category"] for row in taxonomy}
    prefix_to_category = {code.split("_", 1)[0] + "_": category
                          for code, category in code_to_category.items()}
    for row in taxonomy:
        prefix = row["taxonomy_code"].split("_", 1)[0] + "_"
        if prefix_to_category[prefix] != row["aspect_category"]:
            raise ValueError(f"Inconsistent taxonomy prefix mapping: {row}")
    return {
        "aspect_categories": {label: i for i, label in enumerate(categories)},
        "taxonomy_codes": {label: i for i, label in enumerate(codes)},
        "taxonomy_code_to_category": code_to_category,
        "sentiments": {label: i for i, label in enumerate(["negative", "neutral", "positive"])},
        "implicitness": {"explicit": 0, "implicit": 1},
        "taxonomy": taxonomy,
    }

