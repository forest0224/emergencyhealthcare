

"""Locked, atomic workbook storage for assessment records."""

import csv
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from io import BytesIO, StringIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

from filelock import FileLock
from openpyxl import Workbook, load_workbook
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from backend.security import hash_patient_id, scrub_free_text

INPUTS = {
    "Encounter ID": "encounter_id", "Age": "age", "Gender": "gender",
    "Arrival Hour": "arrival_hour", "Chief Complaint": "chief_complaint",
    "Systolic Blood Pressure": "systolic_bp", "Diastolic Blood Pressure": "diastolic_bp",
    "Heart Rate": "heart_rate", "Respiratory Rate": "respiratory_rate",
    "Temperature": "temperature", "Oxygen Saturation": "spo2", "Pain Score": "pain_score",
}
LABELS = ("Critical", "Emergent", "Urgent", "Less Urgent")
PROBABILITIES = tuple(f"ESI {n} Probability" for n in range(1, 5))
COLUMNS = (
    "Patient ID", "Assessment Date", "Assessment Time", *INPUTS,
    "Predicted Acuity", "Confidence", *PROBABILITIES, "Safety Flags", "Imputed Fields",
    "Reasoning Source", "Model Version", "Assessment Status", "Record Type",
)
SUMMARY = (
    "Patient ID", "Assessment Date", "Assessment Time", "Age", "Gender",
    "Predicted Acuity", "Assessment Status", "Record Type",
)
SHEET = "Patient Records"
INDEX = "_Submissions"
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def safe_cell(value):
    if isinstance(value, str):
        value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", value)
        if value.lstrip().startswith(("=", "+", "-", "@")) or value.startswith(("\t", "\r", "\n")):
            return "'" + value
    return value


def new_workbook():
    book = Workbook()
    sheet = book.active
    sheet.title = SHEET
    sheet.append(COLUMNS)
    sheet.freeze_panes = "A2"
    for cell in sheet[1]:
        cell.fill = PatternFill("solid", fgColor="142C42")
        cell.font = Font(color="FFFFFF", bold=True)
        sheet.column_dimensions[cell.column_letter].width = max(18, len(cell.value) + 3)
    sheet.column_dimensions["H"].width = 52
    for number, color in enumerate(("FFC7CE", "FCE4D6", "FFF2CC", "C6EFCE"), 1):
        sheet.conditional_formatting.add(
            "P2:P1048576", FormulaRule(formula=[f'LEFT($P2,5)="ESI {number}"'],
                                       fill=PatternFill("solid", fgColor=color))
        )
    return book


def style_row(sheet, row):
    for cell in sheet[row]:
        if row % 2 == 0:
            cell.fill = PatternFill("solid", fgColor="EDF4FA")
        if cell.column in range(17, 22):
            cell.number_format = "0.0%"
    sheet.cell(row, 2).number_format = "yyyy-mm-dd"
    sheet.cell(row, 3).number_format = "hh:mm:ss"
    sheet.cell(row, 8).alignment = Alignment(wrap_text=True, vertical="top")
    sheet.auto_filter.ref = f"A1:{get_column_letter(len(COLUMNS))}{row}"


def add_record(sheet, record):
    sheet.append([safe_cell(record.get(column)) for column in COLUMNS])
    style_row(sheet, sheet.max_row)


def _record(values):
    record = dict(zip(COLUMNS, values))
    for key in ("Assessment Date", "Assessment Time"):
        value = record[key]
        if isinstance(value, datetime) and key == "Assessment Date":
            value = value.date()
        if hasattr(value, "isoformat"):
            record[key] = value.isoformat()
    return record


class RecordConflict(Exception):
    """A submission key is already associated with a different request."""


class PatientRecords:
    def __init__(self, path=None):
        self.path = Path(path or os.getenv("PATIENT_RECORDS_PATH") or "data/patient_records.xlsx")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(str(self.path) + ".lock", timeout=8)

    def _open(self):
        if self.path.exists():
            return load_workbook(self.path)
        book = new_workbook()
        sheet = book.create_sheet(INDEX)
        sheet.sheet_state = "hidden"
        sheet.append(("Submission", "Fingerprint", "Workbook Row"))
        return book

    def _save(self, book):
        with NamedTemporaryFile(dir=self.path.parent, suffix=".xlsx", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            book.save(temporary)
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def begin(self, *, actor, key, patient, record_type, mode):
        """Reserve one assessment row and identify replays under the same lock."""
        scoped = hashlib.sha256(f"{actor}\0{key}".encode()).hexdigest()
        fingerprint = hashlib.sha256(json.dumps(
            [mode, record_type, patient], sort_keys=True, allow_nan=False
        ).encode()).hexdigest()
        with self.lock:
            book = self._open()
            try:
                sheet = book[SHEET]
                index = book[INDEX]
                for submission, digest, row_number in index.iter_rows(min_row=2, values_only=True):
                    if submission == scoped:
                        if digest != fingerprint:
                            raise RecordConflict()
                        return _record([cell.value for cell in sheet[row_number]]), False
                now = datetime.now(timezone.utc).replace(microsecond=0)
                record = {name: patient.get(key) for name, key in INPUTS.items()}
                record["Encounter ID"] = (
                    hash_patient_id(record["Encounter ID"]) if record["Encounter ID"] else None
                )
                record["Chief Complaint"] = scrub_free_text(record["Chief Complaint"])
                used = {row[0] for row in sheet.iter_rows(min_row=2, max_col=1, values_only=True)}
                while True:
                    assessment = f"PAT-{now:%Y%m%d}-{uuid4().hex.upper()}"
                    if assessment not in used:
                        break
                record.update({
                    "Patient ID": assessment, "Assessment Date": now.date(),
                    "Assessment Time": now.time(), "Assessment Status": "Pending",
                    "Record Type": record_type,
                })
                add_record(sheet, record)
                index.append((scoped, fingerprint, sheet.max_row))
                self._save(book)
                return record, True
            finally:
                book.close()

    def finish(self, assessment, result=None):
        status = "Failed" if result is None else "Completed"
        fields = {"Assessment Status": status}
        if result is not None:
            acuity = result.get("predicted", result.get("predicted_acuity"))
            fields.update({
                "Predicted Acuity": f"ESI {acuity} - {LABELS[acuity - 1]}",
                "Confidence": result["confidence"],
                "Safety Flags": "; ".join(result["safety_flags"]),
                "Imputed Fields": "; ".join(result["imputed_fields"]),
                "Reasoning Source": result.get("llm_backend") or "none",
                "Model Version": result["model_version"],
                **{column: result["proba"].get(str(i))
                   for i, column in enumerate(PROBABILITIES, 1)},
            })
        with self.lock:
            book = self._open()
            try:
                sheet = book[SHEET]
                for row in sheet.iter_rows(min_row=2):
                    if row[0].value == assessment:
                        if row[-2].value != "Pending":
                            raise RecordConflict()
                        for name, value in fields.items():
                            row[COLUMNS.index(name)].value = safe_cell(value)
                        self._save(book)
                        return
                raise RecordConflict()
            finally:
                book.close()

    def rows(self):
        with self.lock:
            book = self._open()
            try:
                if not self.path.exists():
                    self._save(book)
                rows = [_record(row) for row in book[SHEET].iter_rows(min_row=2, values_only=True)]
                return list(reversed(rows))
            finally:
                book.close()


def export_records(rows, format_name):
    if format_name == "csv":
        out = StringIO()
        writer = csv.DictWriter(out, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows({key: safe_cell(row.get(key)) for key in COLUMNS} for row in rows)
        return out.getvalue().encode("utf-8-sig")
    book = new_workbook()
    try:
        for record in rows:
            typed = dict(record)
            typed["Assessment Date"] = datetime.fromisoformat(record["Assessment Date"]).date()
            typed["Assessment Time"] = datetime.strptime(record["Assessment Time"], "%H:%M:%S").time()
            add_record(book[SHEET], typed)
        out = BytesIO()
        book.save(out)
        return out.getvalue()
    finally:
        book.close()
