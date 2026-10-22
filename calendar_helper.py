"""Business-day calendar per KB-00.3: Monday-Saturday, skipping warehouse holidays.
Builds a [CALENDAR] block so the model reads dates instead of counting them."""
import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

DATA = Path("dataset/Capstone_Project_Dataset_FTTT_2026")
TODAY = date(2026, 10, 20)
HOLIDAYS = {date.fromisoformat(d.strip())
            for d in pd.read_csv(DATA / "data" / "holidays.csv", dtype=str).iloc[:, 0]}

DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8,
          "sep": 9, "oct": 10, "nov": 11, "dec": 12,
          "जनवरी": 1, "फ़रवरी": 2, "फरवरी": 2, "मार्च": 3, "अप्रैल": 4, "मई": 5, "जून": 6,
          "जुलाई": 7, "अगस्त": 8, "सितंबर": 9, "सितम्बर": 9, "अक्टूबर": 10, "अक्तूबर": 10,
          "नवंबर": 11, "नवम्बर": 11, "दिसंबर": 12, "दिसम्बर": 12}


def month_num(word):
    word = word.lower().strip(".")
    return MONTHS.get(word) or MONTHS.get(word[:3])


def find_dates(text):
    text = str(text).translate(DEV_DIGITS)
    found = []
    for y, m, d in re.findall(r"\b(\d{4})-(\d{2})-(\d{2})\b", text):
        found.append((int(y), int(m), int(d)))
    for d, mon, y in re.findall(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+|[\u0900-\u097F]+)\.?,?\s*(\d{4})?", text):
        if month_num(mon):
            found.append((int(y or 2026), month_num(mon), int(d)))
    for mon, d, y in re.findall(r"\b([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s*(\d{4})?\b", text):
        if month_num(mon):
            found.append((int(y or 2026), month_num(mon), int(d)))
    out = set()
    for y, m, d in found:
        try:
            out.add(date(y, m, d))
        except ValueError:
            pass
    return sorted(out)


def is_business_day(d):
    return d.weekday() != 6 and d not in HOLIDAYS


def add_business_days(start, n):
    d, count = start, 0
    while count < n:
        d += timedelta(days=1)
        if is_business_day(d):
            count += 1
    return d


def fmt(d):
    return f"{d.day} {d.strftime('%B %Y')} ({d.strftime('%a')})"


def calendar_block(question):
    """Uses dates the customer mentions; falls back to today. One result per line."""
    starts = find_dates(question) or [TODAY]
    lines = []
    for start in starts:
        label = "today" if start == TODAY else "date mentioned by the customer"
        lines.append(f"Counting from {fmt(start)} [{label}]:")
        for n in range(1, 11):
            unit = "business day" if n == 1 else "business days"
            lines.append(f"  {n} {unit} after {fmt(start)} = {fmt(add_business_days(start, n))}")
        for n in (7, 10, 15, 30):
            lines.append(f"  {n} calendar days after {fmt(start)} = {fmt(start + timedelta(days=n))}")
    return ("[CALENDAR] Computed with KB-00.3 rules (Sundays and holidays "
            + ", ".join(str(h) for h in sorted(HOLIDAYS)) + " skipped). "
            + "Today is 20 October 2026.\n" + "\n".join(lines))