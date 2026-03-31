"""Parse Google Calendar .ics exports."""
from pathlib import Path
from typing import Generator

from config import GCAL_DIR


def iter_events(cal_dir: Path = GCAL_DIR) -> Generator[dict, None, None]:
    from icalendar import Calendar

    for path in sorted(cal_dir.rglob("*.ics")):
        try:
            cal = Calendar.from_ical(path.read_bytes())
        except Exception as e:
            print(f"Warning: could not parse {path.name}: {e}")
            continue

        for component in cal.walk():
            if component.name != "VEVENT":
                continue

            summary = str(component.get("SUMMARY", "")).strip()
            description = str(component.get("DESCRIPTION", "")).strip()
            location = str(component.get("LOCATION", "")).strip()
            dtstart = component.get("DTSTART")
            dtend = component.get("DTEND")

            parts = [f"Event: {summary}"]
            if description:
                parts.append(f"Description: {description}")
            if location:
                parts.append(f"Location: {location}")
            if dtstart:
                parts.append(f"Start: {dtstart.dt}")
            if dtend:
                parts.append(f"End: {dtend.dt}")

            yield {
                "text": "\n".join(parts),
                "metadata": {
                    "source": "gcal",
                    "summary": summary,
                    "start": str(dtstart.dt) if dtstart else "",
                    "end": str(dtend.dt) if dtend else "",
                    "location": location,
                },
            }


def load_all(cal_dir: Path = GCAL_DIR) -> list[dict]:
    return list(iter_events(cal_dir))
