"""
Prints the published figures the end-to-end regression cases are checked
against, read straight from the CMS export (no database or model needed).

    python scripts/regression_facts.py
    python scripts/regression_facts.py "Real GDP" "Inflation"
    python scripts/regression_facts.py --at Inflation 2024-05 2026-04

tests/e2e/cases.json pins numbers like "47.8" (Total Expenditure, 2024-Q4).
When the data is re-exported those can change; re-run this and update the
cases rather than guessing.
"""
import sys
from pathlib import Path

import pandas as pd

CMS = Path(__file__).parent.parent / "data" / "cms"

DEFAULT_SERIES = [
    "Real GDP",
    "Inflation",
    "Number of International Visitors",
    "Total Expenditure",
    "Non-Hydrocarbon Revenue",
    "Government Revenues - Non-Hydrocarbon",
    "Trade Balance (Goods & Services)",
]


def _read(prefix: str) -> pd.DataFrame:
    path = next(CMS.glob(f"{prefix}*.csv"))
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _load():
    details = _read("P02_Published_IndicatorDetails")
    indicators = _read("P01_Published_Indicators")[["PublishedIndicatorId", "NameEN"]]
    points = _read("P03_Published_DataPoints")
    details = details.merge(indicators.rename(columns={"NameEN": "ParentEN"}),
                            on="PublishedIndicatorId", how="left")
    return details, points


def at(name: str, periods: list[str]) -> int:
    """Qatar's published reading of one series at the given periods."""
    details, points = _load()
    ids = details[details.NameEN.str.strip().str.lower() == name.lower()].PublishedIndicatorDetailId
    rows = points[points.PublishedIndicatorDetailId.isin(ids)
                  & points.CountryEN.isin(["Qatar", ""]) & points.Period.isin(periods)]
    for _, r in rows.sort_values("Period").iterrows():
        print(f"{name} {r.Period} ({r.IntervalEN}): {r.Actual}")
    return 0


def main(names: list[str]) -> int:
    details, points = _load()
    for name in names:
        rows = details[details.NameEN.str.strip().str.lower() == name.lower()]
        if rows.empty:
            print(f"\n## {name}: no published series with this name")
            continue
        for _, d in rows.iterrows():
            series = points[(points.PublishedIndicatorDetailId == d.PublishedIndicatorDetailId)
                            & (points.CountryEN.isin(["Qatar", ""]))]
            print(f"\n## {name}  (under: {d.ParentEN}; unit: {d.UnitEN}; format: {d.Format})")
            for interval, group in series.groupby("IntervalEN"):
                group = group.assign(value=pd.to_numeric(group.Actual, errors="coerce"))
                group = group.dropna(subset=["value"]).sort_values("Period")
                if group.empty:
                    continue
                top = group.loc[group.value.idxmax()]
                low = group.loc[group.value.idxmin()]
                print(f"  {interval}: {len(group)} readings {group.Period.iloc[0]}..{group.Period.iloc[-1]}"
                      f" | max {top.value} ({top.Period}) | min {low.value} ({low.Period})")
                tail = ", ".join(f"{p}={v}" for p, v in zip(group.Period.tail(8), group.value.tail(8)))
                print(f"    latest: {tail}")
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["--at"]:
        sys.exit(at(sys.argv[2], sys.argv[3:]))
    sys.exit(main(sys.argv[1:] or DEFAULT_SERIES))
