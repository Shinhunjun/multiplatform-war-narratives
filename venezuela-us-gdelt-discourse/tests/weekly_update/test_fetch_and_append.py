from __future__ import annotations

import pandas as pd

import append_master_dataset
import fetch_weekly_events


def test_parse_masterfile_lines_filters_export_urls_by_date_window() -> None:
    text = "\n".join(
        [
            "100 a http://data.gdeltproject.org/gdeltv2/20260322000000.export.CSV.zip",
            "100 b http://data.gdeltproject.org/gdeltv2/20260322000000.mentions.CSV.zip",
            "100 c http://data.gdeltproject.org/gdeltv2/20260323001500.export.CSV.zip",
        ]
    )

    urls = fetch_weekly_events.parse_masterfile_lines(text, start_date="20260323", end_date="20260323")

    assert urls == ["http://data.gdeltproject.org/gdeltv2/20260323001500.export.CSV.zip"]


def test_map_export_row_extracts_repo_schema_for_ven_usa_dyad() -> None:
    row = [""] * 61
    row[0] = "123"
    row[1] = "20260323"
    row[3] = "2026"
    row[6] = "VENGOV"
    row[7] = "VEN"
    row[16] = "USA"
    row[17] = "USA"
    row[26] = "112"
    row[29] = "4"
    row[30] = "-5.0"
    row[34] = "-2.5"
    row[60] = "http://example.com/story"

    mapped = fetch_weekly_events.map_export_row(row, export_timestamp="20260323174500")

    assert mapped == {
        "GLOBALEVENTID": "123",
        "GDELTExportTimestamp": "20260323174500",
        "Date": "20260323",
        "Year": "2026",
        "Actor1Name": "VENGOV",
        "Actor1CountryCode": "VEN",
        "Actor2Name": "USA",
        "Actor2CountryCode": "USA",
        "EventCode": "112",
        "QuadClass": "4",
        "GoldsteinScale": "-5.0",
        "AvgTone": "-2.5",
        "SourceURL": "http://example.com/story",
    }


def test_rows_to_append_drops_same_date_overlap_by_occurrence_count() -> None:
    recent_existing = pd.DataFrame(
        [
            {
                "Date": "20260323",
                "Year": "2026",
                "Actor1Name": "VENGOV",
                "Actor1CountryCode": "VEN",
                "Actor2Name": "USA",
                "Actor2CountryCode": "USA",
                "EventCode": "112",
                "QuadClass": "4",
                "GoldsteinScale": "-5.0",
                "AvgTone": "-2.5",
                "SourceURL": "http://example.com/a",
                "Title": "Old",
                "Text": "Old text",
                "Scrape_Status": "Success",
                "Error_Details": "",
            }
        ]
    )
    incoming = pd.DataFrame(
        [
            recent_existing.iloc[0].to_dict(),
            {
                "Date": "20260323",
                "Year": "2026",
                "Actor1Name": "USA",
                "Actor1CountryCode": "USA",
                "Actor2Name": "VENGOV",
                "Actor2CountryCode": "VEN",
                "EventCode": "113",
                "QuadClass": "4",
                "GoldsteinScale": "-3.0",
                "AvgTone": "-1.5",
                "SourceURL": "http://example.com/b",
                "Title": "New",
                "Text": "New text",
                "Scrape_Status": "Success",
                "Error_Details": "",
            },
        ]
    )

    appended = append_master_dataset.rows_to_append(incoming, recent_existing)

    assert len(appended) == 1
    assert appended.iloc[0]["SourceURL"] == "http://example.com/b"
