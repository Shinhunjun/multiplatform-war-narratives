from __future__ import annotations

import importlib.util
import sys
import types
import uuid
from pathlib import Path

import pandas as pd
import pytest

import consolidate_yearly


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "data-collection"


def load_data_collection_script(
    monkeypatch: pytest.MonkeyPatch,
    script_name: str,
    argv: list[str],
) -> types.ModuleType:
    fake_newspaper = types.ModuleType("newspaper")

    class FakeConfig:
        def __init__(self) -> None:
            self.browser_user_agent = ""
            self.request_timeout = 0

    class FakeArticle:
        def __init__(self, url: object, config: object | None = None) -> None:
            self.url = url
            self.config = config
            self.title = ""
            self.text = ""

        def download(self) -> None:
            return None

        def parse(self) -> None:
            return None

    fake_newspaper.Config = FakeConfig
    fake_newspaper.Article = FakeArticle
    monkeypatch.setitem(sys.modules, "newspaper", fake_newspaper)

    fake_waybackpy = types.ModuleType("waybackpy")

    class FakeSnapshot:
        def __init__(self, archive_url: str) -> None:
            self.archive_url = archive_url

    class FakeUrl:
        def __init__(self, original_url: object, user_agent: str) -> None:
            self.original_url = original_url
            self.user_agent = user_agent

        def near(self, year: int, month: int, day: int) -> FakeSnapshot:
            return FakeSnapshot(f"https://web.archive.org/{year:04d}{month:02d}{day:02d}/{self.original_url}")

    fake_waybackpy.Url = FakeUrl
    monkeypatch.setitem(sys.modules, "waybackpy", fake_waybackpy)

    monkeypatch.setattr(sys, "argv", argv)
    script_path = SCRIPTS_DIR / script_name
    module_name = f"test_{script_name.replace('.py', '')}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_consolidation_row(
    *,
    date: int,
    year: int,
    source_url: str,
    title: object,
    text: object,
    status: str,
    actor1_country: str = "VEN",
    actor2_country: str = "USA",
    actor1_name: str = "Actor One",
    actor2_name: str = "Actor Two",
    event_code: int = 120,
    quad_class: int = 1,
    goldstein: float = 1.5,
    tone: float = 0.2,
    error_details: object = "",
) -> dict[str, object]:
    return {
        "Date": date,
        "Year": year,
        "Actor1Name": actor1_name,
        "Actor1CountryCode": actor1_country,
        "Actor2Name": actor2_name,
        "Actor2CountryCode": actor2_country,
        "EventCode": event_code,
        "QuadClass": quad_class,
        "GoldsteinScale": goldstein,
        "AvgTone": tone,
        "SourceURL": source_url,
        "Title": title,
        "Text": text,
        "Scrape_Status": status,
        "Error_Details": error_details,
    }


def test_scrape_prepare_dataframe_builds_year_subset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = load_data_collection_script(monkeypatch, "scrape_by_year.py", ["scrape_by_year.py", "2020"])

    input_file = tmp_path / "bq.csv"
    output_file = tmp_path / "ven_usa_2020.csv"
    pd.DataFrame(
        [
            {"SQLDATE": 20200101, "SourceURL": "http://a.com/story"},
            {"SQLDATE": 20200105, "SourceURL": "Unspecified"},
            {"SQLDATE": 20200106, "SourceURL": ""},
            {"SQLDATE": 20190101, "SourceURL": "http://old.com/story"},
        ]
    ).to_csv(input_file, index=False)

    mod.INPUT_FILE = str(input_file)
    mod.OUTPUT_FILE = str(output_file)
    mod.YEAR = 2020

    out = mod.prepare_dataframe()

    assert len(out) == 1
    assert out.iloc[0]["SourceURL"] == "http://a.com/story"
    assert out.iloc[0]["Scrape_Status"] == "Pending"
    assert "domain" in out.columns
    assert output_file.exists()


def test_scrape_prepare_dataframe_resume_adds_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = load_data_collection_script(monkeypatch, "scrape_by_year.py", ["scrape_by_year.py", "2020"])

    resumed = tmp_path / "ven_usa_2020.csv"
    pd.DataFrame([{"SourceURL": "http://a.com"}]).to_csv(resumed, index=False)
    mod.OUTPUT_FILE = str(resumed)

    out = mod.prepare_dataframe()
    assert "Scrape_Status" in out.columns
    assert out.iloc[0]["Scrape_Status"] == "Pending"


def test_scrape_article_handles_success_empty_and_error(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = load_data_collection_script(monkeypatch, "scrape_by_year.py", ["scrape_by_year.py", "2020"])

    class SuccessArticle:
        def __init__(self, url: object, config: object | None = None) -> None:
            self.title = "Good title"
            self.text = "Body text"

        def download(self) -> None:
            return None

        def parse(self) -> None:
            return None

    monkeypatch.setattr(mod, "Article", SuccessArticle)
    ok = mod.scrape_article("http://example.com")
    assert ok["Scrape_Status"] == "Success"
    assert ok["Title"] == "Good title"

    class EmptyArticle(SuccessArticle):
        def __init__(self, url: object, config: object | None = None) -> None:
            super().__init__(url, config)
            self.text = ""

    monkeypatch.setattr(mod, "Article", EmptyArticle)
    empty = mod.scrape_article("http://example.com/empty")
    assert empty["Scrape_Status"] == "Empty_Content"

    class BrokenArticle:
        def __init__(self, url: object, config: object | None = None) -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr(mod, "Article", BrokenArticle)
    err = mod.scrape_article("http://example.com/error")
    assert err["Scrape_Status"] == "Error"
    assert "boom" in err["Error_Details"]


def test_scrape_main_processes_pending_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = load_data_collection_script(monkeypatch, "scrape_by_year.py", ["scrape_by_year.py", "2020"])

    df = pd.DataFrame(
        [
            {
                "SourceURL": "http://done.com",
                "domain": "done.com",
                "Title": "already",
                "Text": "done",
                "Scrape_Status": "Success",
                "Error_Details": "",
            },
            {
                "SourceURL": "http://a.com/1",
                "domain": "a.com",
                "Title": None,
                "Text": None,
                "Scrape_Status": "Pending",
                "Error_Details": None,
            },
            {
                "SourceURL": "http://a.com/2",
                "domain": "a.com",
                "Title": None,
                "Text": None,
                "Scrape_Status": "Pending",
                "Error_Details": None,
            },
        ]
    )

    mod.OUTPUT_FILE = str(tmp_path / "ven_usa_2020.csv")
    mod.BATCH_SIZE = 1
    monkeypatch.setattr(mod, "prepare_dataframe", lambda: df.copy())
    monkeypatch.setattr(mod.time, "sleep", lambda seconds: None)

    class DummyBar:
        def update(self, n: int) -> None:
            return None

        def set_description(self, desc: str) -> None:
            self.desc = desc

        def close(self) -> None:
            return None

    monkeypatch.setattr(mod, "tqdm", lambda *args, **kwargs: DummyBar())

    def fake_scrape(url: object) -> dict[str, object]:
        if str(url).endswith("/1"):
            return {"Title": "T1", "Text": "Body 1", "Scrape_Status": "Success", "Error_Details": ""}
        return {"Title": None, "Text": None, "Scrape_Status": "Error", "Error_Details": "failed"}

    monkeypatch.setattr(mod, "scrape_article", fake_scrape)
    mod.main()

    out = pd.read_csv(mod.OUTPUT_FILE, low_memory=False)
    assert out.loc[1, "Scrape_Status"] == "Success"
    assert out.loc[2, "Scrape_Status"] == "Error"
    assert out.loc[2, "Error_Details"] == "failed"


def test_rescue_get_wayback_url_and_scrape(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = load_data_collection_script(monkeypatch, "rescue_by_year.py", ["rescue_by_year.py", "2021"])

    class GoodUrl:
        def __init__(self, original_url: object, user_agent: str) -> None:
            self.original_url = original_url

        def near(self, year: int, month: int, day: int) -> object:
            return types.SimpleNamespace(archive_url=f"https://archive/{year}/{month}/{day}")

    monkeypatch.setattr(mod.waybackpy, "Url", GoodUrl)
    assert mod.get_wayback_url("http://example.com", 20210115) == "https://archive/2021/1/15"
    assert mod.get_wayback_url("http://example.com", 999) is not None

    class BadUrl:
        def __init__(self, original_url: object, user_agent: str) -> None:
            pass

        def near(self, year: int, month: int, day: int) -> object:
            raise RuntimeError("no snapshot")

    monkeypatch.setattr(mod.waybackpy, "Url", BadUrl)
    assert mod.get_wayback_url("http://example.com", 20210115) is None

    class ArchivedArticle:
        def __init__(self, url: object, config: object | None = None) -> None:
            self.title = "Archived title"
            self.text = "x" * 120

        def download(self) -> None:
            return None

        def parse(self) -> None:
            return None

    monkeypatch.setattr(mod, "Article", ArchivedArticle)
    ok = mod.scrape_archived_article("https://archive/test")
    assert ok["Status"] == "Success (Archived)"
    assert ok["Title"] == "Archived title"


def test_rescue_prepare_dataframe_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = load_data_collection_script(monkeypatch, "rescue_by_year.py", ["rescue_by_year.py", "2020"])

    resumed = tmp_path / "ven_usa_2020_rescued.csv"
    pd.DataFrame([{"SourceURL": "http://a.com", "SQLDATE": 20200101}]).to_csv(resumed, index=False)
    mod.OUTPUT_FILE = str(resumed)
    out = mod.prepare_dataframe()
    assert len(out) == 1

    input_file = tmp_path / "ven_usa_2020.csv"
    pd.DataFrame([{"SourceURL": "http://a.com", "Date": 20200101, "Scrape_Status": "Error"}]).to_csv(input_file, index=False)
    mod.OUTPUT_FILE = str(tmp_path / "missing_rescued.csv")
    mod.INPUT_FILE = str(input_file)
    out2 = mod.prepare_dataframe()
    assert "SQLDATE" in out2.columns
    assert out2.loc[0, "SQLDATE"] == 20200101

    mod.INPUT_FILE = str(tmp_path / "does_not_exist.csv")
    with pytest.raises(SystemExit):
        mod.prepare_dataframe()


def test_rescue_main_marks_rescue_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = load_data_collection_script(monkeypatch, "rescue_by_year.py", ["rescue_by_year.py", "2020"])

    df = pd.DataFrame(
        [
            {"SourceURL": "http://has-archive.com", "SQLDATE": 20200101, "Scrape_Status": "Error", "Title": None, "Text": None, "Error_Details": ""},
            {"SourceURL": "http://already-success.com", "SQLDATE": 20200101, "Scrape_Status": "Success", "Title": "ok", "Text": "ok", "Error_Details": ""},
            {"SourceURL": "http://no-archive.com", "SQLDATE": 20200101, "Scrape_Status": "Failed", "Title": None, "Text": None, "Error_Details": ""},
        ]
    )

    mod.OUTPUT_FILE = str(tmp_path / "ven_usa_2020_rescued.csv")
    mod.LOG_FILE = str(tmp_path / "duration_log_2020.txt")
    mod.BATCH_SIZE = 1
    monkeypatch.setattr(mod, "prepare_dataframe", lambda: df.copy())
    monkeypatch.setattr(mod.time, "sleep", lambda seconds: None)

    class FakeTqdm:
        def __call__(self, iterable: object) -> object:
            return iterable

        @staticmethod
        def write(message: str) -> None:
            return None

    monkeypatch.setattr(mod, "tqdm", FakeTqdm())
    monkeypatch.setattr(
        mod,
        "get_wayback_url",
        lambda original_url, date_int: "https://archive/ok" if "has-archive" in str(original_url) else None,
    )
    monkeypatch.setattr(
        mod,
        "scrape_archived_article",
        lambda archive_url: {"Title": "rescued", "Text": "x" * 150, "Status": "Success (Archived)"},
    )

    mod.main()

    out = pd.read_csv(mod.OUTPUT_FILE, low_memory=False)
    assert out.loc[0, "Scrape_Status"] == "Success (Archived)"
    assert out.loc[2, "Scrape_Status"] == "Rescue_Failed"
    assert "Wayback: No snapshot found" in str(out.loc[2, "Error_Details"])

    log = Path(mod.LOG_FILE)
    assert log.exists()
    lines = log.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "Index,Duration_Seconds"
    assert len(lines) >= 3


def test_consolidate_helpers_cover_key_behaviors(tmp_path: Path) -> None:
    assert consolidate_yearly.parse_years("2013-2011,2015,2015") == [2011, 2012, 2013, 2015]
    assert consolidate_yearly.resolve_path(tmp_path, "x.csv") == tmp_path / "x.csv"

    df = pd.DataFrame([{"A": 1}])
    consolidate_yearly.ensure_columns(df, ["A", "B"])
    assert "B" in df.columns

    ser = pd.Series(["a", "", pd.NA, "  b "], dtype="object")
    present_mask = consolidate_yearly.present(ser)
    assert present_mask.tolist() == [True, False, False, True]
    assert consolidate_yearly.normalize(ser).tolist() == ["a", "", "<NA>", "b"]

    keys = consolidate_yearly.key_with_occurrence(
        pd.DataFrame(
            [
                make_consolidation_row(date=20200101, year=2020, source_url="http://a.com", title="t", text="x", status="Success"),
                make_consolidation_row(date=20200101, year=2020, source_url="http://a.com", title="t", text="x", status="Success"),
            ]
        )
    )
    assert keys.iloc[0].endswith("||0")
    assert keys.iloc[1].endswith("||1")
    assert consolidate_yearly.issue_type(True, False) == "Title_Only"
    assert consolidate_yearly.issue_type(False, True) == "Text_Only"


def test_consolidate_year_replaces_only_both_missing_rows(tmp_path: Path) -> None:
    year = 2020
    orig = pd.DataFrame(
        [
            make_consolidation_row(
                date=20200101,
                year=year,
                source_url="http://a.com/1",
                title="orig good",
                text="orig text",
                status="Success",
            ),
            make_consolidation_row(
                date=20200102,
                year=year,
                source_url="http://a.com/2",
                title=pd.NA,
                text=pd.NA,
                status="Error",
            ),
            make_consolidation_row(
                date=20200103,
                year=year,
                source_url="http://a.com/3",
                title="partial title",
                text="",
                status="Error",
            ),
        ]
    )
    resc = pd.DataFrame(
        [
            make_consolidation_row(
                date=20200101,
                year=year,
                source_url="http://a.com/1",
                title="resc differs",
                text="resc differs",
                status="Success (Archived)",
            ),
            make_consolidation_row(
                date=20200102,
                year=year,
                source_url="http://a.com/2",
                title="rescued title",
                text="rescued text",
                status="Success (Archived)",
            ),
            make_consolidation_row(
                date=20200103,
                year=year,
                source_url="http://a.com/3",
                title="resc partial",
                text="resc text",
                status="Success (Archived)",
            ),
        ]
    )
    orig.to_csv(tmp_path / f"ven_usa_{year}.csv", index=False)
    resc.to_csv(tmp_path / f"ven_usa_{year}_rescued.csv", index=False)

    consolidated, problematic, audit = consolidate_yearly.consolidate_year(tmp_path, year)

    replaced_row = consolidated.loc[consolidated["SourceURL"] == "http://a.com/2"].iloc[0]
    partial_row = consolidated.loc[consolidated["SourceURL"] == "http://a.com/3"].iloc[0]

    assert replaced_row["Title"] == "rescued title"
    assert replaced_row["Text"] == "rescued text"
    assert replaced_row["Scrape_Status"] == "Success (Archived)"
    assert pd.isna(partial_row["Text"]) or str(partial_row["Text"]).strip() == ""
    assert len(problematic) == 1
    assert problematic.iloc[0]["Issue_Type"] == "Title_Only"
    assert audit["RowsReplacedFromRescue"] == 1
    assert audit["PotentialOverwriteRiskRows"] >= 1
    assert audit["OriginalPartialRows"] == 1
    assert audit["RescuedFileExists"] is True


def test_consolidate_main_writes_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    year = 2020
    pd.DataFrame(
        [
            make_consolidation_row(
                date=20200101,
                year=year,
                source_url="http://a.com/1",
                title="title",
                text="text",
                status="Success",
            ),
            make_consolidation_row(
                date=20200102,
                year=year,
                source_url="http://a.com/2",
                title="title only",
                text="",
                status="Error",
            ),
        ]
    ).to_csv(tmp_path / f"ven_usa_{year}.csv", index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "consolidate_yearly.py",
            "--base-dir",
            str(tmp_path),
            "--years",
            "2020",
            "--output",
            "gdelt_scraped.csv",
            "--problematic",
            "problematic_rows.csv",
            "--audit",
            "consolidation_audit.csv",
            "--no-progress",
        ],
    )

    consolidate_yearly.main()

    out = tmp_path / "gdelt_scraped.csv"
    prob = tmp_path / "problematic_rows.csv"
    audit = tmp_path / "consolidation_audit.csv"
    assert out.exists()
    assert prob.exists()
    assert audit.exists()

    out_df = pd.read_csv(out, low_memory=False)
    prob_df = pd.read_csv(prob, low_memory=False)
    audit_df = pd.read_csv(audit, low_memory=False)
    assert len(out_df) == 2
    assert len(prob_df) == 1
    assert int(audit_df.loc[0, "OriginalPartialRows"]) == 1
