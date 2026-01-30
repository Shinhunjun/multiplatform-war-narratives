"""
Main preprocessing class for Reddit data.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from .config import PreprocessConfig
from .filters import is_valid_comment, is_valid_submission
from .text_cleaner import clean_text


class RedditPreprocessor:
    """Main class for preprocessing Reddit submissions and comments."""

    def __init__(self, config: Optional[PreprocessConfig] = None):
        """Initialize preprocessor with config."""
        self.config = config or PreprocessConfig()
        self.config.ensure_output_dir()

        self.stats = {
            "submissions": {
                "total": 0,
                "kept": 0,
                "deleted_content": 0,
                "deleted_author": 0,
                "bot_author": 0,
                "too_short": 0,
            },
            "comments": {
                "total": 0,
                "kept": 0,
                "deleted_content": 0,
                "deleted_author": 0,
                "bot_author": 0,
                "too_short": 0,
            },
        }

    def load_json_files(self, directory: Path, pattern: str = "*.json") -> Dict:
        """Load all JSON files from directory into a single dict."""
        all_data = {}
        files = list(directory.glob(pattern))

        print(f"Loading {len(files)} files from {directory}")

        for file_path in tqdm(files, desc="Loading files"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Filter out metadata keys (starting with _)
                    items = {k: v for k, v in data.items() if not k.startswith("_")}
                    all_data.update(items)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                print(f"Error loading {file_path}: {e}")

        return all_data

    def load_submissions(self) -> pd.DataFrame:
        """Load all submissions into a DataFrame."""
        data = self.load_json_files(
            self.config.submissions_dir, pattern="*_monthly_filtered.json"
        )
        print(f"Loaded {len(data)} submissions")

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(data, orient="index")
        # Reset index without adding it as a column (id already exists in data)
        df = df.reset_index(drop=True)

        self.stats["submissions"]["total"] = len(df)
        return df

    def load_comments(self) -> pd.DataFrame:
        """Load all comments into a DataFrame."""
        data = self.load_json_files(
            self.config.comments_dir, pattern="*_monthly_filtered.json"
        )
        print(f"Loaded {len(data)} comments")

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(data, orient="index")
        # Reset index without adding it as a column (id already exists in data)
        df = df.reset_index(drop=True)

        self.stats["comments"]["total"] = len(df)
        return df

    def preprocess_submissions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess submissions DataFrame."""
        if df.empty:
            return df

        print("Preprocessing submissions...")
        original_count = len(df)

        # Track filtering reasons
        deleted_content_mask = df["title"].apply(
            lambda x: x in ["[deleted]", "[removed]"] if isinstance(x, str) else False
        )
        deleted_author_mask = df["author"].apply(
            lambda x: x in ["[deleted]", None] if isinstance(x, str) or x is None else False
        )
        bot_mask = df["author"].apply(
            lambda x: x in [
                "AutoModerator", "autotldr", "empleadoEstatalBot", "RemindMeBot"
            ] if isinstance(x, str) else False
        )

        self.stats["submissions"]["deleted_content"] = deleted_content_mask.sum()
        self.stats["submissions"]["deleted_author"] = deleted_author_mask.sum()
        self.stats["submissions"]["bot_author"] = bot_mask.sum()

        # Apply filtering
        valid_mask = df.apply(
            lambda row: is_valid_submission(
                title=row.get("title"),
                selftext=row.get("selftext"),
                author=row.get("author"),
                remove_deleted=self.config.remove_deleted,
                remove_bots=self.config.remove_bots,
                min_words=2,  # Titles can be shorter
            ),
            axis=1,
        )

        df = df[valid_mask].copy()

        # Clean text fields
        print("Cleaning submission text...")
        df["title_clean"] = df["title"].apply(
            lambda x: clean_text(
                x,
                remove_urls_flag=self.config.remove_urls,
                remove_markdown_flag=self.config.remove_markdown,
                remove_edit_markers_flag=False,  # Titles don't have edit markers
                remove_emojis_flag=self.config.remove_emojis,
                normalize_accents_flag=self.config.normalize_accents,
                lowercase=self.config.lowercase,
            )
        )

        df["selftext_clean"] = df["selftext"].apply(
            lambda x: clean_text(
                x,
                remove_urls_flag=self.config.remove_urls,
                remove_markdown_flag=self.config.remove_markdown,
                remove_edit_markers_flag=self.config.remove_edit_markers,
                remove_emojis_flag=self.config.remove_emojis,
                normalize_accents_flag=self.config.normalize_accents,
                lowercase=self.config.lowercase,
            )
            if x and isinstance(x, str)
            else ""
        )

        # Combine title and selftext for full text analysis
        df["full_text"] = df["title_clean"] + " " + df["selftext_clean"]
        df["full_text"] = df["full_text"].str.strip()

        self.stats["submissions"]["kept"] = len(df)
        print(
            f"Submissions: {original_count} -> {len(df)} "
            f"({len(df)/original_count*100:.1f}% kept)"
        )

        return df

    def preprocess_comments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess comments DataFrame."""
        if df.empty:
            return df

        print("Preprocessing comments...")
        original_count = len(df)

        # Track filtering reasons
        deleted_content_mask = df["body"].apply(
            lambda x: x in ["[deleted]", "[removed]"] if isinstance(x, str) else False
        )
        deleted_author_mask = df["author"].apply(
            lambda x: x in ["[deleted]", None] if isinstance(x, str) or x is None else False
        )
        bot_mask = df["author"].apply(
            lambda x: x in [
                "AutoModerator", "autotldr", "empleadoEstatalBot", "RemindMeBot"
            ] if isinstance(x, str) else False
        )

        self.stats["comments"]["deleted_content"] = deleted_content_mask.sum()
        self.stats["comments"]["deleted_author"] = deleted_author_mask.sum()
        self.stats["comments"]["bot_author"] = bot_mask.sum()

        # Apply filtering
        valid_mask = df.apply(
            lambda row: is_valid_comment(
                body=row.get("body"),
                author=row.get("author"),
                remove_deleted=self.config.remove_deleted,
                remove_bots=self.config.remove_bots,
                min_words=self.config.min_word_count,
            ),
            axis=1,
        )

        df = df[valid_mask].copy()

        # Clean text
        print("Cleaning comment text...")
        tqdm.pandas(desc="Cleaning comments")
        df["body_clean"] = df["body"].progress_apply(
            lambda x: clean_text(
                x,
                remove_urls_flag=self.config.remove_urls,
                remove_markdown_flag=self.config.remove_markdown,
                remove_edit_markers_flag=self.config.remove_edit_markers,
                remove_emojis_flag=self.config.remove_emojis,
                normalize_accents_flag=self.config.normalize_accents,
                lowercase=self.config.lowercase,
            )
        )

        # Filter again after cleaning (some might become too short)
        df = df[df["body_clean"].str.split().str.len() >= self.config.min_word_count]

        self.stats["comments"]["kept"] = len(df)
        self.stats["comments"]["too_short"] = original_count - len(df) - (
            self.stats["comments"]["deleted_content"]
            + self.stats["comments"]["deleted_author"]
            + self.stats["comments"]["bot_author"]
        )

        print(
            f"Comments: {original_count} -> {len(df)} "
            f"({len(df)/original_count*100:.1f}% kept)"
        )

        return df

    def _fix_column_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix mixed-type columns for parquet compatibility."""
        # Fix 'edited' column: can be False (bool) or timestamp (int)
        if "edited" in df.columns:
            # Convert to float: False -> 0.0, timestamps stay as float
            df["edited"] = df["edited"].apply(
                lambda x: 0.0 if x is False or x is None else float(x)
            )

        # Fix any other object columns that might have mixed types
        for col in df.select_dtypes(include=["object"]).columns:
            # Convert to string to ensure consistency
            if col not in ["body", "body_clean", "title", "title_clean", "selftext", "selftext_clean", "full_text"]:
                df[col] = df[col].astype(str)

        return df

    def save_data(
        self,
        submissions_df: pd.DataFrame,
        comments_df: pd.DataFrame,
    ) -> Tuple[Path, Path]:
        """Save preprocessed data to files."""
        output_dir = self.config.output_dir

        # Fix column types for parquet compatibility
        submissions_df = self._fix_column_types(submissions_df)
        comments_df = self._fix_column_types(comments_df)

        if self.config.output_format == "parquet":
            sub_path = output_dir / "submissions_clean.parquet"
            com_path = output_dir / "comments_clean.parquet"
            submissions_df.to_parquet(sub_path, index=False)
            comments_df.to_parquet(com_path, index=False)
        elif self.config.output_format == "csv":
            sub_path = output_dir / "submissions_clean.csv"
            com_path = output_dir / "comments_clean.csv"
            submissions_df.to_csv(sub_path, index=False)
            comments_df.to_csv(com_path, index=False)
        else:  # json
            sub_path = output_dir / "submissions_clean.json"
            com_path = output_dir / "comments_clean.json"
            submissions_df.to_json(sub_path, orient="records", lines=True)
            comments_df.to_json(com_path, orient="records", lines=True)

        print(f"\nSaved preprocessed data to {output_dir}")
        print(f"  - Submissions: {sub_path}")
        print(f"  - Comments: {com_path}")

        return sub_path, com_path

    def print_stats(self):
        """Print preprocessing statistics."""
        print("\n" + "=" * 60)
        print("PREPROCESSING STATISTICS")
        print("=" * 60)

        for data_type in ["submissions", "comments"]:
            stats = self.stats[data_type]
            print(f"\n{data_type.upper()}:")
            print(f"  Total:           {stats['total']:,}")
            print(f"  Kept:            {stats['kept']:,} ({stats['kept']/stats['total']*100:.1f}%)")
            print(f"  Deleted content: {stats['deleted_content']:,}")
            print(f"  Deleted author:  {stats['deleted_author']:,}")
            print(f"  Bot author:      {stats['bot_author']:,}")
            if data_type == "comments":
                print(f"  Too short:       {stats['too_short']:,}")

        total_original = self.stats["submissions"]["total"] + self.stats["comments"]["total"]
        total_kept = self.stats["submissions"]["kept"] + self.stats["comments"]["kept"]
        print(f"\nTOTAL: {total_original:,} -> {total_kept:,} ({total_kept/total_original*100:.1f}% kept)")
        print("=" * 60)

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run full preprocessing pipeline."""
        print("Starting preprocessing pipeline...")
        print(f"Config: remove_deleted={self.config.remove_deleted}, "
              f"remove_bots={self.config.remove_bots}, "
              f"min_words={self.config.min_word_count}")

        # Load data
        submissions_df = self.load_submissions()
        comments_df = self.load_comments()

        # Preprocess
        submissions_df = self.preprocess_submissions(submissions_df)
        comments_df = self.preprocess_comments(comments_df)

        # Save
        self.save_data(submissions_df, comments_df)

        # Print stats
        self.print_stats()

        return submissions_df, comments_df


def main():
    """Main entry point."""
    config = PreprocessConfig()
    preprocessor = RedditPreprocessor(config)
    preprocessor.run()


if __name__ == "__main__":
    main()
