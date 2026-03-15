"""
src/data.py
Data loading, cleaning, and preprocessing for KDHS ECD project
==============================================================
Centralized path management and data utilities
"""

from pathlib import Path
import pandas as pd
import numpy as np
import pyreadstat
from typing import Tuple, Dict, Optional

# ============================================================================
# PATHS
# ============================================================================
def get_project_root() -> Path:
    """Get project root directory"""
    return Path(__file__).resolve().parent.parent


def get_data_paths() -> Dict[str, Path]:
    """
    Get all project paths

    Returns:
        Dict with all directory paths
    """
    root = get_project_root()
    return {
        "root": root,
        "raw": root / "data" / "raw",
        "interim": root / "data" / "interim",
        "processed": root / "data" / "processed",
        "outputs": root / "data" / "outputs",
        "geo": root / "geo",
        "models": root / "models",
        "figures": root / "reports" / "figures",
        "tables": root / "reports" / "tables",
        "tuning": root / "reports" / "tuning",
        "reports": root / "reports",
        "configs": root / "configs",
        "logs": root / "logs",
        "notebooks": root / "notebooks",
        "src": root / "src",
        "api": root / "api",
        "dashboard": root / "dashboard",
    }


# ============================================================================
# COLUMN MAPPINGS
# ============================================================================
RENAME_MAP = {
    "B19": "child_age_months",
    "B4": "child_sex",
    "V012": "mother_age",
    "V106": "mother_education_level",
    "V133": "mother_years_education",
    "V714": "mother_currently_working",
    "V501": "marital_status",
    "V130": "religion",
    "V190": "wealth_quintile",
    "V025": "urban_rural",
    "V024": "county",
    "V113": "drinking_water_source",
    "V116": "toilet_facility",
    "V119": "has_electricity",
    "V161": "cooking_fuel",
    "V136": "household_members",
}

ECD_RENAME = {
    "ECD21": "phys_walk_uneven_surface",
    "ECD22": "phys_jump_two_feet",
    "ECD23": "phys_dress_self",
    "ECD24": "phys_buttons",
    "ECD25": "lang_words_10plus",
    "ECD26": "lang_sentence_3plus",
    "ECD27": "lang_sentence_5plus",
    "ECD28": "lang_use_pronouns",
    "ECD29": "lang_name_objects",
    "ECD30": "lit_letters_5plus",
    "ECD31": "lit_write_name",
    "ECD32": "lit_numbers_1_5",
    "ECD33": "lit_count_3_objects",
    "ECD34": "lit_count_10_objects",
    "ECD35": "soc_independent_activity",
    "ECD36": "soc_name_familiar_people",
    "ECD37": "soc_help_others",
    "ECD38": "soc_get_along_children",
    "ECD39": "soc_often_sad",
    "ECD40": "soc_aggressive_behavior",
}

FULL_RENAME_MAP = {**RENAME_MAP, **ECD_RENAME}

DOMAIN_ITEMS = {
    "physical": [
        "phys_walk_uneven_surface",
        "phys_jump_two_feet",
        "phys_dress_self",
        "phys_buttons",
    ],
    "language": [
        "lang_words_10plus",
        "lang_sentence_3plus",
        "lang_sentence_5plus",
        "lang_use_pronouns",
        "lang_name_objects",
    ],
    "literacy": [
        "lit_letters_5plus",
        "lit_write_name",
        "lit_numbers_1_5",
        "lit_count_3_objects",
        "lit_count_10_objects",
    ],
    "socio_emotional": [
        "soc_independent_activity",
        "soc_name_familiar_people",
        "soc_help_others",
        "soc_get_along_children",
        "soc_often_sad",
        "soc_aggressive_behavior",
    ],
}

ECD_COLS = list(ECD_RENAME.values())


# ============================================================================
# PRINT HELPERS
# ============================================================================
def print_section(title: str) -> None:
    """Pretty section header for terminal output"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_shape(label: str, df: pd.DataFrame) -> None:
    """Print rows and columns of a dataframe"""
    print(f"{label}")
    print(f"Rows    : {df.shape[0]:,}")
    print(f"Columns : {df.shape[1]:,}")


def print_value_counts(series: pd.Series, title: str, top_n: int = 10) -> None:
    """Print top value counts"""
    print(f"\n{title}")
    print(series.value_counts(dropna=False).head(top_n))


# ============================================================================
# DATA LOADING
# ============================================================================
def load_kdhs_data(kr_file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load KDHS KR file (.SAV format)

    Args:
        kr_file_path: Path to KEKR8CFL.SAV file. If None, uses default path.

    Returns:
        DataFrame with raw KDHS data
    """
    paths = get_data_paths()

    if kr_file_path is None:
        kr_file_path = paths["raw"] / "KEKR8CFL.SAV"

    if not kr_file_path.exists():
        raise FileNotFoundError(f"KDHS file not found: {kr_file_path}")

    df, meta = pyreadstat.read_sav(str(kr_file_path), apply_value_formats=True)
    print(f"✅ Loaded KDHS data from: {kr_file_path}")
    print_shape("RAW KDHS DATASET", df)

    return df


# ============================================================================
# DATA CLEANING
# ============================================================================
def rename_columns(df: pd.DataFrame, rename_map: Dict[str, str] = None) -> pd.DataFrame:
    """Rename KDHS columns to meaningful names"""
    if rename_map is None:
        rename_map = FULL_RENAME_MAP

    existing_map = {k: v for k, v in rename_map.items() if k in df.columns}
    df_renamed = df.rename(columns=existing_map)

    print(f"✅ Renamed columns: {len(existing_map)} columns matched and renamed")
    return df_renamed


def select_relevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only columns used in this project"""
    selected_cols = [c for c in FULL_RENAME_MAP.values() if c in df.columns]
    df_selected = df[selected_cols].copy()

    print(f"✅ Selected relevant variables: {len(selected_cols)} columns kept")
    return df_selected


def filter_age_range(df: pd.DataFrame, min_age: int = 36, max_age: int = 59) -> pd.DataFrame:
    """Filter children to target age range (36-59 months)"""
    before_rows = len(df)
    df_filtered = df[
        (df["child_age_months"] >= min_age) &
        (df["child_age_months"] <= max_age)
    ].copy()
    removed_rows = before_rows - len(df_filtered)

    print(f"✅ Filtered to age {min_age}-{max_age} months")
    print(f"   Removed rows outside target age: {removed_rows:,}")
    print_shape("AFTER AGE FILTER", df_filtered)

    return df_filtered


def remove_non_dejure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows flagged as 'Not a dejure resident'
    Checks across row values because value labels may appear in different columns.
    """
    def row_has_not_dejure_resident(row) -> bool:
        return row.astype(str).str.contains("Not a dejure resident", case=False, na=False).any()

    before_rows = len(df)
    mask_not_dejure = df.apply(row_has_not_dejure_resident, axis=1)
    flagged = int(mask_not_dejure.sum())

    df_clean = df.loc[~mask_not_dejure].copy()
    removed_rows = before_rows - len(df_clean)

    print(f"⚠️ Rows flagged as not-dejure: {flagged:,}")
    print(f"✅ Removed not-dejure rows: {removed_rows:,}")
    print_shape("AFTER REMOVING NON-DEJURE RESIDENTS", df_clean)

    return df_clean


def apply_skip_logic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply skip logic for ECD items.
    If 3-word sentence = No, then 5-word sentence can be filled as No when missing.
    """
    if "lang_sentence_3plus" not in df.columns or "lang_sentence_5plus" not in df.columns:
        print("⚠️ Skip logic not applied: required language columns missing")
        return df

    mask_fill = (
        df["lang_sentence_5plus"].isna() &
        (df["lang_sentence_3plus"] == "No")
    )
    filled_rows = int(mask_fill.sum())

    df.loc[mask_fill, "lang_sentence_5plus"] = "No"

    inconsistent = df[
        (df["lang_sentence_5plus"] == "Yes") &
        (df["lang_sentence_3plus"] == "No")
    ]

    print(f"✅ Applied skip logic")
    print(f"   Filled lang_sentence_5plus = 'No' for: {filled_rows:,} rows")

    if len(inconsistent) > 0:
        print(f"⚠️ Inconsistent rows found (5-word = Yes while 3-word = No): {len(inconsistent):,}")
    else:
        print("✅ No skip-logic inconsistencies found")

    return df


def convert_categories_to_string(df: pd.DataFrame) -> pd.DataFrame:
    """Convert category columns to string for better compatibility"""
    cat_cols = df.select_dtypes(include="category").columns.tolist()

    if len(cat_cols) > 0:
        df[cat_cols] = df[cat_cols].astype("string")

    print(f"✅ Converted category columns to string: {len(cat_cols)}")
    return df


# ============================================================================
# ECD ITEM RECODING
# ============================================================================
def recode_ecd_items(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recode ECD items to numeric (0/1)
    1 = Pass/Achieved, 0 = Fail/Not Achieved
    """
    df_recode = df.copy()

    yes_no_map = {
        "Yes": 1,
        "No": 0,
        "Don't know": np.nan,
        "Don’t know": np.nan,
    }

    for c in ECD_COLS:
        if c in df_recode.columns:
            df_recode[c] = df_recode[c].map(yes_no_map)

    sad_map = {
        "Never": 1,
        "A few times a year": 1,
        "Monthly": 1,
        "Weekly": 0,
        "Daily": 0,
        "Don't know": np.nan,
        "Don’t know": np.nan,
    }

    aggressive_map = {
        "Not at all": 1,
        "The same or less": 1,
        "More": 0,
        "A lot more": 0,
        "Don't know": np.nan,
        "Don’t know": np.nan,
    }

    if "soc_often_sad" in df_recode.columns:
        df_recode["soc_often_sad"] = df_recode["soc_often_sad"].map(sad_map)

    if "soc_aggressive_behavior" in df_recode.columns:
        df_recode["soc_aggressive_behavior"] = df_recode["soc_aggressive_behavior"].map(aggressive_map)

    for c in ECD_COLS:
        if c in df_recode.columns:
            df_recode[c] = pd.to_numeric(df_recode[c], errors="coerce")

    print(f"✅ Recoded ECD items to numeric for {len([c for c in ECD_COLS if c in df_recode.columns])} columns")
    return df_recode


# ============================================================================
# DATA COMPLETENESS
# ============================================================================
def check_completeness(df: pd.DataFrame, ecd_cols: list = None) -> pd.DataFrame:
    """Check ECD data completeness"""
    if ecd_cols is None:
        ecd_cols = [c for c in ECD_COLS if c in df.columns]

    df_check = df.copy()
    df_check["complete_ecd_strict"] = df_check[ecd_cols].notna().all(axis=1)

    print("\n📊 ECD COMPLETENESS SUMMARY")
    print(df_check["complete_ecd_strict"].value_counts(dropna=False))
    print("\n📊 ECD COMPLETENESS PERCENT")
    print((df_check["complete_ecd_strict"].value_counts(normalize=True, dropna=False) * 100).round(2))

    return df_check


def print_missing_summary(df: pd.DataFrame, top_n: int = 20) -> None:
    """Print missing values summary"""
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]

    print("\n📊 MISSING VALUES SUMMARY")
    if missing.empty:
        print("No missing values found.")
        return

    print(missing.head(top_n))


def print_pipeline_summary(
    raw_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    age_df: pd.DataFrame,
    dejure_df: pd.DataFrame,
    final_df: pd.DataFrame,
) -> None:
    """Print dissertation-friendly pipeline counts"""
    print_section("DATA CLEANING PIPELINE SUMMARY")
    summary = pd.DataFrame(
        {
            "Step": [
                "Raw KDHS dataset",
                "After selecting relevant variables",
                "After age filter (36-59 months)",
                "After removing non-dejure residents",
                "Final complete ECD dataset",
            ],
            "Rows": [
                raw_df.shape[0],
                selected_df.shape[0],
                age_df.shape[0],
                dejure_df.shape[0],
                final_df.shape[0],
            ],
            "Columns": [
                raw_df.shape[1],
                selected_df.shape[1],
                age_df.shape[1],
                dejure_df.shape[1],
                final_df.shape[1],
            ],
        }
    )
    print(summary.to_string(index=False))


# ============================================================================
# FULL PIPELINE
# ============================================================================
def load_and_clean_data(
    kr_file_path: Optional[Path] = None,
    save_interim: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Path]]:
    """
    Complete data loading and cleaning pipeline

    Args:
        kr_file_path: Path to KDHS file
        save_interim: Whether to save intermediate files

    Returns:
        Tuple of (cleaned DataFrame with completeness flag, paths dict)
    """
    paths = get_data_paths()

    for d in [
        paths["raw"],
        paths["interim"],
        paths["processed"],
        paths["figures"],
        paths["tables"],
        paths["models"],
        paths["configs"],
        paths["logs"],
    ]:
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------
    # STEP 1: LOAD RAW DATA
    # ------------------------------------------------------------------------
    print_section("STEP 1: LOAD RAW KDHS DATA")
    raw_df = load_kdhs_data(kr_file_path)

    # ------------------------------------------------------------------------
    # STEP 2: RENAME COLUMNS
    # ------------------------------------------------------------------------
    print_section("STEP 2: RENAME COLUMNS")
    renamed_df = rename_columns(raw_df)
    print_shape("AFTER COLUMN RENAME", renamed_df)

    # ------------------------------------------------------------------------
    # STEP 3: SELECT RELEVANT VARIABLES
    # ------------------------------------------------------------------------
    print_section("STEP 3: SELECT RELEVANT VARIABLES")
    selected_df = select_relevant_columns(renamed_df)
    print_shape("AFTER SELECTING RELEVANT VARIABLES", selected_df)
    print("\nSelected columns:")
    print(selected_df.columns.tolist())

    # ------------------------------------------------------------------------
    # STEP 4: FILTER TARGET AGE RANGE
    # ------------------------------------------------------------------------
    print_section("STEP 4: FILTER AGE RANGE")
    age_df = filter_age_range(selected_df, min_age=36, max_age=59)

    # ------------------------------------------------------------------------
    # STEP 5: REMOVE NON-DEJURE RESIDENTS
    # ------------------------------------------------------------------------
    print_section("STEP 5: REMOVE NON-DEJURE RESIDENTS")
    dejure_df = remove_non_dejure(age_df)

    # ------------------------------------------------------------------------
    # STEP 6: APPLY SKIP LOGIC
    # ------------------------------------------------------------------------
    print_section("STEP 6: APPLY SKIP LOGIC")
    skip_df = apply_skip_logic(dejure_df.copy())
    print_shape("AFTER APPLYING SKIP LOGIC", skip_df)

    # ------------------------------------------------------------------------
    # STEP 7: CONVERT CATEGORY COLUMNS
    # ------------------------------------------------------------------------
    print_section("STEP 7: CONVERT CATEGORY COLUMNS")
    string_df = convert_categories_to_string(skip_df)
    print_shape("AFTER CATEGORY CONVERSION", string_df)

    # ------------------------------------------------------------------------
    # STEP 8: CHECK COMPLETENESS
    # ------------------------------------------------------------------------
    print_section("STEP 8: CHECK ECD COMPLETENESS")
    df = check_completeness(string_df)

    # final complete cases
    df_clean = df[df["complete_ecd_strict"]].copy()

    print("\n📊 FINAL COMPLETE ECD DATASET")
    print_shape("FINAL MODELLING DATASET", df_clean)

    # ------------------------------------------------------------------------
    # EXTRA OUTPUTS FOR REPORTING
    # ------------------------------------------------------------------------
    print_section("ADDITIONAL OUTPUTS FOR REPORTING")
    print_missing_summary(df, top_n=20)
    print_pipeline_summary(
        raw_df=raw_df,
        selected_df=selected_df,
        age_df=age_df,
        dejure_df=dejure_df,
        final_df=df_clean,
    )

    # ------------------------------------------------------------------------
    # SAVE INTERIM FILES
    # ------------------------------------------------------------------------
    if save_interim:
        selected_path = paths["interim"] / "kr_selected_36_59.parquet"
        clean_path = paths["interim"] / "kr_clean.parquet"
        summary_path = paths["tables"] / "data_cleaning_summary.csv"

        df.to_parquet(selected_path, index=False)
        df_clean.to_parquet(clean_path, index=False)

        summary_df = pd.DataFrame(
            {
                "step": [
                    "raw_kdhs_dataset",
                    "after_selecting_relevant_variables",
                    "after_age_filter_36_59_months",
                    "after_removing_non_dejure",
                    "final_complete_ecd_dataset",
                ],
                "rows": [
                    raw_df.shape[0],
                    selected_df.shape[0],
                    age_df.shape[0],
                    dejure_df.shape[0],
                    df_clean.shape[0],
                ],
                "columns": [
                    raw_df.shape[1],
                    selected_df.shape[1],
                    age_df.shape[1],
                    dejure_df.shape[1],
                    df_clean.shape[1],
                ],
            }
        )
        summary_df.to_csv(summary_path, index=False)

        print("\n✅ SAVED FILES")
        print(f"Selected/intermediate dataset : {selected_path}")
        print(f"Final cleaned dataset         : {clean_path}")
        print(f"Cleaning summary table        : {summary_path}")

    return df, paths


# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================
def sorted_bar(series, horizontal=True, percent=False, title="", xlabel="",
               figsize=(8, 6), return_fig=False):
    """
    Create sorted bar plot (Jupyter-compatible)

    Args:
        return_fig: If True, returns (fig, ax) instead of showing
    """
    import matplotlib.pyplot as plt

    s = series.sort_values(ascending=True).dropna()
    if percent:
        s = s * 100
        xlabel = xlabel or "Percent (%)"

    fig, ax = plt.subplots(figsize=figsize)

    if horizontal:
        s.plot(kind="barh", ax=ax, color="steelblue", edgecolor="black")
        ax.set_xlabel(xlabel or "Value")
        ax.set_ylabel(series.name or "Category")
    else:
        s.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
        ax.set_xlabel(series.name or "Category")
        ax.set_ylabel(xlabel or "Value")
        ax.tick_params(axis="x", rotation=45)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="y" if horizontal else "x", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if return_fig:
        return fig, ax
    else:
        plt.show()
        return None


def plot_top_counts(series, title, top_n=15):
    """Plot top counts for a series"""
    vc = series.value_counts(dropna=False).head(top_n)
    sorted_bar(vc, horizontal=True, percent=False, title=title, xlabel="Count")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print_section("RUNNING src/data.py")
    df, paths = load_and_clean_data(save_interim=True)

    print_section("QUICK PREVIEW OF OUTPUT DATA")
    print(df.head())

    if "complete_ecd_strict" in df.columns:
        final_df = df[df["complete_ecd_strict"]].copy()

        print("\nFirst 5 rows of final complete ECD dataset:")
        print(final_df.head())

        print("\nFinal dataset columns:")
        print(final_df.columns.tolist())

        print("\nFinal dataset shape:")
        print(final_df.shape)

    print("\n✅ Done.")