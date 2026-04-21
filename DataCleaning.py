"""
01_clean_data.py
----------------
NJ Literacy & COVID Economic Impact — Data Cleaning Script

Cleans and merges three raw datasets into analysis-ready CSVs:
  1. NJ School Performance Reports (SPR) — 7 Excel files, 2017-18 through 2023-24
  2. NJ District Factor Groups (DFG)     — Socioeconomic tier per district
  3. NJ Labor Force Monthly Data         — County unemployment rates by year

Each SPR file has a slightly different structure due to COVID-era reporting
changes. This script handles every year-specific quirk automatically.

Outputs written to data/processed/:
  ela_by_district_year.csv   — District ELA proficiency + DFG tier, all years
  ela_by_dfg_year.csv        — ELA proficiency averaged by DFG tier and year
  county_unemployment.csv    — Annual avg unemployment by county and year

Usage:
  Place all raw files in data/raw/ then run:
  python scripts/01_clean_data.py

Requirements:
  pip install pandas openpyxl
"""

import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

RAW_DIR  = os.path.join("data", "raw")
PROC_DIR = os.path.join("data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)

SPR_FILES = {
    "2017-2018": os.path.join(RAW_DIR, "spr_district_2017-18.xlsx"),
    "2018-2019": os.path.join(RAW_DIR, "spr_district_2018-19_.xlsx"),
    "2019-2020": os.path.join(RAW_DIR, "spr_district_2019-20.xlsx"),
    "2020-2021": os.path.join(RAW_DIR, "spr_district_2020-21.xlsx"),
    "2021-2022": os.path.join(RAW_DIR, "spr_district_2021-22.xlsx"),
    "2022-2023": os.path.join(RAW_DIR, "spr_district_2022-23.xlsx"),
    "2023-2024": os.path.join(RAW_DIR, "spr_district_2023-24.xlsx"),
}

DFG_FILE   = os.path.join(RAW_DIR, "DFG2000.xlsx")
UNEMP_FILE = os.path.join(RAW_DIR, "lfmnth_Historical_data_2010-2024.xlsx")

DFG_ORDER = ["A", "B", "CD", "DE", "FG", "GH", "I", "J"]


def clean_dfg(filepath):
    print("  Loading District Factor Groups...")
    df = pd.read_excel(filepath, sheet_name="VERS2", header=0)
    df.columns = ["county_code", "county_name", "district_code",
                  "district_name", "dfg", "dfg_1990"]
    df = df[["county_code", "district_code", "district_name", "dfg"]].copy()
    df = df[df["dfg"].notna()].copy()
    df["district_code"] = df["district_code"].astype(int).astype(str).str.zfill(4)
    df["county_code"]   = df["county_code"].astype(int).astype(str).str.zfill(2)
    df["dfg"]           = df["dfg"].str.strip().str.upper()
    dfg_rank_map = {tier: i + 1 for i, tier in enumerate(DFG_ORDER)}
    df["dfg_rank"] = df["dfg"].map(dfg_rank_map)
    print(f"    -> {len(df)} districts with valid DFG assignments")
    return df.reset_index(drop=True)


def _to_numeric(series):
    return pd.to_numeric(
        series.astype(str).str.replace(r"[<=><>%*N]", "", regex=True).str.strip(),
        errors="coerce"
    )


def _get_econ_disadvantaged(xl, year):
    sheet = "EnrollmentTrendsByStudentGroup"
    if sheet not in xl.sheet_names:
        return pd.DataFrame()
    df = xl.parse(sheet, header=0, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    econ_col = "Economically Disadvantaged Students"
    if econ_col not in df.columns:
        return pd.DataFrame()
    out = df[["CountyCode", "DistrictCode", econ_col]].copy()
    out.rename(columns={"CountyCode": "county_code", "DistrictCode": "district_code",
                        econ_col: "econ_disadvantaged_pct"}, inplace=True)
    out["econ_disadvantaged_pct"] = _to_numeric(out["econ_disadvantaged_pct"])
    out["district_code"] = out["district_code"].astype(str).str.zfill(4)
    out["county_code"]   = out["county_code"].astype(str).str.zfill(2)
    out = out.groupby(["county_code", "district_code"], as_index=False)["econ_disadvantaged_pct"].mean()
    return out


def extract_ela_2017_18(xl, year):
    df = xl.parse("ELAPerformanceTrends", header=0, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    ela = df[df["Subject"].str.upper() == "LAL"].copy()
    ela["ela_proficiency_pct"] = _to_numeric(ela["MetExcExpPerc"])
    ela["district_code"] = ela["DistrictCode"].astype(str).str.zfill(4)
    ela["county_code"]   = ela["CountyCode"].astype(str).str.zfill(2)
    agg = (ela.groupby(["county_code", "DistrictName", "district_code"], as_index=False)
              ["ela_proficiency_pct"].mean()
              .rename(columns={"DistrictName": "district_name"}))
    agg["school_year"] = year
    return agg[["school_year", "county_code", "district_code", "district_name", "ela_proficiency_pct"]]


def extract_ela_with_school_year(xl, year, target_year):
    sheet = "ELAMathPerformanceTrends"
    df = xl.parse(sheet, header=0, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    ela = df[(df["Subject"].str.upper() == "ELA") & (df["SchoolYear"] == target_year)].copy()
    if ela.empty:
        ela = df[df["Subject"].str.upper() == "ELA"].copy()
    ela["ela_proficiency_pct"] = _to_numeric(ela["ProficiencyRateforFederalAccountability"])
    ela["district_code"] = ela["DistrictCode"].astype(str).str.zfill(4)
    ela["county_code"]   = ela["CountyCode"].astype(str).str.zfill(2)
    ela["school_year"]   = year
    return ela[["school_year", "county_code", "district_code", "DistrictName", "ela_proficiency_pct"]].rename(columns={"DistrictName": "district_name"})


def clean_spr_year(year, filepath):
    xl = pd.ExcelFile(filepath)
    COVID_NO_TEST = {"2019-2020", "2020-2021"}

    if year == "2017-2018":
        ela_df = extract_ela_2017_18(xl, year)
    elif year in COVID_NO_TEST:
        print(f"    Warning: {year}: No NJSLA assessment (COVID year) - ELA proficiency will be NaN")
        enr = xl.parse("EnrollmentTrendsByStudentGroup", header=0, dtype=str)
        enr.columns = [c.strip() for c in enr.columns]
        ela_df = enr[["CountyCode", "DistrictCode", "DistrictName"]].copy()
        ela_df.rename(columns={"CountyCode": "county_code", "DistrictCode": "district_code",
                                "DistrictName": "district_name"}, inplace=True)
        ela_df["ela_proficiency_pct"] = float("nan")
        ela_df["school_year"] = year
        ela_df["district_code"] = ela_df["district_code"].astype(str).str.zfill(4)
        ela_df["county_code"]   = ela_df["county_code"].astype(str).str.zfill(2)
    else:
        ela_df = extract_ela_with_school_year(xl, year, year)

    econ_df = _get_econ_disadvantaged(xl, year)

    if not econ_df.empty:
        result = ela_df.merge(econ_df, on=["county_code", "district_code"], how="left")
    else:
        ela_df["econ_disadvantaged_pct"] = float("nan")
        result = ela_df.copy()

    for col in ["school_year", "county_code", "district_code", "district_name",
                "ela_proficiency_pct", "econ_disadvantaged_pct"]:
        if col not in result.columns:
            result[col] = float("nan") if "pct" in col else ""

    return result[["school_year", "county_code", "district_code", "district_name",
                   "ela_proficiency_pct", "econ_disadvantaged_pct"]]


def clean_all_spr():
    print("\n  Loading SPR school performance files...")
    frames = []
    for year, filepath in SPR_FILES.items():
        if not os.path.exists(filepath):
            print(f"    Warning: File not found: {filepath} - skipping {year}")
            continue
        print(f"    Processing {os.path.basename(filepath)} -> {year}")
        df = clean_spr_year(year, filepath)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["year_start"] = combined["school_year"].str[:4].astype(int)
    combined.sort_values(["district_code", "year_start"], inplace=True)
    combined.drop(columns=["year_start"], inplace=True)
    print(f"    -> {len(combined):,} district-year rows "
          f"({combined['district_code'].nunique()} districts x {combined['school_year'].nunique()} years)")
    return combined.reset_index(drop=True)


def clean_unemployment(filepath):
    print("\n  Loading unemployment data...")
    xl = pd.ExcelFile(filepath)
    target_years = [str(y) for y in range(2017, 2025)]
    available = [y for y in target_years if y in xl.sheet_names]
    if not available:
        print(f"    No matching year sheets found.")
        return pd.DataFrame()
    records = []
    for year_str in available:
        raw = xl.parse(year_str, header=None)
        header_row_idx = None
        for idx, row in raw.iterrows():
            if any("COUNTY/LABOR AREA" in str(v).upper() for v in row if pd.notna(v)):
                header_row_idx = idx
                break
        if header_row_idx is None:
            continue
        data = raw.iloc[header_row_idx + 2:].reset_index(drop=True)
        county_name = None
        for _, row in data.iterrows():
            cell = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
            if ", NJ" in cell or "County" in cell:
                county_name = cell.replace(", NJ", "").replace(" County", "").strip()
                continue
            if "Unemployment Rate" in cell and county_name:
                ann_avg = row.iloc[13] if len(row) > 13 else None
                if pd.notna(ann_avg):
                    try:
                        records.append({"year": int(year_str), "county": county_name,
                                        "annual_avg_unemployment_rate": float(ann_avg)})
                    except (ValueError, TypeError):
                        pass
                county_name = None
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df.sort_values(["county", "year"], inplace=True)
    print(f"    -> {len(df):,} county-year records "
          f"({df['county'].nunique()} counties, {df['year'].min()}-{df['year'].max()})")
    return df.reset_index(drop=True)


def build_ela_district(spr_df, dfg_df):
    print("\n  Merging ELA data with District Factor Groups...")
    merged = spr_df.merge(dfg_df[["county_code", "district_code", "dfg", "dfg_rank"]],
                          on=["county_code", "district_code"], how="left")
    n_unmatched = merged["dfg"].isna().sum()
    if n_unmatched:
        print(f"    Note: {n_unmatched} rows without DFG (vocational/charter/special districts - expected)")
    period_map = {
        "2017-2018": "Pre-COVID", "2018-2019": "Pre-COVID",
        "2019-2020": "COVID (no assessment)", "2020-2021": "COVID (no assessment)",
        "2021-2022": "Recovery", "2022-2023": "Recovery", "2023-2024": "Recovery",
    }
    merged["period"] = merged["school_year"].map(period_map).fillna("Other")
    print(f"    -> {len(merged):,} district-year rows with DFG attached")
    return merged


def build_ela_by_dfg(ela_district):
    print("\n  Aggregating ELA proficiency by DFG tier and school year...")
    has_data = ela_district[ela_district["dfg"].notna()]
    agg = (has_data.groupby(["school_year", "dfg", "dfg_rank", "period"], as_index=False)
                   .agg(ela_proficiency_mean=("ela_proficiency_pct", "mean"),
                        ela_proficiency_median=("ela_proficiency_pct", "median"),
                        district_count=("district_code", "nunique"),
                        econ_disadvantaged_mean=("econ_disadvantaged_pct", "mean")))
    agg["year_start"] = agg["school_year"].str[:4].astype(int)
    agg.sort_values(["dfg_rank", "year_start"], inplace=True)
    agg.drop(columns=["year_start"], inplace=True)
    print(f"    -> {len(agg)} DFG-tier x school-year rows")
    return agg.reset_index(drop=True)


def main():
    print("=" * 62)
    print("NJ Literacy & COVID Economic Impact - 01_clean_data.py")
    print("=" * 62)

    dfg_df = clean_dfg(DFG_FILE) if os.path.exists(DFG_FILE) else pd.DataFrame()
    spr_df = clean_all_spr()
    unemp_df = clean_unemployment(UNEMP_FILE) if os.path.exists(UNEMP_FILE) else pd.DataFrame()

    if not spr_df.empty and not dfg_df.empty:
        ela_district = build_ela_district(spr_df, dfg_df)
        ela_dfg = build_ela_by_dfg(ela_district)
    else:
        ela_district = spr_df.copy()
        ela_dfg = pd.DataFrame()

    print("\n  Saving processed files...")
    outputs = {"ela_by_district_year.csv": ela_district,
               "ela_by_dfg_year.csv": ela_dfg,
               "county_unemployment.csv": unemp_df}
    for fname, df in outputs.items():
        path = os.path.join(PROC_DIR, fname)
        if df.empty:
            print(f"    Warning: Skipping {fname} - no data")
        else:
            df.to_csv(path, index=False)
            print(f"    Saved: {fname}  ({len(df):,} rows)")

    print("\n" + "=" * 62)
    print("CLEANING COMPLETE")
    print("=" * 62)

    if not ela_dfg.empty:
        pre  = ela_dfg[ela_dfg["period"] == "Pre-COVID"].groupby("dfg")["ela_proficiency_mean"].mean()
        post = ela_dfg[ela_dfg["period"] == "Recovery"].groupby("dfg")["ela_proficiency_mean"].mean()
        print("\n  ELA Proficiency Summary (Pre-COVID vs Recovery):")
        print(f"  {'DFG':>5} | {'Pre-COVID%':>10} | {'Recovery%':>10} | {'Change':>9}")
        print(f"  {'-'*5}-+-{'-'*10}-+-{'-'*10}-+-{'-'*9}")
        for tier in DFG_ORDER:
            p  = pre.get(tier, float("nan"))
            r  = post.get(tier, float("nan"))
            ch = (r - p) if not (pd.isna(p) or pd.isna(r)) else float("nan")
            p_s  = f"{p:.1f}%" if not pd.isna(p) else "N/A"
            r_s  = f"{r:.1f}%" if not pd.isna(r) else "N/A"
            ch_s = f"{ch:+.1f}pp" if not pd.isna(ch) else "N/A"
            print(f"  {tier:>5} | {p_s:>10} | {r_s:>10} | {ch_s:>9}")

    if not unemp_df.empty:
        peak = unemp_df.loc[unemp_df["annual_avg_unemployment_rate"].idxmax()]
        print(f"\nPeak unemployment: {peak['county']} in {peak['year']} ({peak['annual_avg_unemployment_rate']:.1f}%)")

    print(f"\nOutputs saved to: {PROC_DIR}/")
    print("Next step: python scripts/02_analysis.py")


if __name__ == "__main__":
    main()