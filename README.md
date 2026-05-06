# NJ Literacy Rates, COVID-19 Economic Impact, and Educational Recovery

# Overview

This repository analyzes New Jersey K–8 English Language Arts proficiency rates from 2017 to 2024. Specificlaly examining how the COVID-19 pandemic affected student literacy rates across districts with different socioeconomic profiles. 

The main argument is that economic disadvantage is the primary driver of slow literacy recovery in certain NJ regions, and that targeted funding to the most effected districts is a necessary addition to the state's new universal literacy screening programs.

The analysis proceeds in three parts:

Literacy trends by economic tier, COVID economic shock by region, and recovery gap analysis.



# Policy Context

In August 2024, New Jersey signed P.L. 2024, c.52 into law, mandating universal literacy screenings for all K–3 students twice annually beginning in the 2025–2026 school year. The legislation created the Office of Learning Equity and Academic Recovery (LEAR) and allocated $5.25 million for literacy initiatives statewide.
While universal screening is a meaningful step, this analysis shows that literacy proficiency gaps in NJ are strongly associated with pre-existing economic disadvantage, which was deepened by COVID-19. Districts in the lowest DFG tiers (A and B) have shown the slowest recovery trajectories and are furthest from the state's 80% proficiency goal. This suggests that a uniform screening approach, without differentiated funding and intervention support for the most affected regions, may be insufficient to close these gaps.

# Data Sources

This analysis uses twp publicly available datasets. All files can be downloaded directly from the links below.

1. NJ School Performance Reports (NJDOE)

District-level ELA proficiency rates for grades 3–8, broken down by student subgroup, for school years 2017–2018 through 2023–2024
Download:https://www.nj.gov/education/spr/download/

Instructions: Select each school year from the dropdown menu and download the district/state level ZIP file. Download all years from 2017–2018 through 2023–2024 and place them in data/raw/.
Note: The 2019–2020 and 2020–2021 files do not contain ELA assessment results because statewide assessments were cancelled during the COVID-19 pandemic. These years are retained for enrollment and demographic data only.

2. NJ District Factor Groups (NJDOE)

A classification of every NJ school district into one of eight socioeconomic tiers (A through J), from lowest to highest income, based on six census variables including poverty rate, unemployment, and median family income
Where to download: https://www.nj.gov/education/stateaid/dfg.shtml

Instructions: Download this single Excel file and place it in data/raw/ as DFG2000.xlsx

3. NJ County Unemployment Rates (BLS LAUS)

Annual average unemployment rates for all 21 NJ counties, 2018–2024, from the Bureau of Labor Statistics Local Area Unemployment Statistics program
Where to download: https://www.bls.gov/lau/data.htm

Instructions: Click "Multi-Screen Data Search," select State = New Jersey, Area Type = Counties, and select all 21 NJ counties. Select annual average data for 2018–2024. Download as CSV and save to data/raw/ as bls_county_unemployment_nj.csv

4. NJ School Distrcit Shapefile (NJDOE)
Gives a shapefile for the school distrcits across the state.

Download: https://njogis-newjersey.opendata.arcgis.com/datasets/newjersey::school-districts-unified-for-nj-3424/about

Note: DFG classifications have not been updated since 2004 census data. They are used here as a stable structural measure of district socioeconomic status, not as a current demographic snapshot.


# How to Reproduce This Analysis

## Prerequisites
You will need Python 3.8 or higher. All required packages are listed in requirements.txt.

Step 1 — Clone the repository

Step 2 — Install dependencies

Step 3 — Download the raw data

Step 4 — Run the scripts in order

Each script must be run in order. Script 1 produces the cleaned CSVs that Scripts 2, 3, and 4 depend on.

# Script Descriptions

## 1) Clean_data.py
Loads the raw NJDOE School Performance Report ZIP files for each year, extracts the district-level ELA assessment data, and merges it with the District Factor Group classifications. Filters to grades 3–8 ELA proficiency rates. Handles the missing assessment years (2019–2020 and 2020–2021) by retaining those rows with null proficiency values so the timeline remains continuous. Outputs three cleaned CSV files to data/processed/.

## 2) Analysis.py
Performs the core analysis in two parts. First, it calculates the COVID recovery gap for each DFG tier — defined as the difference between the 2018–2019 pre-COVID proficiency baseline and the 2023–2024 proficiency rate. Second, it runs an ordinary least squares (OLS) regression with 2023–2024 ELA proficiency as the outcome variable and peak COVID unemployment rate (2020 annual average by county) as the primary predictor, controlling for pre-COVID proficiency baseline. Prints regression results and summary statistics to the console.

## 3) AnalysisVisualization.py
Produces figures saved to output/figures/:

## 4) LiteracyBaseHeatMaps.py
Produces three heat maps showing student ELA proficiency across the state. 


# Results


Key Finding 1: Pre-COVID proficiency gaps were already substantial. Prior to the pandemic, ELA proficiency rates varied dramatically across DFG tiers. Districts in the lowest tiers (A and B) showed proficiency rates approximately 50 percentage points below the highest tiers (I and J) in 2018–2019.

Key Finding 2: COVID widened the gap unevenly. The pandemic-era drop in proficiency was not uniform. Lower DFG districts experienced drops of approximately 6 to 8 percentage points compared to  2-5 points in higher DFG districts, suggesting that remote learning infrastructure, family economic stability, and access to academic support were less available in poorer districts.

Key Finding 3: Economic shock explains recovery speed. The regression analysis found that Economic Disadvantage rates in 2020 were a statistically significant predictor of 2023–2024 ELA proficiency (β = -0.096, p = < .05), even after controlling for pre-COVID baseline proficiency. Counties that experienced larger unemployment spikes show slower literacy recovery.
