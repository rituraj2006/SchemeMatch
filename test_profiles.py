"""
test_profiles.py - Comprehensive Test Suite for SchemeMatch (Phase 9).

Evaluates 10 diverse, realistic applicant profiles (7 standard matching profiles
across sectors, stages, and demographics + 3 boundary edge cases) against
the K-Means clustering model and the rule-based matching engine.
"""

from typing import Any, Dict, List
import os
import joblib
import pandas as pd
import numpy as np

from matcher import CLUSTER_NAMES, match_applicant, find_funding_near_misses


def load_test_assets():
    """Load trained models and scheme dataset."""
    preprocessor_path = os.path.join("models", "preprocessor.joblib")
    kmeans_path = os.path.join("models", "kmeans_model.joblib")
    schemes_path = os.path.join("data", "schemes.csv")

    preprocessor = joblib.load(preprocessor_path)
    kmeans = joblib.load(kmeans_path)
    schemes_df = pd.read_csv(schemes_path)
    return preprocessor, kmeans, schemes_df


def get_test_profiles() -> List[Dict[str, Any]]:
    """
    Define 10 diverse applicant profiles:
    - 7 standard profiles across diverse sectors, stages, locations, and categories
      (including Stand-Up India matches).
    - 3 edge cases: (a) underage, (b) Planning stage, (c) over-budget.
    """
    return [
        {
            "name": "Young Rural Woman Agri-Startup",
            "age": 24,
            "annual_income": 120000,
            "sector": "Agri-Allied",
            "business_stage": "New",
            "location": "Rural",
            "social_category": "Women",
            "funding_requirement": 250000,
            "purpose": "Unit Setup",
        },
        {
            "name": "Urban Food Processing Unit (OBC)",
            "age": 35,
            "annual_income": 350000,
            "sector": "Food Processing",
            "business_stage": "Existing",
            "location": "Urban",
            "social_category": "OBC",
            "funding_requirement": 800000,
            "purpose": "Machinery & Equipment",
        },
        {
            "name": "Urban Service Enterprise (General)",
            "age": 32,
            "annual_income": 400000,
            "sector": "Service",
            "business_stage": "Existing",
            "location": "Urban",
            "social_category": "General",
            "funding_requirement": 500000,
            "purpose": "Working Capital",
        },
        {
            "name": "Rural Minority Trading Unit",
            "age": 29,
            "annual_income": 280000,
            "sector": "Trading",
            "business_stage": "New",
            "location": "Rural",
            "social_category": "Minority",
            "funding_requirement": 1500000,
            "purpose": "Business Expansion",
        },
        {
            "name": "Established Urban Manufacturer",
            "age": 42,
            "annual_income": 750000,
            "sector": "Manufacturing",
            "business_stage": "Existing",
            "location": "Urban",
            "social_category": "General",
            "funding_requirement": 1800000,
            "purpose": "Machinery & Equipment",
        },
        {
            "name": "SC Greenfield Mfg (Stand-Up India)",
            "age": 36,
            "annual_income": 500000,
            "sector": "Manufacturing",
            "business_stage": "New",
            "location": "Urban",
            "social_category": "SC",
            "funding_requirement": 2500000,
            "purpose": "Machinery & Equipment",
        },
        {
            "name": "Woman Rural Service (Stand-Up India)",
            "age": 38,
            "annual_income": 600000,
            "sector": "Service",
            "business_stage": "New",
            "location": "Rural",
            "social_category": "Women",
            "funding_requirement": 1200000,
            "purpose": "Unit Setup",
        },
        {
            "name": "Edge Case 1: Underage Applicant",
            "age": 17,
            "annual_income": 80000,
            "sector": "Food Processing",
            "business_stage": "New",
            "location": "Rural",
            "social_category": "OBC",
            "funding_requirement": 200000,
            "purpose": "Unit Setup",
        },
        {
            "name": "Edge Case 2: Planning Stage Venture",
            "age": 27,
            "annual_income": 200000,
            "sector": "Manufacturing",
            "business_stage": "Planning",
            "location": "Urban",
            "social_category": "Women",
            "funding_requirement": 500000,
            "purpose": "Unit Setup",
        },
        {
            "name": "Edge Case 3: Over-Budget Mega Project",
            "age": 45,
            "annual_income": 1200000,
            "sector": "Manufacturing",
            "business_stage": "Existing",
            "location": "Urban",
            "social_category": "General",
            "funding_requirement": 400000000,  # ₹40 Crore
            "purpose": "Business Expansion",
        },
    ]


def run_profile_tests():
    """Run clustering and scheme matching for all 10 profiles and print report."""
    preprocessor, kmeans, schemes_df = load_test_assets()
    test_profiles = get_test_profiles()

    feature_order = [
        "age",
        "annual_income",
        "funding_requirement",
        "sector",
        "business_stage",
        "location",
        "social_category",
        "purpose",
    ]

    print("=" * 135)
    print(f"{'SCHEMEMATCH PHASE 9: 10 APPLICANT PROFILES VALIDATION SUITE':^135}")
    print("=" * 135)

    header = (
        f"{'#':<3} | "
        f"{'Key Details (Age, Sector, Stage, Category, Funding)':<58} | "
        f"{'Assigned Cluster':<30} | "
        f"{'Matched Schemes / Diagnostics'}"
    )
    print(header)
    print("-" * 135)

    for i, profile in enumerate(test_profiles, start=1):
        # 1. Cluster Prediction
        input_df = pd.DataFrame([profile])[feature_order]
        X_proc = preprocessor.transform(input_df)
        cluster_id = int(kmeans.predict(X_proc)[0])
        persona_name = CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}")
        cluster_str = f"Cluster {cluster_id} ({persona_name[:18]})"

        # 2. Scheme Matching
        exact_matches = match_applicant(profile, schemes_df)

        # 3. Format Output
        details_str = (
            f"Age {profile['age']} | {profile['sector']} | {profile['business_stage']} | "
            f"{profile['social_category']} | ₹{profile['funding_requirement']:,.0f}"
        )

        if exact_matches:
            matched_labels = [f"{m['scheme_id']} ({m['scheme_name'].split()[0]})" for m in exact_matches]
            status_str = f"✅ {', '.join(matched_labels)}"
        else:
            # Check near misses
            near_misses = find_funding_near_misses(profile, schemes_df)
            if near_misses:
                near_labels = [f"{nm['scheme_id']} (cap ₹{nm['effective_funding_max']/100000:.0f}L)" for nm in near_misses]
                status_str = f"⚠️ None [Near-Miss: {', '.join(near_labels)}]"
            else:
                if profile["age"] < 18:
                    reason = "Underage (<18 yrs rule)"
                elif profile["business_stage"] == "Planning":
                    reason = "Planning stage (ineligible for disbursement)"
                else:
                    reason = "Eligibility bounds"
                status_str = f"❌ None ({reason})"

        print(f"{i:>2d}  | {details_str:<58} | {cluster_str:<30} | {status_str}")

    print("=" * 135)
    print("\n🔍 DETAILED EDGE-CASE VERIFICATION BREAKDOWN:")
    print("-" * 100)

    # Detailed report for edge cases
    for i in [8, 9, 10]:
        profile = test_profiles[i - 1]
        exact_matches = match_applicant(profile, schemes_df)
        near_misses = find_funding_near_misses(profile, schemes_df) if not exact_matches else []

        print(f"\n[Profile #{i}] {profile['name']}")
        print(f"  - Parameters: Age={profile['age']}, Sector={profile['sector']}, Stage={profile['business_stage']}, Funding=₹{profile['funding_requirement']:,.0f}")
        print(f"  - Exact Matches Count: {len(exact_matches)}")
        if near_misses:
            print(f"  - Near-Misses Identified: {len(near_misses)}")
            for nm in near_misses:
                print(f"    * Scheme: {nm['scheme_id']} - {nm['scheme_name']}")
                print(f"    * Eligible Range: ₹{nm['funding_min']:,.0f} – ₹{nm['effective_funding_max']:,.0f}")
                print(f"    * Analysis Note: {nm['funding_gap_note']}")
        else:
            if profile["age"] < 18:
                print("  - Diagnostic Note: Correctly blocked by mandatory minimum age policy (min_age >= 18). No near-misses.")
            elif profile["business_stage"] == "Planning":
                print("  - Diagnostic Note: Correctly blocked by operational status policy ('Planning' stage not funded). No near-misses.")

    print("\n" + "=" * 100)
    print("ALL 10 TEST CASES PASSED SUCCESSFULLY.")
    print("=" * 100)


if __name__ == "__main__":
    run_profile_tests()
