"""
matcher.py - Rule-Based Government Scheme Matching Engine for SchemeMatch.

Phase 7: Evaluates entrepreneur applicant profiles against verified Indian
government schemes (PMEGP, PMFME, PMMY Mudra, Stand-Up India).
"""

from typing import Any, Dict, List, Optional
import os
import joblib
import pandas as pd
import numpy as np

# Mapping of K-Means cluster IDs (K=4) to interpreted business personas
CLUSTER_NAMES: Dict[int, str] = {
    0: "Urban Service Professionals",
    1: "Young Rural Micro-Startups",
    2: "Established Industrialists",
    3: "Rural Growth Enterprises",
}


def match_applicant(profile: Dict[str, Any], schemes_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Match an applicant profile against government schemes based on mandatory criteria.

    Parameters
    ----------
    profile : dict
        Applicant attributes: age, annual_income, sector, business_stage,
        location, social_category, funding_requirement, purpose.
    schemes_df : pd.DataFrame
        DataFrame of government schemes loaded from data/schemes.csv.

    Returns
    -------
    list of dict
        List of matched schemes, each containing scheme metadata,
        matched reasons, and critical verification checklist items.
    """
    matched_schemes: List[Dict[str, Any]] = []

    # Extract applicant features with defensive fallbacks
    try:
        age = int(profile.get("age", 0))
    except (ValueError, TypeError):
        age = 0

    try:
        funding_req = float(profile.get("funding_requirement", 0))
    except (ValueError, TypeError):
        funding_req = 0.0

    sector = str(profile.get("sector", "")).strip()
    business_stage = str(profile.get("business_stage", "")).strip()
    location = str(profile.get("location", "")).strip()
    social_category = str(profile.get("social_category", "")).strip()

    for _, scheme in schemes_df.iterrows():
        scheme_id = str(scheme["scheme_id"]).strip()
        scheme_name = str(scheme["scheme_name"]).strip()
        min_age = int(scheme["min_age"])
        funding_min = float(scheme["funding_min"])
        funding_max = float(scheme["funding_max"])
        support_type = str(scheme["support_type"]).strip()
        official_url = str(scheme["official_url"]).strip()
        last_verified = str(scheme["last_verified"]).strip()

        matched_reasons: List[str] = []
        verification_needed: List[str] = []

        # ------------------------------------------------------------------
        # 1. Minimum Age Check
        # ------------------------------------------------------------------
        if age < min_age:
            continue
        matched_reasons.append(
            f"Age ({age} yrs) meets scheme minimum requirement of {min_age}+ years."
        )

        # ------------------------------------------------------------------
        # 2. Target Sector Check
        # ------------------------------------------------------------------
        target_sectors = [s.strip() for s in str(scheme["target_sector"]).split(";")]
        if sector not in target_sectors:
            continue
        matched_reasons.append(
            f"Sector '{sector}' is supported (target sectors: {scheme['target_sector']})."
        )

        # ------------------------------------------------------------------
        # 3. Business Stage Check
        # Rule: 'Planning' must NOT match New or Existing.
        # For PMFME: 'New (ODOP)' matches 'New' with an ODOP verification note.
        # ------------------------------------------------------------------
        scheme_stages = [s.strip() for s in str(scheme["business_stage"]).split(";")]
        if business_stage == "Planning":
            continue
        elif business_stage == "New":
            has_new = "New" in scheme_stages
            has_new_odop = any(s.startswith("New (ODOP)") for s in scheme_stages)
            if not (has_new or has_new_odop):
                continue
            matched_reasons.append("Business stage 'New' qualifies for scheme support.")
        elif business_stage == "Existing":
            if "Existing" not in scheme_stages:
                continue
            matched_reasons.append("Business stage 'Existing' qualifies for scheme support.")
        else:
            continue

        # ------------------------------------------------------------------
        # 4. Social Category Check
        # ------------------------------------------------------------------
        allowed_categories = [c.strip() for c in str(scheme["social_category"]).split(";")]
        if "All" in allowed_categories:
            matched_reasons.append(
                f"Open to all social categories (applicant is '{social_category}')."
            )
        elif social_category in allowed_categories:
            matched_reasons.append(
                f"Applicant category '{social_category}' satisfies scheme eligibility ({scheme['social_category']})."
            )
        else:
            continue

        # ------------------------------------------------------------------
        # 5. Location Condition Check
        # ------------------------------------------------------------------
        allowed_locations = [loc.strip() for loc in str(scheme["location_condition"]).split(";")]
        if location not in allowed_locations:
            continue
        matched_reasons.append(
            f"Location '{location}' is eligible under scheme geographic scope."
        )

        # ------------------------------------------------------------------
        # 6. Funding Requirement Limits Check
        # Rule: Enforce PMEGP's ₹20,00,000 cap when sector is Service or Trading.
        # ------------------------------------------------------------------
        effective_funding_max = funding_max
        if scheme_id == "SCH01" and sector in ["Service", "Trading"]:
            effective_funding_max = 2000000.0  # ₹20 Lakhs cap for Service / Trading in PMEGP

        if not (funding_min <= funding_req <= effective_funding_max):
            continue

        if scheme_id == "SCH01" and sector in ["Service", "Trading"]:
            matched_reasons.append(
                f"Funding need ₹{funding_req:,.0f} falls within PMEGP Service/Trading sector limit (max ₹20,00,000)."
            )
        else:
            matched_reasons.append(
                f"Funding need ₹{funding_req:,.0f} is within eligible range (₹{funding_min:,.0f} – ₹{effective_funding_max:,.0f})."
            )

        # ------------------------------------------------------------------
        # 7. Verification Checklist Builder
        # ------------------------------------------------------------------
        if scheme_id == "SCH01":  # PMEGP
            if sector == "Manufacturing" and funding_req > 1000000:
                verification_needed.append(
                    "Education: Minimum Class 8 pass certificate mandatory for manufacturing projects > ₹10 Lakhs."
                )
            elif sector in ["Service", "Trading"] and funding_req > 500000:
                verification_needed.append(
                    "Education: Minimum Class 8 pass certificate mandatory for service/trading projects > ₹5 Lakhs."
                )

            if location == "Rural" or social_category in ["SC", "ST", "OBC", "Women", "Minority"]:
                verification_needed.append(
                    "Subsidy Rate: Special Category / Rural higher margin money subsidy (25%-35%); category/rural certificate required."
                )
            else:
                verification_needed.append(
                    "Subsidy Rate: General Category standard margin money subsidy (15% Urban / 25% Rural); 10% own contribution required."
                )

            verification_needed.append(
                "Training & Setup: Strictly new units only; mandatory completion of EDP training before subsidy disbursement."
            )
            verification_needed.append(
                "Bank Clearance: Detailed Project Report (DPR) and bank branch credit appraisal required."
            )

        elif scheme_id == "SCH02":  # PMFME
            verification_needed.append(
                "Enterprise Type: Must be an unincorporated micro food processing enterprise (< 10 workers)."
            )
            verification_needed.append(
                "Education & Contribution: Individual 18+ with Class 8 pass; minimum 10% beneficiary contribution."
            )
            verification_needed.append(
                "Family Restriction: Maximum 1 beneficiary per family eligible for financial support."
            )
            if business_stage == "New":
                verification_needed.append(
                    "ODOP Compliance: New units supported strictly for products under the designated One District One Product (ODOP) list."
                )
            elif business_stage == "Existing":
                verification_needed.append(
                    "Formalization: Existing unit must undertake formalization, FSSAI compliance, and technological upgrade."
                )

        elif scheme_id == "SCH03":  # PMMY (Mudra)
            verification_needed.append(
                "Entity Type: Non-corporate, non-farm small/micro business in manufacturing, trading, or service."
            )
            if funding_req <= 50000:
                verification_needed.append(
                    "Mudra Tier: Shishu (loans up to ₹50,000) — simplified documentation, zero collateral."
                )
            elif funding_req <= 500000:
                verification_needed.append(
                    "Mudra Tier: Kishore (loans ₹50,000 – ₹5,00,000) — machinery/equipment quotations and business proof required."
                )
            elif funding_req <= 1000000:
                verification_needed.append(
                    "Mudra Tier: Tarun (loans ₹5,00,000 – ₹10,00,000) — project proposal, bank statements, and tax documents required."
                )
            else:
                verification_needed.append(
                    "Mudra Tier: Tarun Plus (loans ₹10,00,000 – ₹20,00,000) — repeat borrower verification required: must have successfully repaid a prior Tarun loan."
                )
            verification_needed.append(
                "Security: Collateral-free loan backed by Credit Guarantee Fund for Micro Units (CGFMU)."
            )

        elif scheme_id == "SCH04":  # Stand-Up India
            verification_needed.append(
                "Beneficiary Criteria: SC, ST, and/or Woman entrepreneur (>= 51% stake held by SC/ST/Women if non-individual firm)."
            )
            verification_needed.append(
                "Enterprise Status: Greenfield (brand new) enterprise setup only."
            )
            verification_needed.append(
                "Credit Compliance: Borrower must not be a defaulter at any commercial bank or financial institution."
            )
            verification_needed.append(
                "Live Portal Status: Verify current portal status and apply through the official Stand-Up India portal before filing."
            )

        matched_schemes.append({
            "scheme_id": scheme_id,
            "scheme_name": scheme_name,
            "support_type": support_type,
            "matched_reasons": matched_reasons,
            "verification_needed": verification_needed,
            "official_url": official_url,
            "last_verified": last_verified,
        })

    return matched_schemes


def find_funding_near_misses(profile: Dict[str, Any], schemes_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Identify government schemes where the applicant satisfies all non-funding
    eligibility criteria (age, sector, business stage, social category, location)
    but fails strictly because the funding requirement is outside the scheme's limits.

    Parameters
    ----------
    profile : dict
        Applicant attributes: age, annual_income, sector, business_stage,
        location, social_category, funding_requirement, purpose.
    schemes_df : pd.DataFrame
        DataFrame of government schemes loaded from data/schemes.csv.

    Returns
    -------
    list of dict
        List of near-miss schemes containing:
        scheme_id, scheme_name, support_type, funding_min, effective_funding_max,
        official_url, and funding_gap_note.
    """
    near_misses: List[Dict[str, Any]] = []

    try:
        age = int(profile.get("age", 0))
    except (ValueError, TypeError):
        age = 0

    try:
        funding_req = float(profile.get("funding_requirement", 0))
    except (ValueError, TypeError):
        funding_req = 0.0

    sector = str(profile.get("sector", "")).strip()
    business_stage = str(profile.get("business_stage", "")).strip()
    location = str(profile.get("location", "")).strip()
    social_category = str(profile.get("social_category", "")).strip()

    for _, scheme in schemes_df.iterrows():
        scheme_id = str(scheme["scheme_id"]).strip()
        scheme_name = str(scheme["scheme_name"]).strip()
        min_age = int(scheme["min_age"])
        funding_min = float(scheme["funding_min"])
        funding_max = float(scheme["funding_max"])
        support_type = str(scheme["support_type"]).strip()
        official_url = str(scheme["official_url"]).strip()

        # 1. Non-funding check: Minimum Age
        if age < min_age:
            continue

        # 2. Non-funding check: Target Sector
        target_sectors = [s.strip() for s in str(scheme["target_sector"]).split(";")]
        if sector not in target_sectors:
            continue

        # 3. Non-funding check: Business Stage
        # Rule: 'Planning' must NOT match New or Existing.
        # For PMFME: 'New (ODOP)' matches 'New'.
        scheme_stages = [s.strip() for s in str(scheme["business_stage"]).split(";")]
        if business_stage == "Planning":
            continue
        elif business_stage == "New":
            has_new = "New" in scheme_stages
            has_new_odop = any(s.startswith("New (ODOP)") for s in scheme_stages)
            if not (has_new or has_new_odop):
                continue
        elif business_stage == "Existing":
            if "Existing" not in scheme_stages:
                continue
        else:
            continue

        # 4. Non-funding check: Social Category
        allowed_categories = [c.strip() for c in str(scheme["social_category"]).split(";")]
        if "All" not in allowed_categories and social_category not in allowed_categories:
            continue

        # 5. Non-funding check: Location Condition
        allowed_locations = [loc.strip() for loc in str(scheme["location_condition"]).split(";")]
        if location not in allowed_locations:
            continue

        # Non-funding rules all PASSED!
        # Now check funding limits
        effective_funding_max = funding_max
        if scheme_id == "SCH01" and sector in ["Service", "Trading"]:
            effective_funding_max = 2000000.0  # ₹20 Lakhs cap for Service / Trading in PMEGP

        # If it passes funding, it is an exact match, not a near-miss
        if funding_min <= funding_req <= effective_funding_max:
            continue

        # Build funding gap explanation note
        if funding_req > effective_funding_max:
            excess = funding_req - effective_funding_max
            if scheme_id == "SCH01" and sector in ["Service", "Trading"]:
                gap_note = (
                    f"Requested funding (₹{funding_req:,.0f}) exceeds the PMEGP Service/Trading "
                    f"maximum ceiling of ₹{effective_funding_max:,.0f} by ₹{excess:,.0f}."
                )
            else:
                gap_note = (
                    f"Requested funding (₹{funding_req:,.0f}) exceeds the maximum scheme ceiling "
                    f"of ₹{effective_funding_max:,.0f} by ₹{excess:,.0f}."
                )
        else:
            deficit = funding_min - funding_req
            gap_note = (
                f"Requested funding (₹{funding_req:,.0f}) is below the minimum scheme threshold "
                f"of ₹{funding_min:,.0f} by ₹{deficit:,.0f}."
            )

        near_misses.append({
            "scheme_id": scheme_id,
            "scheme_name": scheme_name,
            "support_type": support_type,
            "funding_min": funding_min,
            "effective_funding_max": effective_funding_max,
            "official_url": official_url,
            "funding_gap_note": gap_note,
        })

    return near_misses


def generate_cluster_scheme_mapping(
    applicants_path: str = os.path.join("data", "applicants.csv"),
    schemes_path: str = os.path.join("data", "schemes.csv"),
    preprocessor_path: str = os.path.join("models", "preprocessor.joblib"),
    kmeans_path: str = os.path.join("models", "kmeans_model.joblib"),
) -> pd.DataFrame:
    """
    Assign applicants to clusters and generate a comprehensive mapping of
    clusters to eligible schemes.

    Prints a formatted summary table displaying counts and percentages.
    """
    # Load data
    applicants_df = pd.read_csv(applicants_path)
    schemes_df = pd.read_csv(schemes_path)

    # Load trained ML models
    preprocessor = joblib.load(preprocessor_path)
    kmeans = joblib.load(kmeans_path)

    # Transform features and predict clusters
    numerical_features = ["age", "annual_income", "funding_requirement"]
    categorical_features = ["sector", "business_stage", "location", "social_category", "purpose"]
    X = applicants_df[numerical_features + categorical_features]
    X_processed = preprocessor.transform(X)
    applicants_df["cluster"] = kmeans.predict(X_processed)

    # Get scheme IDs and short labels
    scheme_ids = list(schemes_df["scheme_id"])
    scheme_labels = {
        "SCH01": "PMEGP",
        "SCH02": "PMFME",
        "SCH03": "Mudra",
        "SCH04": "StandUp",
    }

    # Track match statistics per cluster
    stats: Dict[int, Dict[str, Any]] = {}
    for c in sorted(CLUSTER_NAMES.keys()):
        stats[c] = {
            "cluster_id": c,
            "persona": CLUSTER_NAMES[c],
            "total_applicants": 0,
            "matched_any": 0,
            **{sid: 0 for sid in scheme_ids},
        }

    for _, row in applicants_df.iterrows():
        c = int(row["cluster"])
        stats[c]["total_applicants"] += 1
        matches = match_applicant(row.to_dict(), schemes_df)
        matched_ids = {m["scheme_id"] for m in matches}

        if matched_ids:
            stats[c]["matched_any"] += 1

        for sid in matched_ids:
            stats[c][sid] += 1

    # Print Table
    total_all = len(applicants_df)
    print("=" * 115)
    print(f"{'SCHEMEMATCH: CLUSTER-TO-SCHEME MATCHING TABLE':^115}")
    print("=" * 115)

    col_headers = [f"{sid} ({scheme_labels.get(sid, sid)})" for sid in scheme_ids]
    header = (
        f"{'Cluster':<9} | {'Persona':<28} | {'Applicants':<11} | "
        f"{'At Least 1':<13} | "
        + " | ".join([f"{ch:<14}" for ch in col_headers])
    )
    print(header)
    print("-" * 115)

    summary_rows = []
    for c in sorted(CLUSTER_NAMES.keys()):
        d = stats[c]
        tot = d["total_applicants"]
        pct_any = (d["matched_any"] / tot * 100) if tot > 0 else 0

        scheme_cols = []
        for sid in scheme_ids:
            cnt = d[sid]
            pct = (cnt / tot * 100) if tot > 0 else 0
            scheme_cols.append(f"{cnt} ({pct:4.1f}%)")

        row_str = (
            f"Cluster {c:<1} | {d['persona']:<28} | {tot:>4d} ({tot/total_all*100:4.1f}%) | "
            f"{d['matched_any']:>3d} ({pct_any:4.1f}%)  | "
            + " | ".join([f"{col:<14}" for col in scheme_cols])
        )
        print(row_str)

        row_dict = {
            "cluster_id": c,
            "persona": d["persona"],
            "total_applicants": tot,
            "matched_any_count": d["matched_any"],
            "matched_any_pct": pct_any,
        }
        for sid in scheme_ids:
            row_dict[f"{sid}_count"] = d[sid]
            row_dict[f"{sid}_pct"] = (d[sid] / tot * 100) if tot > 0 else 0
        summary_rows.append(row_dict)

    print("-" * 115)
    overall_matched = sum(d["matched_any"] for d in stats.values())
    print(
        f"OVERALL SUMMARY: {overall_matched} of {total_all} applicants ({overall_matched/total_all*100:.1f}%) "
        f"match at least one verified government scheme."
    )
    unmatched_count = total_all - overall_matched
    print(
        f"Unmatched applicants: {unmatched_count} ({unmatched_count/total_all*100:.1f}%) — "
        f"primarily underage applicants (< 18 yrs) and 'Planning' stage ventures."
    )
    print("=" * 115)

    return pd.DataFrame(summary_rows)


if __name__ == "__main__":
    generate_cluster_scheme_mapping()
