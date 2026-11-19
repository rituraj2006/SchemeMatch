# 🎯 SchemeMatch: Beneficiary Clustering & Government Scheme Matching Platform

> **Course:** INT396 — Unsupervised Learning  
> **Problem Reference:** Smart India Hackathon (SIH #18 / SIH26092)  
> **Core Focus:** Beneficiary Archetype Segmentation & Intelligent Scheme Recommendation for Marginalized Entrepreneurs

---

## 📌 1. Project Overview & Problem Statement

### The Problem
Micro, Small, and Medium Enterprises (MSMEs) and aspiring entrepreneurs across India face severe information asymmetry when attempting to access central and state government credit and subsidy programs. Schemes like **PMEGP**, **PMFME**, **PMMY (Mudra)**, and **Stand-Up India** carry nuanced, multidimensional eligibility criteria—spanning age thresholds, sector restrictions, operational stages, demographic reservations, and sector-specific funding caps. Traditional portal search engines rely on simple keyword lookups that frequently miss applicant-venture compatibility, causing high application rejection rates and under-utilization of allocated public funds.

### The SchemeMatch Solution
**SchemeMatch** provides a dual-engine decision-support system designed to empower marginalized and underserved entrepreneurs—specifically **Women, Scheduled Castes (SC), Scheduled Tribes (ST), Other Backward Classes (OBC), Minorities, and Rural Micro-Enterprises**. 

The platform couples:
1. **Unsupervised Machine Learning**: Clusters applicant multidimensional attributes to discover latent demographic and economic personas.
2. **Deterministic Rule Engine with Near-Miss Feedback**: Rigorously evaluates statutory government policies and provides prescriptive guidance when funding requests fall outside scheme limits.
3. **Interactive Streamlit Web Interface**: An intuitive portal for entrepreneurs, policy analysts, and field officers.

---

## 🏗️ 2. System Architecture

SchemeMatch employs a decoupled, modular two-stage pipeline:

```
[ Applicant Profile ]
         │
         ├───► [ 1. Unsupervised Clustering Subsystem ]
         │         ├─ ColumnTransformer (StandardScaler + OneHotEncoder)
         │         ├─ Trained K-Means Pipeline (K=4)
         │         └─► Output: Assigned Persona Archetype & Diagnostic Insights
         │
         └───► [ 2. Deterministic Rule Matching Engine (matcher.py) ]
                   ├─ Hard-Constraint Policy Filtering (Age, Sector, Stage, Category, Location)
                   ├─ Dynamic Financial Bounds (Sector-specific caps, e.g., PMEGP ₹20L Service cap)
                   │
                   ├──► [ Exact Matches ] ──► Actionable Verification Checklist + Official Portal Links
                   │
                   └──► [ 0 Matches? ] ──► [ Near-Miss Funding Detector ]
                                                ├─ Checks if non-funding criteria passed
                                                └─► Prescriptive Guidance on Funding Gaps
```

### Component Details

#### A. K-Means Unsupervised Clustering Pipeline
- **Preprocessing**: Implemented via scikit-learn `ColumnTransformer`:
  - `StandardScaler()` applied to continuous numerical features (`age`, `annual_income`, `funding_requirement`).
  - `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` applied to categorical features (`sector`, `business_stage`, `location`, `social_category`, `purpose`).
  - Transforms raw 8-dimensional profiles into an orthogonal 24-dimensional feature matrix.
- **Model Training**: `KMeans(n_clusters=4, random_state=42, n_init=10)` clusters applicants into 4 distinct economic segments.
- **Dimensionality Reduction**: `PCA(n_components=2, random_state=42)` projects the 24D space to 2D for cluster separation and centroid visualization.

#### B. Deterministic Rule-Based Matching Engine
While clustering identifies *who the applicant is*, scheme eligibility requires strict statutory compliance. The matching engine evaluates:
1. **Minimum Age**: Enforces $Age \ge min\_age$ (legal minimum of 18 years).
2. **Target Sector**: Verifies alignment with semicolon-separated eligible sectors.
3. **Business Stage**: Enforces operational stage compatibility:
   - `New`: Greenfield setups (supports `New` and `New (ODOP)` with verification flags).
   - `Existing`: Operational units eligible for expansion or modernization.
   - `Planning`: Strictly excluded from loan disbursements until formal establishment.
4. **Social Category**: Enforces demographic reservations (e.g., Stand-Up India's `SC;ST;Women` mandate).
5. **Location Condition**: Verifies `Rural` vs `Urban` eligibility and flags higher subsidy slabs (25%–35% under PMEGP for rural units).
6. **Dynamic Funding Limits**: Checks $funding\_min \le funding\_requirement \le effective\_funding\_max$, including PMEGP's statutory **₹20,00,000 cap** for `Service` and `Trading` units (vs. ₹50,00,000 for `Manufacturing`).

#### C. Prescriptive Near-Miss Funding Detector
When an applicant passes all 5 non-funding policy gates but fails solely due to funding requirement:
- Automatically detects the gap (excess or deficit).
- Quantifies the exact variance (e.g., *"Requested funding of ₹40L exceeds Mudra ceiling of ₹20L by ₹20L"*).
- Recommends the scheme with its supported range so the applicant can calibrate their project size.

---

## 📊 3. Dataset Summary

The system is evaluated on two structured datasets located in `data/`:

### 1. `data/schemes.csv` (Verified Government Schemes)
Contains 4 major central credit and subsidy schemes with 14 verified fields:

| Scheme ID | Scheme Name | Target Sector | Stage | Funding Range | Social Category | Core Support Type |
|---|---|---|---|---|---|---|
| **SCH01** | PMEGP (Prime Minister's Employment Generation Programme) | Manufacturing, Service, Trading | New | ₹0 – ₹50,00,000 *(₹20L for Service/Trading)* | All (Higher subsidy for SC/ST/OBC/Women/Minorities) | Margin Money Subsidy (15%–35%) + Bank Loan |
| **SCH02** | PMFME (PM Formalisation of Micro Food Processing Enterprises) | Food Processing | Existing, New (ODOP) | ₹0 – ₹10,00,000 | All | Credit-Linked Capital Subsidy (35% max 10L) |
| **SCH03** | PMMY (Pradhan Mantri Mudra Yojana) | Mfg, Service, Trading, Agri-Allied | New, Existing | ₹0 – ₹20,00,000 *(Shishu, Kishore, Tarun, Tarun Plus)* | All | Collateral-Free Micro/Small Business Loan |
| **SCH04** | Stand-Up India | Mfg, Service, Trading, Agri-Allied | New | ₹10,00,000 – ₹1,00,00,000 | SC, ST, Women | Composite Bank Loan (Term + Working Capital) |

### 2. `data/applicants.csv` (Synthetic Entrepreneur Profiles)
- **400 realistic applicant records** (`SYN_001` through `SYN_400`) generated using NumPy and Pandas (`np.random.seed(42)`).
- Realistic multivariate correlation reflecting Indian MSME economic segments:
  - Young rural micro-startups (modest income, agricultural focus, low capital).
  - Urban service and trading ventures (moderate income, working capital needs).
  - Established manufacturing units (higher income, machinery expansion).
  - Women and minority enterprise ventures.
- Includes ~8% realistic edge cases (12 underage profiles at age 16–17; 32 ideation profiles in "Planning" stage) to test boundary rejection.

---

## 📈 4. Key Results & Analytical Findings

### A. Optimal $K$ Evaluation (Elbow Method & Silhouette Analysis)
To select the optimal number of clusters for `X_processed`, $K \in [2, 10]$ was evaluated:

| Number of Clusters ($K$) | Inertia (WCSS) | Silhouette Score | Business & Statistical Justification |
|:---:|:---:|:---:|---|
| **2** | 1945.1953 | 0.3051 | Overly coarse (splits only along extreme income/funding). |
| **3** | 1664.2649 | 0.1691 | Merges distinct rural micro-units with urban services. |
| **4** | **1573.3975** | **0.1508** | **Optimal selection**: Distinct inflection on elbow curve; produces 4 highly interpretable, actionable entrepreneurial personas. |
| **5** | 1500.6726 | 0.1132 | Diminishing silhouette score; fragments natural clusters. |
| **6** | 1436.7406 | 0.1074 | Sub-clusters become noisy. |

### B. Identified Persona Clusters ($K=4$)

| Cluster ID | Persona Name | Count (% Total) | Mean Age | Mean Income | Mean Funding Need | Dominant Sector | Dominant Stage | Primary Purpose |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **0** | **Urban Service Professionals** | 119 (29.8%) | 30.8 yrs | ₹3,26,723 | ₹4,73,782 | Service | Existing | Working Capital |
| **1** | **Young Rural Micro-Startups** | 143 (35.8%) | 23.3 yrs | ₹1,34,615 | ₹2,14,545 | Agri-Allied | New | Unit Setup |
| **2** | **Established Industrialists** | 64 (16.0%) | 40.3 yrs | ₹6,52,812 | ₹27,60,312 | Manufacturing | Existing | Machinery & Equipment |
| **3** | **Rural Growth Enterprises** | 74 (18.5%) | 37.5 yrs | ₹2,94,595 | ₹4,79,459 | Manufacturing | New | Business Expansion |

### C. PCA Dimensionality Reduction
- **PC1 Explained Variance**: **35.73%**
- **PC2 Explained Variance**: **8.73%**
- **Total Explained Variance**: **44.46%**
- 2D scatter plots show distinct spatial segregation across clusters with clear separation of cluster centroids.

### D. Cluster-to-Scheme Matching Matrix
Cross-tabulation across the 400 applicants confirms that unsupervised segmentation correlates strongly with institutional eligibility:

| Cluster | Persona | Total Applicants | Qualified ($\ge 1$ Scheme) | SCH01 (PMEGP) | SCH02 (PMFME) | SCH03 (Mudra) | SCH04 (Stand-Up) |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **0** | Urban Service Professionals | 119 | **109 (91.6%)** | 46 (38.7%) | 7 (5.9%) | 102 (85.7%) | 0 (0.0%) |
| **1** | Young Rural Micro-Startups | 143 | **124 (86.7%)** | 29 (20.3%) | 28 (19.6%) | 96 (67.1%) | 0 (0.0%) |
| **2** | Established Industrialists | 64 | **18 (28.1%)** | 4 (6.2%) | 1 (1.6%) | 13 (20.3%) | 3 (4.7%) |
| **3** | Rural Growth Enterprises | 74 | **67 (90.5%)** | 25 (33.8%) | 14 (18.9%) | 53 (71.6%) | 0 (0.0%) |

- **Overall Population Match Rate**: **318 of 400 applicants (79.5%)** qualify for at least one government scheme.
- **Unmatched Population (20.5%)**: Exactly accounts for intentional boundary cases (underage $<18$, ideation "Planning" stage, and mega-projects exceeding MSME ceilings).

### E. Validation Profile Test Suite (`test_profiles.py`)
A comprehensive test suite of 10 diverse applicant profiles achieved a **100% test pass rate (10/10)**:
- Standard applicants across Agri-Allied, Food Processing, Service, Trading, and Manufacturing qualified for their targeted programs.
- **Stand-Up India** successfully matched both SC Greenfield Manufacturing (₹25L) and Women Greenfield Service (₹12L).
- Edge cases correctly triggered policy diagnostics:
  - Age 17 blocked by statutory 18+ policy.
  - Planning stage blocked by active operational stage rule.
  - ₹40 Crore mega-project triggered near-miss detection for Mudra with clear ceiling feedback.

---

## 📁 5. Project Directory Structure

```text
SchemeMatch/
│
├── main.py                               # Interactive Streamlit Web Application (Phase 8)
├── matcher.py                            # Rule-Based Matching Engine & Near-Miss Detector (Phase 7)
├── test_profiles.py                      # Validation Test Suite for 10 Example Profiles (Phase 9)
├── requirements.txt                      # Python dependencies
├── .gitignore                            # GitHub security and clean repository rules
├── README.md                             # Academic and technical documentation
│
├── data/
│   ├── schemes.csv                       # 4 verified government credit & subsidy schemes
│   └── applicants.csv                    # 400 synthetic applicant profiles (Phase 2)
│
├── models/                               # Serialized ML model artifacts (ignored by Git)
│   ├── preprocessor.joblib               # Fitted ColumnTransformer (StandardScaler + OneHotEncoder)
│   └── kmeans_model.joblib               # Trained K-Means clustering model (K=4)
│
├── notebooks/                            # Interactive Jupyter research notebooks
│   ├── 01_eda.ipynb                      # Exploratory Data Analysis & Feature Distributions (Phase 3)
│   └── 02_clustering.ipynb               # Preprocessing, Optimal K, Final Model, PCA (Phase 5 & 6)
│
└── visualizations/                       # Saved high-resolution figures (150 DPI)
    ├── eda_numerical_distributions.png   # Histograms & KDE for age, income, funding
    ├── eda_categorical_counts.png        # Bar charts for sector, stage, location, category
    ├── eda_income_vs_funding.png         # Income vs Funding scatter plot by sector
    ├── kmeans_elbow_silhouette.png       # Elbow Method and Silhouette score curves
    └── kmeans_clusters_pca.png           # 2D PCA projection of K-Means clusters & centroids
```

---

## 🚀 6. Installation & Usage Instructions

### Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/rituraj2006/SchemeMatch.git
cd SchemeMatch
```

### 2. Create and Activate Virtual Environment
```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Rule-Based Matching Engine (CLI)
Executes cluster prediction and outputs the 400-applicant Cluster-to-Scheme mapping table:
```bash
python matcher.py
```

### 5. Run the 10-Profile Test Validation Suite
Tests 7 standard matching profiles and 3 boundary edge cases:
```bash
python test_profiles.py
```

### 6. Launch the Interactive Web Application
Starts the Streamlit interactive dashboard:
```bash
streamlit run main.py
```
Open your browser at `http://localhost:8501` to use the interactive matching interface.

---

## ⚠️ 7. Limitations & Statutory Disclaimer

1. **Synthetic Training Data**: Applicant profiles in `data/applicants.csv` are synthetically generated for academic demonstration under course INT396 to model real-world distributions without exposing Personally Identifiable Information (PII).
2. **Advisory Tool**: SchemeMatch is an independent academic research and decision-support prototype. Recommendations generated by this platform represent **potential eligibility matches** based strictly on published guidelines. They do **not** constitute legal credit approval, administrative sanction, or guarantee of financial disbursement by any government ministry, nodal agency, or participating commercial bank.
3. **Institutional Formalities**: Final disbursement is contingent upon formal project appraisal, Detailed Project Report (DPR) clearance, Entrepreneurship Development Programme (EDP) training completion, and statutory bank KYC compliance.

---

## 👨‍💻 Author & Acknowledgments

- **Course:** INT396 (Unsupervised Learning)
- **Problem Statement:** SIH26092 / SIH #18 (Beneficiary Clustering for Scheme Matching)
- **Built with:** Python, scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, and Streamlit.
