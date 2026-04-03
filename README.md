============================================================
#   AITGS+ — AI-Driven Autonomous Traffic Governance System
# ============================================================
#
#   What this project does:
#   ------------------------
#   Most cloud systems like AWS, NGINX, and Envoy react to
#   overload AFTER it has already happened. AITGS+ is different.
#   It uses Machine Learning to PREDICT when the system is about
#   to get overloaded — and takes action BEFORE it breaks.
#
#   This script does three things:
#   1. Trains a classifier to predict SLO violations (will this
#      request cause the system to breach its performance target?)
#   2. Trains a regressor to estimate system stress in the future
#   3. Compares both models against existing approaches from
#      published research papers
#
#   Dataset: Cloud Workload Dataset (Kaggle)
#   Authors: AITGS+ Research Team
# ============================================================


# ── Step 0: Import all the libraries we need ────────────────
#
#   pandas    → for loading and working with our dataset
#   numpy     → for math operations
#   matplotlib / seaborn → for drawing the graphs
#   sklearn   → the machine learning library (models, metrics)
#   warnings  → just to keep the output clean

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")   # hide unnecessary warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    mean_absolute_error,
    r2_score,
    accuracy_score,
)


# ============================================================
#   SECTION 1 — Load the Dataset
# ============================================================
#
#   We load the cloud workload CSV file. Each row represents
#   one job/task running on a cloud server. Columns include
#   CPU usage, memory, waiting time, error rate, and more.

print("Loading dataset...")
df = pd.read_csv("/content/cloud_workload_dataset.csv")

print(f"  ✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"  ✓ Columns: {list(df.columns)}\n")


# ============================================================
#   SECTION 2 — Feature Engineering (Creating New Columns)
# ============================================================
#
#   The raw dataset doesn't have a column that directly tells
#   us "how stressed is the system right now?" — so we create
#   one ourselves. This is called Feature Engineering.


# ── 2a. Aggregate Stress Score ───────────────────────────────
#
#   This is AITGS+'s core idea. We combine four metrics into
#   one single "stress score" that represents how hard the
#   system is working overall:
#
#     - CPU Utilization     → 35% weight (most important)
#     - Memory Consumption  → 25% weight
#     - Task Waiting Time   → 25% weight (long queues = stress)
#     - Error Rate          → 15% weight
#
#   Each metric is normalised to 0–100 so they're comparable.

df["Stress_Score"] = (
    df["CPU_Utilization (%)"] * 0.35
    + (df["Memory_Consumption (MB)"] / df["Memory_Consumption (MB)"].max()) * 100 * 0.25
    + (df["Task_Waiting_Time (ms)"] / df["Task_Waiting_Time (ms)"].max()) * 100 * 0.25
    + (df["Error_Rate (%)"] / df["Error_Rate (%)"].max()) * 100 * 0.15
)

print("  ✓ Aggregate Stress Score created")


# ── 2b. SLO Violation Label (our classification target) ──────
#
#   SLO = Service Level Objective. It's the internal performance
#   target a cloud system must meet (e.g. respond in under 250ms).
#
#   We label a task as an SLO Violation if its stress score
#   goes above 70 — meaning the system is under severe load
#   and is likely to breach performance guarantees.
#
#   0 = No Violation (system is fine)
#   1 = SLO Violation (system is in danger)

df["SLO_Violation"] = (df["Stress_Score"] > 70).astype(int)

violation_count = df["SLO_Violation"].sum()
print(f"  ✓ SLO Violation labels created")
print(f"    → {violation_count} violations ({violation_count/len(df)*100:.1f}% of dataset)\n")


# ── 2c. Marginal Impact Score ────────────────────────────────
#
#   One of AITGS+'s unique ideas: instead of asking
#   "is this request heavy?", we ask
#   "how much EXTRA stress does one more request add?"
#
#   This solves the "many small requests" problem — where
#   hundreds of cheap API calls collectively overload a server.
#   Existing systems like AWS ALB miss this entirely.

df["Marginal_Impact"] = (
    (df["Task_Execution_Time (ms)"] / df["Task_Execution_Time (ms)"].max()) * 0.5
    + (df["Task_Waiting_Time (ms)"] / df["Task_Waiting_Time (ms)"].max()) * 0.5
) * 100

print("  ✓ Marginal Impact Score created")


# ── 2d. Intent Classification ────────────────────────────────
#
#   AITGS+ classifies every API request by its business intent:
#     - Critical   → must always be served (e.g. Payment API)
#     - Important  → served unless system is in crisis
#     - Deferrable → can be delayed or dropped under high load
#
#   We map the dataset's Job_Priority column to these classes.

intent_map = {
    "High":   "Critical",
    "Medium": "Important",
    "Low":    "Deferrable",
}
df["Intent_Class"] = df["Job_Priority"].map(intent_map)

print("  ✓ Intent Classes mapped (Critical / Important / Deferrable)")
print(f"    → Distribution:\n{df['Intent_Class'].value_counts().to_string()}\n")


# ============================================================
#   SECTION 3 — Encode Categorical Columns
# ============================================================
#
#   Machine Learning models only understand numbers.
#   Columns like "Job_Priority = High" need to be converted
#   to numbers like 0, 1, 2. This is called Label Encoding.

le = LabelEncoder()

df["Job_Priority_enc"]        = le.fit_transform(df["Job_Priority"])
df["Scheduler_Type_enc"]      = le.fit_transform(df["Scheduler_Type"])
df["Resource_Allocation_enc"] = le.fit_transform(df["Resource_Allocation_Type"])

print("  ✓ Categorical columns encoded to numbers")


# ============================================================
#   SECTION 4 — Prepare Features and Targets
# ============================================================
#
#   Features (X) = the columns the model looks at to make predictions
#   Target (y)   = the column the model is trying to predict
#
#   We have two prediction tasks:
#     y_cls → will this cause an SLO violation? (yes/no)
#     y_reg → what will the stress score be? (a number)

FEATURES = [
    "CPU_Utilization (%)",
    "Memory_Consumption (MB)",
    "Task_Execution_Time (ms)",
    "System_Throughput (tasks/sec)",
    "Task_Waiting_Time (ms)",
    "Number_of_Active_Users",
    "Network_Bandwidth_Utilization (Mbps)",
    "Error_Rate (%)",
    "Marginal_Impact",           # our custom feature
    "Job_Priority_enc",
    "Scheduler_Type_enc",
    "Resource_Allocation_enc",
]

X     = df[FEATURES]
y_cls = df["SLO_Violation"]     # for classification
y_reg = df["Stress_Score"]      # for regression

