
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
