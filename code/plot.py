# Part 3: Exploratory Data Analysis
# CS210 - NYC Motor Vehicle Collisions

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# BASE points to the CS210-Final-Project folder regardless of where you run the script from
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# load the cleaned data
df = pd.read_csv(BASE + "\\data\\collisions_clean.csv")
print(f"loaded {df.shape[0]:,} rows, making graphs now")

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

# PLOT 1: crashes by hour - when during the day do most crashes happen?
hourly = df.groupby("hour").size()

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(hourly.index, hourly.values, color="steelblue", alpha=0.8)
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Number of Crashes")
ax.set_title("Crashes by Hour of Day")
ax.set_xticks(range(0, 24))
plt.tight_layout()
plt.savefig(BASE + "\\plots\\plot1_crashes_by_hour.png")
plt.show()
print("saved plot1")

# PLOT 2: average severity by hour - more crashes doesnt always = worse crashes, this checks actual severity
hourly_sev = df.groupby("hour")["severity_score"].mean()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(hourly_sev.index, hourly_sev.values, color="coral", linewidth=2, marker="o", markersize=4)
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Avg Severity Score")
ax.set_title("Average Crash Severity by Hour of Day")
ax.set_xticks(range(0, 24))
plt.tight_layout()
plt.savefig(BASE + "\\plots\\plot2_severity_by_hour.png")
plt.show()
print("saved plot2")

# PLOT 3: rush hour vs non rush hour severity
rush_sev = df.groupby("is_rush_hour")["severity_score"].mean()
rush_sev.index = ["Not Rush Hour", "Rush Hour"]

fig, ax = plt.subplots(figsize=(6, 5))
rush_sev.plot(kind="bar", ax=ax, color=["steelblue", "coral"], edgecolor="white")
ax.set_xlabel("")
ax.set_ylabel("Avg Severity Score")
ax.set_title("Rush Hour vs Non Rush Hour — Average Severity")
ax.set_xticklabels(rush_sev.index, rotation=0)
plt.tight_layout()
plt.savefig(BASE + "\\plots\\plot3_rush_hour_severity.png")
plt.show()
print("saved plot3")

# PLOT 4: weekday vs weekend severity
weekend_sev = df.groupby("is_weekend")["severity_score"].mean()
weekend_sev.index = ["Weekday", "Weekend"]

fig, ax = plt.subplots(figsize=(6, 5))
weekend_sev.plot(kind="bar", ax=ax, color=["teal", "mediumpurple"], edgecolor="white")
ax.set_xlabel("")
ax.set_ylabel("Avg Severity Score")
ax.set_title("Weekday vs Weekend — Average Severity")
ax.set_xticklabels(weekend_sev.index, rotation=0)
plt.tight_layout()
plt.savefig(BASE + "\\plots\\plot4_weekend_severity.png")
plt.show()
print("saved plot4")

# PLOT 5: night driving severity
night_sev = df.groupby("is_night")["severity_score"].mean()
night_sev.index = ["Normal Hours", "Late Night (12am-5am)"]

fig, ax = plt.subplots(figsize=(6, 5))
night_sev.plot(kind="bar", ax=ax, color=["cadetblue", "darkslateblue"], edgecolor="white")
ax.set_xlabel("")
ax.set_ylabel("Avg Severity Score")
ax.set_title("Late Night Driving vs Normal Hours — Average Severity")
ax.set_xticklabels(night_sev.index, rotation=0)
plt.tight_layout()
plt.savefig(BASE + "\\plots\\plot5_night_severity.png")
plt.show()
print("saved plot5")

# PLOT 6: crashes by borough - which borough has the most crashes total?
borough_counts = (
    df[df["borough"] != "Unknown"]
    .groupby("borough").size()
    .sort_values(ascending=False)
)

fig, ax = plt.subplots(figsize=(8, 5))
borough_counts.plot(kind="bar", ax=ax, color="teal", edgecolor="white")
ax.set_xlabel("Borough")
ax.set_ylabel("Number of Crashes")
ax.set_title("Total Crashes by Borough")
ax.set_xticklabels(borough_counts.index, rotation=15, ha="right")
plt.tight_layout()
plt.savefig(BASE + "\\plots\\plot6_crashes_by_borough.png")
plt.show()
print("saved plot6")

# PLOT 7: crashes by factor group - which group of contributing factors causes the most crashes?
factor_counts = df[df["factor_group"] != "Other"].groupby("factor_group").size().sort_values()

fig, ax = plt.subplots(figsize=(9, 5))
factor_counts.plot(kind="barh", ax=ax, color="slateblue", edgecolor="white")
ax.set_xlabel("Number of Crashes")
ax.set_title("Crashes by Contributing Factor Group")
plt.tight_layout()
plt.savefig(BASE + "\\plots\\plot7_factor_groups.png")
plt.show()
print("saved plot7")

# PLOT 8: factor group vs severity - which factor group leads to the worst crashes? distraction vs alcohol etc
factor_sev = (
    df[df["factor_group"] != "Other"]
    .groupby("factor_group")["severity_score"]
    .mean()
    .sort_values()
)

fig, ax = plt.subplots(figsize=(9, 5))
factor_sev.plot(kind="barh", ax=ax, color="tomato", edgecolor="white")
ax.set_xlabel("Avg Severity Score")
ax.set_title("Average Crash Severity by Contributing Factor Group")
plt.tight_layout()
plt.savefig(BASE + "\\plots\\plot8_factor_severity.png")
plt.show()
print("saved plot8")

# PLOT 9: num vehicles vs severity - more vehicles involved = worse crash?
vehicle_sev = df.groupby("num_vehicles")["severity_score"].mean()

fig, ax = plt.subplots(figsize=(7, 5))
vehicle_sev.plot(kind="bar", ax=ax, color="darkorange", edgecolor="white")
ax.set_xlabel("Number of Vehicles Involved")
ax.set_ylabel("Avg Severity Score")
ax.set_title("Number of Vehicles Involved vs Average Severity")
ax.set_xticklabels(vehicle_sev.index, rotation=0)
plt.tight_layout()
plt.savefig(BASE + "\\plots\\plot9_num_vehicles_severity.png")
plt.show()
print("saved plot9")

# PLOT 10: cyclist injuries by hour - when are cyclists most at risk during the day?
if "number_of_cyclist_injured" in df.columns:
    cyclist = df.groupby("hour")["number_of_cyclist_injured"].sum()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(cyclist.index, cyclist.values, color="darkorange", alpha=0.8)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Total Cyclist Injuries")
    ax.set_title("Cyclist Injuries by Hour of Day")
    ax.set_xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig(BASE + "\\plots\\plot10_cyclist_injuries.png")
    plt.show()
    print("saved plot10")

print("\nall plots done!")