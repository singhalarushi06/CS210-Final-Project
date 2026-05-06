# Part 3: Exploratory Data Analysis
# CS210 - NYC Motor Vehicle Collisions

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

df = pd.read_csv(BASE + "\\data\\collisions_clean.csv")
print(f"loaded {df.shape[0]:,} rows, making graphs now")

sns.set_theme(style="whitegrid", palette="muted")

# PLOT 1: when during the day do most crashes happen?
hourly = df.groupby("hour").size()

plt.figure(figsize=(10, 5))
plt.bar(hourly.index, hourly.values, color="steelblue", alpha=0.8)
plt.title("Crashes by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Crashes")
plt.xticks(range(0, 24))
plt.savefig(BASE + "\\plots\\plot1_crashes_by_hour.png")
plt.show()
print("saved plot1")

# PLOT 2: does a higher crash count = more severe crashes?
hourly_sev = df.groupby("hour")["severity_score"].mean()

plt.figure(figsize=(10, 5))
plt.plot(hourly_sev.index, hourly_sev.values, color="coral", linewidth=2, marker="o", markersize=4)
plt.title("Average Crash Severity by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Avg Severity Score")
plt.xticks(range(0, 24))
plt.savefig(BASE + "\\plots\\plot2_severity_by_hour.png")
plt.show()
print("saved plot2")

# PLOT 3: are rush hour crashes actually more severe or just more frequent?
rush_sev = df.groupby("is_rush_hour")["severity_score"].mean()
rush_sev.index = ["Not Rush Hour", "Rush Hour"]

plt.figure(figsize=(6, 5))
plt.bar(rush_sev.index, rush_sev.values, color=["steelblue", "coral"])
plt.title("Rush Hour vs Non Rush Hour — Average Severity")
plt.ylabel("Avg Severity Score")
plt.savefig(BASE + "\\plots\\plot3_rush_hour_severity.png")
plt.show()
print("saved plot3")

# PLOT 4: are weekend crashes worse than weekday crashes?
weekend_sev = df.groupby("is_weekend")["severity_score"].mean()
weekend_sev.index = ["Weekday", "Weekend"]

plt.figure(figsize=(6, 5))
plt.bar(weekend_sev.index, weekend_sev.values, color=["teal", "mediumpurple"])
plt.title("Weekday vs Weekend — Average Severity")
plt.ylabel("Avg Severity Score")
plt.savefig(BASE + "\\plots\\plot4_weekend_severity.png")
plt.show()
print("saved plot4")

# PLOT 5: is late night driving more dangerous than normal hours?
night_sev = df.groupby("is_night")["severity_score"].mean()
night_sev.index = ["Normal Hours", "Late Night (12am-5am)"]

plt.figure(figsize=(6, 5))
plt.bar(night_sev.index, night_sev.values, color=["cadetblue", "darkslateblue"])
plt.title("Late Night Driving vs Normal Hours — Average Severity")
plt.ylabel("Avg Severity Score")
plt.savefig(BASE + "\\plots\\plot5_night_severity.png")
plt.show()
print("saved plot5")

# PLOT 6: which borough has the most crashes?
borough_counts = (
    df[df["borough"] != "Unknown"]
    .groupby("borough").size()
    .sort_values(ascending=False)
)

plt.figure(figsize=(8, 5))
plt.bar(borough_counts.index, borough_counts.values, color="teal")
plt.title("Total Crashes by Borough")
plt.xlabel("Borough")
plt.ylabel("Number of Crashes")
plt.savefig(BASE + "\\plots\\plot6_crashes_by_borough.png")
plt.show()
print("saved plot6")

# PLOT 7: which contributing factor causes the most crashes?
factor_counts = df[df["factor_group"] != "Other"].groupby("factor_group").size().sort_values()

plt.figure(figsize=(9, 5))
plt.barh(factor_counts.index, factor_counts.values, color="slateblue")
plt.title("Crashes by Contributing Factor Group")
plt.xlabel("Number of Crashes")
plt.savefig(BASE + "\\plots\\plot7_factor_groups.png")
plt.show()
print("saved plot7")

# PLOT 8: which factor group leads to the worst crashes?
factor_sev = (
    df[df["factor_group"] != "Other"]
    .groupby("factor_group")["severity_score"]
    .mean()
    .sort_values()
)

plt.figure(figsize=(9, 5))
plt.barh(factor_sev.index, factor_sev.values, color="tomato")
plt.title("Average Crash Severity by Contributing Factor Group")
plt.xlabel("Avg Severity Score")
plt.savefig(BASE + "\\plots\\plot8_factor_severity.png")
plt.show()
print("saved plot8")

# PLOT 9: does more vehicles involved = worse crash?
vehicle_sev = df.groupby("num_vehicles")["severity_score"].mean()

plt.figure(figsize=(7, 5))
plt.bar(vehicle_sev.index, vehicle_sev.values, color="darkorange")
plt.title("Number of Vehicles Involved vs Average Severity")
plt.xlabel("Number of Vehicles Involved")
plt.ylabel("Avg Severity Score")
plt.savefig(BASE + "\\plots\\plot9_num_vehicles_severity.png")
plt.show()
print("saved plot9")

# PLOT 10: when are cyclists most at risk during the day?
if "number_of_cyclist_injured" in df.columns:
    cyclist = df.groupby("hour")["number_of_cyclist_injured"].sum()

    plt.figure(figsize=(10, 5))
    plt.bar(cyclist.index, cyclist.values, color="darkorange", alpha=0.8)
    plt.title("Cyclist Injuries by Hour of Day")
    plt.xlabel("Hour of Day")
    plt.ylabel("Total Cyclist Injuries")
    plt.xticks(range(0, 24))
    plt.savefig(BASE + "\\plots\\plot10_cyclist_injuries.png")
    plt.show()
    print("saved plot10")

print("\nall plots done!")