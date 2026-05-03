# Part 5: SQL Analysis
# CS210 - NYC Motor Vehicle Collisions
# runs SQL queries on the database and saves results to the results folder
# run after sql_setup.py

import pandas as pd
import sqlite3
import os

# BASE points to the CS210-Final-Project folder regardless of where you run the script from
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# connect to the database we created in sql_setup.py
db_path = BASE + "\\data\\collisions.db"
conn = sqlite3.connect(db_path)
print(f"Connected to {db_path}")

# make sure results folder exists
os.makedirs(BASE + "\\results", exist_ok=True)

# helper function so we dont repeat the same connect/print/save steps every time
def run_query(label, query):
    print(f"\n{'='*60}")
    print(f"Query: {label}")
    print('='*60)
    df = pd.read_sql(query, conn)
    print(df.to_string(index=False))
    filename = label.lower().replace(" ", "_") + ".csv"
    df.to_csv(BASE + "\\results\\" + filename, index=False)
    print(f"Saved -> results\\{filename}")
    return df


# TIME ANALYSIS QUERIES
# these answer the core research question about when crashes are worst

# Q1: total crashes and severity by hour 
run_query("q01_crashes_by_hour", """
    SELECT 
        hour,
        COUNT(*) as total_crashes,
        ROUND(AVG(severity_score), 3) as avg_severity,
        SUM(is_severe) as severe_crashes,
        ROUND(SUM(is_severe) * 100.0 / COUNT(*), 1) as pct_severe
    FROM crashes
    GROUP BY hour
    ORDER BY hour
""")

# Q2: which single hour has the absolute worst average severity
# interesting because it might not be rush hour
run_query("q02_deadliest_hours", """
    SELECT 
        hour,
        COUNT(*) as total_crashes,
        ROUND(AVG(severity_score), 3) as avg_severity,
        SUM(is_severe) as severe_crashes
    FROM crashes
    GROUP BY hour
    ORDER BY avg_severity DESC
    LIMIT 5
""")

# Q3: rush hour vs non rush hour. Are rush hour crashes actually more severe or just more frequent
run_query("q03_rush_hour_vs_not", """
    SELECT 
        CASE WHEN is_rush_hour = 1 THEN 'Rush Hour' ELSE 'Not Rush Hour' END as time_period,
        COUNT(*) as total_crashes,
        ROUND(AVG(severity_score), 3) as avg_severity,
        SUM(is_severe) as severe_crashes,
        ROUND(SUM(is_severe) * 100.0 / COUNT(*), 1) as pct_severe
    FROM crashes
    GROUP BY is_rush_hour
""")

# Q4: late night vs everything else (are midnight crashes worse than the rest of the day) 
run_query("q04_late_night_vs_normal", """
    SELECT 
        CASE WHEN is_night = 1 THEN 'Late Night (12am-5am)' ELSE 'Normal Hours' END as time_period,
        COUNT(*) as total_crashes,
        ROUND(AVG(severity_score), 3) as avg_severity,
        SUM(is_severe) as severe_crashes,
        ROUND(SUM(is_severe) * 100.0 / COUNT(*), 1) as pct_severe
    FROM crashes
    GROUP BY is_night
""")

# Q5: weekday vs weekend breakdown
run_query("q05_weekday_vs_weekend", """
    SELECT 
        CASE WHEN is_weekend = 1 THEN 'Weekend' ELSE 'Weekday' END as day_type,
        COUNT(*) as total_crashes,
        ROUND(AVG(severity_score), 3) as avg_severity,
        SUM(is_severe) as severe_crashes,
        ROUND(SUM(is_severe) * 100.0 / COUNT(*), 1) as pct_severe
    FROM crashes
    GROUP BY is_weekend
""")

# Q6: the dangerous combo of late night and weekend together
# this is probably the worst possible time to be on the road
run_query("q06_night_weekend_combo", """
    SELECT 
        CASE 
            WHEN is_night = 1 AND is_weekend = 1 THEN 'Late Night + Weekend'
            WHEN is_night = 1 AND is_weekend = 0 THEN 'Late Night + Weekday'
            WHEN is_night = 0 AND is_weekend = 1 THEN 'Normal Hours + Weekend'
            ELSE 'Normal Hours + Weekday'
        END as time_combo,
        COUNT(*) as total_crashes,
        ROUND(AVG(severity_score), 3) as avg_severity,
        SUM(is_severe) as severe_crashes,
        ROUND(SUM(is_severe) * 100.0 / COUNT(*), 1) as pct_severe
    FROM crashes
    GROUP BY is_night, is_weekend
    ORDER BY avg_severity DESC
""")

# Q7: crashes by month. Are there seasonal patterns where certain months are worse than others
run_query("q07_crashes_by_month", """
    SELECT 
        month,
        COUNT(*) as total_crashes,
        ROUND(AVG(severity_score), 3) as avg_severity,
        SUM(is_severe) as severe_crashes
    FROM crashes
    GROUP BY month
    ORDER BY month
""")


# GEOGRAPHIC QUERIES
# these show where crashes are worst

# Q8: crashes and severity by borough
run_query("q08_crashes_by_borough", """
    SELECT 
        borough,
        COUNT(*) as total_crashes,
        ROUND(AVG(severity_score), 3) as avg_severity,
        SUM(is_severe) as severe_crashes,
        ROUND(SUM(is_severe) * 100.0 / COUNT(*), 1) as pct_severe
    FROM crashes
    WHERE borough != 'Unknown'
    GROUP BY borough
    ORDER BY total_crashes DESC
""")

# Q9: top 10 most dangerous borough + hour combos
# where AND when are the worst crashes happening
run_query("q09_dangerous_borough_hour", """
    SELECT 
        borough,
        hour,
        COUNT(*) as total_crashes,
        ROUND(AVG(severity_score), 3) as avg_severity,
        SUM(is_severe) as severe_crashes
    FROM crashes
    WHERE borough != 'Unknown'
    GROUP BY borough, hour
    HAVING COUNT(*) >= 10
    ORDER BY avg_severity DESC
    LIMIT 10
""")

# Q10: which borough has the most late night crashes
# interesting to see if manhattan nightlife shows up here
run_query("q10_borough_late_night_crashes", """
    SELECT 
        borough,
        COUNT(*) as late_night_crashes,
        ROUND(AVG(severity_score), 3) as avg_severity,
        ROUND(SUM(is_severe) * 100.0 / COUNT(*), 1) as pct_severe
    FROM crashes
    WHERE is_night = 1 AND borough != 'Unknown'
    GROUP BY borough
    ORDER BY late_night_crashes DESC
""")

# Q11: which borough has the worst weekend crashes
run_query("q11_borough_weekend_severity", """
    SELECT 
        borough,
        COUNT(*) as weekend_crashes,
        ROUND(AVG(severity_score), 3) as avg_severity,
        SUM(is_severe) as severe_crashes
    FROM crashes
    WHERE is_weekend = 1 AND borough != 'Unknown'
    GROUP BY borough
    ORDER BY avg_severity DESC
""")


# CONTRIBUTING FACTOR QUERIES
# these show what behaviors cause the worst crashes

# Q12: crashes by factor group overall
run_query("q12_crashes_by_factor_group", """
    SELECT 
        factor_group,
        COUNT(*) as total_crashes,
        ROUND(AVG(severity_score), 3) as avg_severity,
        SUM(is_severe) as severe_crashes,
        ROUND(SUM(is_severe) * 100.0 / COUNT(*), 1) as pct_severe
    FROM crashes
    WHERE factor_group != 'Other'
    GROUP BY factor_group
    ORDER BY avg_severity DESC
""")

# Q13: alcohol/drug crashes by hour. Now this should heavily spike late night
# if this is true it strongly supports the night driving danger finding
run_query("q13_alcohol_crashes_by_hour", """
    SELECT 
        hour,
        COUNT(*) as alcohol_drug_crashes,
        ROUND(AVG(severity_score), 3) as avg_severity
    FROM crashes
    WHERE factor_group = 'Alcohol/Drugs'
    GROUP BY hour
    ORDER BY alcohol_drug_crashes DESC
    LIMIT 10
""")

# Q14: distraction crashes by hour. Should spike during commute hours
# distracted driving during rush hour vs alcohol at night comparison
run_query("q14_distraction_crashes_by_hour", """
    SELECT 
        hour,
        COUNT(*) as distraction_crashes,
        ROUND(AVG(severity_score), 3) as avg_severity
    FROM crashes
    WHERE factor_group = 'Distraction'
    GROUP BY hour
    ORDER BY distraction_crashes DESC
    LIMIT 10
""")

# Q15: fatigue crashes by hour. Should spike in early morning 4am-6am
# this tests the classic drowsy driving danger window
run_query("q15_fatigue_crashes_by_hour", """
    SELECT 
        hour,
        COUNT(*) as fatigue_crashes,
        ROUND(AVG(severity_score), 3) as avg_severity
    FROM crashes
    WHERE factor_group = 'Fatigue'
    GROUP BY hour
    ORDER BY fatigue_crashes DESC
    LIMIT 10
""")

# Q16: which factor group is most common in each borough
# does brooklyn have more distraction while manhattan has more speeding
run_query("q16_factor_group_by_borough", """
    SELECT 
        borough,
        factor_group,
        COUNT(*) as total_crashes,
        ROUND(AVG(severity_score), 3) as avg_severity
    FROM crashes
    WHERE borough != 'Unknown' AND factor_group != 'Other'
    GROUP BY borough, factor_group
    ORDER BY borough, total_crashes DESC
""")


# VEHICLE TYPE QUERIES
# these show which vehicles are involved in the worst crashes

# Q17: severity by vehicle type - motorcycles should be way worse than sedans
run_query("q17_severity_by_vehicle_type", """
    SELECT 
        vehicle_type_code1,
        COUNT(*) as total_crashes,
        ROUND(AVG(severity_score), 3) as avg_severity,
        SUM(is_severe) as severe_crashes,
        ROUND(SUM(is_severe) * 100.0 / COUNT(*), 1) as pct_severe
    FROM crashes
    WHERE vehicle_type_code1 NOT IN ('OTHER', 'Unknown')
    GROUP BY vehicle_type_code1
    HAVING COUNT(*) >= 50
    ORDER BY avg_severity DESC
""")

# Q18: number of vehicles involved vs severity
# does 3+ vehicles = much worse outcome
run_query("q18_num_vehicles_vs_severity", """
    SELECT 
        num_vehicles,
        COUNT(*) as total_crashes,
        ROUND(AVG(severity_score), 3) as avg_severity,
        SUM(is_severe) as severe_crashes,
        ROUND(SUM(is_severe) * 100.0 / COUNT(*), 1) as pct_severe
    FROM crashes
    GROUP BY num_vehicles
    ORDER BY num_vehicles
""")

# Q19: truck crashes by borough - trucks tend to cause worse damage
run_query("q19_truck_crashes_by_borough", """
    SELECT 
        borough,
        COUNT(*) as truck_crashes,
        ROUND(AVG(severity_score), 3) as avg_severity,
        SUM(is_severe) as severe_crashes
    FROM crashes
    WHERE vehicle_type_code1 = 'TRUCK' AND borough != 'Unknown'
    GROUP BY borough
    ORDER BY truck_crashes DESC
""")


# VULNERABLE ROAD USER QUERIES
# pedestrians and cyclists are the most at risk

# Q20: cyclist injuries by hour - when are cyclists most at risk
run_query("q20_cyclist_injuries_by_hour", """
    SELECT 
        hour,
        SUM(number_of_cyclist_injured) as total_cyclist_injured,
        COUNT(*) as total_crashes,
        ROUND(AVG(number_of_cyclist_injured), 4) as avg_cyclist_injured_per_crash
    FROM crashes
    GROUP BY hour
    ORDER BY total_cyclist_injured DESC
    LIMIT 10
""")

# Q21: pedestrian injuries by borough - where are pedestrians most at risk
run_query("q21_pedestrian_injuries_by_borough", """
    SELECT 
        borough,
        SUM(number_of_pedestrians_injured) as total_pedestrians_injured,
        SUM(number_of_pedestrians_killed) as total_pedestrians_killed,
        COUNT(*) as total_crashes,
        ROUND(AVG(number_of_pedestrians_injured), 4) as avg_injured_per_crash
    FROM crashes
    WHERE borough != 'Unknown'
    GROUP BY borough
    ORDER BY total_pedestrians_injured DESC
""")

# Q22: cyclist crashes by borough - which borough is most dangerous for cyclists
run_query("q22_cyclist_crashes_by_borough", """
    SELECT 
        borough,
        SUM(number_of_cyclist_injured) as total_cyclist_injured,
        SUM(number_of_cyclist_killed) as total_cyclist_killed,
        COUNT(*) as total_crashes
    FROM crashes
    WHERE borough != 'Unknown'
    GROUP BY borough
    ORDER BY total_cyclist_injured DESC
""")


# COMBINED/INTERACTION QUERIES
# these cross multiple dimensions and tend to give the most interesting findings

# Q23: alcohol crashes late night by borough
# where does drunk driving at night concentrate
run_query("q23_alcohol_night_by_borough", """
    SELECT 
        borough,
        COUNT(*) as alcohol_night_crashes,
        ROUND(AVG(severity_score), 3) as avg_severity
    FROM crashes
    WHERE factor_group = 'Alcohol/Drugs' 
    AND is_night = 1
    AND borough != 'Unknown'
    GROUP BY borough
    ORDER BY alcohol_night_crashes DESC
""")

# Q24: motorcycle crashes by hour - when do bikers crash most
run_query("q24_motorcycle_crashes_by_hour", """
    SELECT 
        hour,
        COUNT(*) as motorcycle_crashes,
        ROUND(AVG(severity_score), 3) as avg_severity,
        SUM(is_severe) as severe_crashes
    FROM crashes
    WHERE vehicle_type_code1 = 'MOTORCYCLE'
    GROUP BY hour
    ORDER BY motorcycle_crashes DESC
    LIMIT 10
""")

# Q25: speeding crashes by borough and hour
# where and when do speeding crashes tend to happen
run_query("q25_speeding_by_borough_hour", """
    SELECT 
        borough,
        CASE 
            WHEN hour BETWEEN 6 AND 9   THEN 'Morning Rush'
            WHEN hour BETWEEN 10 AND 15 THEN 'Midday'
            WHEN hour BETWEEN 16 AND 19 THEN 'Evening Rush'
            WHEN hour BETWEEN 20 AND 23 THEN 'Night'
            ELSE 'Late Night'
        END as time_of_day,
        COUNT(*) as speeding_crashes,
        ROUND(AVG(severity_score), 3) as avg_severity
    FROM crashes
    WHERE factor_group = 'Speeding' AND borough != 'Unknown'
    GROUP BY borough, time_of_day
    ORDER BY speeding_crashes DESC
    LIMIT 15
""")

conn.close()
print("\n" + "="*60)
print("All 25 queries done!")
print("Check the results folder for all CSV files")
print("="*60)