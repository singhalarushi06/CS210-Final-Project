# CS210-Final-Project

**Group Members:** Arushi Singhal, Venkat Aditya Mullagiri, Yug Patel <br>
**Project Topic:** NY Vehicle Collision Analysis <br>
**Due Date:** Monday, May 4th, 2026

## Project Overview
The goal of this project was to predict the accident severity based on various time-related factors. We analyzed a sample of 100k (out of more than 2 million) NYC motor vehicle collisions to identify any patterns in the accident severity, and understand the impacts for further analysis of the severity of crashes.

## Dataset Source
[NYC Motor Vehicle Collisions - Crashes](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95/about_data)




## Code Files
There are 5 key files for code. They were used in the following order(use python 3.12):
 - data_selection.py
 - data_cleaning.py
 - Sql_analysis.py
 - plot.py
 - modeling.py

We used the data_selection.py file to fetch 100k rows directly from the NYC Open Data API using two requests of 50k rows each, and save that as our data. The data_cleaning.py file was then used to clean up this dataset. It was also used for the feature engineering aspect of our project.The Sql_analysis.py file runs 25 SQL queries on the SQLite database created during cleaning. It explores crash patterns across time, geography, contributing factors and vehicle types, saving each result as a CSV in the results folder. Our plot.py file was part 3 of our project. It was used for exploring the data visually and storing the plots generated from that code. These plots were saved in the "plots" folder. The fourth part of our project was the modeling.py file. The purpose of that was to train some different models, evaluate the accuracy of each, identify the most important features, and then save the best model to use as a predictive model for further use.
