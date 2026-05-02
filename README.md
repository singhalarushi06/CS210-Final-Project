# CS210-Final-Project

**Group Members:** Arushi Singhal, Venkat Aditya Mullagiri, Yug Patel
**Project Topic:** NY Vehicle Collision Analysis

## Project Overview
The goal of this project was to predict the accident severity based on various time-related factors. We analyzed a sample of 100k (out of more than 2 million) NYC motor vehicle collisions to identify any patterns in the accident severity, and understand the impacts for further analysis of the severity of crashes.

## Dataset Source
[NYC Motor Vehicle Collisions - Crashes](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95/about_data)

## Code Files
There are 4 key files for code. They were used in the following order:
 - data_selection.py
 - data_cleaning.py
 - plot.py
 - modeling.py

We used the data_selection.py file to randomly sample the original data (of >2M rows of data), and save that smaller dataset as our data. The data_cleaning.py file was then used to clean up this dataset. It was also used for the feature engineering aspect of our project. Our plot.py file was part 3 of our project. It was used for exploring the data visually and storing the plots generated from that code. These plots were saved in the "plots" folder. The fourth part of our project was the modeling.py file. The purpose of that was to train some different models, evaluate the accuracy of each, identify the most important features, and then save the best model to use as a predictive model for further use.