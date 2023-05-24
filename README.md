# BPlogger

Phone apps are great for integrating with wearable tech, but often are quite limited in their ability to plot/analyze the data. Additionally, for some quantities they limited users to logging only a single record per-day. 

This repository includes a basic python class for logging blood pressure and heart rate data, as well as doing some basic analysis and plotting of the data.  This can include logging any number of readings per day and importing/exporting from CSV. 

This code demonstrates how to use a basic pydantic model for validation, pandas for logging/data analysis and matplotlib for visualization.  Part of the idea behind this code is to use BP data as an exmaple scientific dataset in order to explore different ways to present raw data, beyond simply just including a CSV file as SI or in a github repository.  Providing routines to load and visualize raw data (in particular using pydantic models to clearly define metadata) within a github SI repository, would greatly improve usability of datasets. 
