# BPlogger

Phone apps are great for integrating with wearable tech, but often are quite limited in their ability to plot/analyze the data. Additionally, for some quantities they limited users to logging only a single record per-day. 

This repository includes a basic python class for logging blood pressure and heart rate data, as well as doing some basic analysis and plotting of the data.  This can include logging any number of readings per day and importing/exporting from CSV. 

This code demonstrates how to use a basic pydantic model for validation, pandas for logging/data analysis and matplotlib for visualization.  Part of the idea behind this code is to use BP data as an exmaple scientific dataset in order to explore different ways to present raw data, beyond simply just including a CSV file as SI or in a github repository.  Providing routines to load and visualize raw data (in particular using pydantic models to clearly define metadata) within a github SI repository, would greatly improve usability of datasets. 


[bp_log_notebook.ipynb](https://github.com/chrisiacovella/BPlogger/blob/main/bp_log_notebook.ipynb) demonstrates the basic usage and built in validation.  For SI material, we obviously wouldn't want necessarily the ability for users to add new data, but providing a basic class with the model and ability to load a CSV file and present summary tables/plots that match those in the text as well as additional clarifying figures (e.g., those in the SI text document) as is shown here, would be helpful.   Todo: add in plotly interactive plots. 



Check out the source from the github repository:

    $ git clone https://github.com/chrisiacovella/BPlogger.git

The core functions of the module will require ``pandas``, ``matplotlib``, and ``pydantic`` to be installed.
To create an environment named bp with this necessary package,
run the following from the top level of the  hoomdxml_reader directory.

    $ conda env create -f environment.yml
