from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ValidationError, validator
import pandas as pd
import matplotlib.pyplot as plt
import math 
import warnings 

# Use Pydantic to create a model for a data and enable some simple validation
class BPReading(BaseModel):
    systolic: int
    diastolic: int
    heartrate: int
    timestamp: datetime

    @validator('systolic', 'diastolic', 'heartrate', pre=True)
    def check_type(cls, value):
        if not isinstance(value, int):
            raise ValueError('Readings should be integer values')
        return value
        
    @validator('systolic', 'diastolic')
    def check_bp(cls, value, values): 
        # first validate the range
        if value > 200 or value < 10:
            if 'systolic' in values:
                label = 'diastolic'
            else:
                label = 'systolic'
            raise ValueError(f'{label} value is not within a reasonable range.')
        # next validate that 
        if 'systolic' in values:
            if values['systolic'] < value:
               raise ValueError(f'diastolic value ({value}) should not be greater than systolic ({values["systolic"]}).')             
        return value

    @validator('heartrate')
    def check_hr(cls, value):
        if value > 250 or value < 20:
            raise ValueError('heartrate value is not within a reasonable range.')
        return value
            

# Class for logging BP and performing some basic analysis
class BP_log:
    def __init__(self, data=None):
        self.init_df = True
        if data != None:
            self.add_reading(data)
            
    # function for adding a reading  
    def add_reading(self, data):

        if self.init_df == True:
            self.bp_readings = pd.DataFrame(BPReading(**data).dict(), index=['reading'])
            self.init_df = False
        else:
            temp_df = pd.DataFrame(BPReading(**data).dict(), index=['reading'])
            self.bp_readings = pd.concat([self.bp_readings, temp_df])

        # save a backup CSV file everytime we add a new record
        self.save(filename='log_backup.csv')
     
    # Save the entire log to a CSV file (i.e., a raw record)
    def save(self, filename):
        self.bp_readings.to_csv(filename, index=True)
        
    # Save a CSV file of the mean/std values grouped by day
    def save_daily_summary(self, filename):
        self.daily_summary()
        self.bp_daily_summary.to_csv(filename, index=True) 
    
    # Save a CSV file of the mean/std values grouped by week
    def save_weekly_summary(self, filename):
        self.weekly_summary()
        self.bp_weekly_summary.to_csv(filename, index=True) 
        
    # Load the CSV file that contains the raw records
    def load(self, filename):
        if '.csv' in filename:
            dtypes = {'systolic': 'int64', 'diastolic': 'int64', 'heartrate': 'int64', 'timestamp': 'str'} 
            self.bp_readings = pd.read_csv(filename, index_col=0, dtype=dtypes, parse_dates=['timestamp'])
        else:
            raise ValueError(f'Load expects a csv file with extension ".csv". {filename} was provided to the function.' )

    # return a pandas dataframe where values are presented as the mean/std on a weekly basis
    def weekly_summary(self):
        temp = self.bp_readings.copy()
        temp['week of'] = temp['timestamp'] - pd.to_timedelta(7, unit='d')

        self.bp_weekly_summary = pd.DataFrame(temp.groupby([pd.Grouper(key='week of', freq='W')])['systolic'].mean().round(2))
        self.bp_weekly_summary['systolic std'] = temp.groupby([pd.Grouper(key='week of', freq='W')])['systolic'].std().round(2)

        self.bp_weekly_summary['diastolic mean'] = temp.groupby([pd.Grouper(key='week of', freq='W')])['diastolic'].mean().round(2)
        self.bp_weekly_summary['diastolic std'] = temp.groupby([pd.Grouper(key='week of', freq='W')])['diastolic'].std().round(2)

        self.bp_weekly_summary['heartrate mean'] = temp.groupby([pd.Grouper(key='week of', freq='W')])['heartrate'].mean().round(2)
        self.bp_weekly_summary['heartrate std'] = temp.groupby([pd.Grouper(key='week of', freq='W')])['heartrate'].std().round(2)

        self.bp_weekly_summary['# readings'] = temp.groupby([pd.Grouper(key='week of', freq='W')])['diastolic'].size()

        self.bp_weekly_summary = self.bp_weekly_summary.rename(columns={'systolic': 'systolic mean'})
        
        return self.bp_weekly_summary

        
    # return a pandas dataframe where values are presented as the mean/std on a daily basis
    def daily_summary(self):
        temp = self.bp_readings.groupby(self.bp_readings.timestamp.dt.strftime("%Y-%m-%d")).mean().round(2)
        temp = temp.rename(columns={'systolic': 'systolic mean', 'diastolic': 'diastolic mean', 'heartrate': 'heartrate mean'})
                                                                                                                    
        temp_std = self.bp_readings.groupby(self.bp_readings.timestamp.dt.strftime("%Y-%m-%d")).std().round(2)
        temp_std = temp_std.rename(columns={'systolic': 'systolic std', 'diastolic': 'diastolic std', 'heartrate': 'heartrate std'})
        temp_std = temp_std.drop('timestamp', axis=1).fillna(0)


        size  = self.bp_readings.groupby(self.bp_readings.timestamp.dt.strftime("%Y-%m-%d")).size()

        self.bp_daily_summary  = temp.join(temp_std['systolic std']).join(temp_std['diastolic std']).join(temp_std['heartrate std'])
        self.bp_daily_summary = self.bp_daily_summary.drop('timestamp', axis=1)
        
        self.bp_daily_summary = self.bp_daily_summary[['systolic mean', 'systolic std','diastolic mean','diastolic std', 'heartrate mean','heartrate std']]
        size = size.rename('# readings')
        self.bp_daily_summary = self.bp_daily_summary.merge(size, left_index=True, right_index=True)
        
        return self.bp_daily_summary
    
    # generic function to plot bar graphs, called by plot_daily_summary and plot_weekly_summary
    def _plotter(self, data_frame, title, max_labels=30):
        
        fig, ax = plt.subplots(figsize=(20,6))

        s_stderr = data_frame['systolic std'].fillna(0)

        d_stderr = data_frame['diastolic std'].fillna(0)

        times = data_frame.index

        ax.bar(times, data_frame['systolic mean'], align='center', yerr=s_stderr, alpha=0.5, ecolor='blue', capsize=10, label='systolic')
        ax.bar(times, data_frame['diastolic mean'], align='center', yerr=d_stderr, alpha=1.0, ecolor='red', capsize=10, label='diastolic')

        ax.axhline(120, color='b', linestyle='--') # horizontal
        ax.axhline(80, color='r', linestyle='--') # horizontal

        skip_label = int(data_frame['systolic std'].shape[0]/max_labels)
        ax.set_xticks(times[::(skip_label+1)])

        ax.xaxis.axis_date
        fig.autofmt_xdate()
         

        ax.set_ylabel('nmHg')
        ax.set_title(title)
        ax.set_ylabel('nmHg')
        ax.set_ylim(60,140)
        ax.legend(loc='best')
        return fig, ax
    
    # create a bar graph containing the daily summary
    def plot_daily_summary(self, max_labels=30):
        self.daily_summary()
        return self._plotter(self.bp_daily_summary,title='BP log daily summary')
     
    # create a bar graph containing the weekly summary
    def plot_weekly_summary(self, max_labels=30):
        self.weekly_summary()
        return self._plotter(self.bp_weekly_summary,title='BP log weekly summary')

    # plot a histogram of the mean daily values
    def plot_histogram(self, start_date=None, end_date=None, bins=10):
        """ histogram generated from the mean values for each day to avoid biasing based on days with more data """
        if self.bp_readings is None:
            raise ValueError('No BP readings have been added yet.')
        if start_date is None:
            start_date = self.bp_readings['timestamp'].min()
        if end_date is None:
            end_date = self.bp_readings['timestamp'].max()
        
        selection = (self.bp_readings['timestamp'] >= start_date ) & (self.bp_readings['timestamp'] <= end_date)
        axarr = self.bp_readings[selection ][['systolic', 'diastolic']].hist(bins=bins, figsize=(10,5))
        mean = self.bp_readings[ selection][['systolic', 'diastolic']].mean()
        stdev = self.bp_readings[ selection][['systolic', 'diastolic']].std()

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        print(type(axarr))
        keys = ['systolic', 'diastolic']
        for i, ax in enumerate(axarr.flatten()):
            ax.set_xlabel("nmHg")
            ax.axvline(mean[keys[i]], color='red', ls='--', lw=4)
            ax.set_ylabel("frequency")
            ax.text(0.05, 0.95, f'mean: {mean[keys[i]].round(2)}\nstd: {stdev[keys[i]].round(2)}', transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    def __repr__(self):
        msg = f'{self.bp_readings.shape[0]} readings total\n\n'
        msg = msg + self.bp_readings.to_string()
        return msg

           
            