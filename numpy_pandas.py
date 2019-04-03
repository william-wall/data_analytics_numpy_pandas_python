# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 11:30:15 2019

@author: william wall @ williamwall.ie

"""

# a relationship between [sec28]Tobacco, Alcohol, Drugs, [sec31]Fighting and Violence and [sec33]Suicide

# import librarys 
import pandas
import numpy
import seaborn
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi

# rename columns to suitable names
addHealth_dict =  {
        
  'AID': 'IDENTIFIER',
  'H1FV3': 'SHOT_YOU',
  'H1FV7' : 'YOU_PULLED_KNIFE_GUN',
   'H1FV8' : 'YOU_SHOT_STAB',
   'H1SU2' : 'YOU_ATTEMPT_SUICIDE',
   'H1TO36': 'COCAINE',
   'H1TO1' : 'SMOKE',
   'H1TO15' : 'ALCOHOL',
   'H1GI20' : 'AGE',
   'H1TO32' : 'MARIJUANA',
   'H1TO42' : 'GENERAL_DRUG'
   
}

# load dataset
health_data = pandas.read_csv('addhealth_pds.csv', low_memory=False)

# rename the columns
print('Data read ... performing rename operation ...')
health_data.rename(columns=addHealth_dict, inplace=True)

print('Data fetched ...')
# test the remaining
health_data.columns

# print out length of rows and columns 
print('Data details')
print('====================================================================================')
print('Fetched: ' + str(len(health_data)) + ' - rows')  # print length of data
print('====================================================================================')
print('Fetched: ' + str(len(health_data.columns)) + ' - columns')  # print number of columns


# update the dataframe to only contain the columns you do want to use in your analysis
health_data = pandas.DataFrame(health_data, columns = addHealth_dict.values())

# display new column amount
print('====================================================================================')
print('New column amount: ' + str(len(health_data.columns)))
health_data.head
health_data.tail
health_data.dtypes

# convert all columns names to uppercase
health_data.columns = map(str.upper, health_data.columns)

# bug fix for display formats to avoid run time errors
pandas.set_option('display.float_format',lambda x:'%f'%x)

# set PANDAS to show all columns in Data frame
pandas.set_option('display.max_columns', None)

# set PANDAS to show all rows in Data frame
pandas.set_option('display.max_rows', None)

# printing records

# number of observations (rows) 6504
print('====================================================================================')
print('Number of Observations - ROWS')
print (len(health_data)) 
print('====================================================================================')

# number of variables (columns) 2829
print('Number of Variables - COLUMNS')
print (len(health_data.columns))
print('====================================================================================')

# converting variables to numeric
print('*** VARIABLES TO NUMERIC ***')

# Unique identifier for records
health_data['IDENTIFIER'] = pandas.to_numeric(health_data['IDENTIFIER'],errors='coerce')
# During the past 30 days, how many times did you use any of these types of illegal drugs? 
#  LSD, PCP, ecstasy, mushrooms, speed, ice, heroin, or pills
health_data['GENERAL_DRUG'] = pandas.to_numeric(health_data['GENERAL_DRUG'],errors='coerce')
# During the past 30 days, how many times did you use marijuana?
health_data['MARIJUANA'] = pandas.to_numeric(health_data['MARIJUANA'],errors='coerce')
# What grade {ARE/WERE} you in - This variable grade will be recoded to age below based on grade level
health_data['AGE'] = pandas.to_numeric(health_data['AGE'],errors='coerce')
# During the past 12 months, on how many days did you drink alcohol? 
health_data['ALCOHOL'] = pandas.to_numeric(health_data['ALCOHOL'],errors='coerce')
#Have you ever tried cigarette smoking, even just 1 or 2 puffs?
health_data['SMOKE'] = pandas.to_numeric(health_data['SMOKE'],errors='coerce')
# someone shot you
health_data['SHOT_YOU'] = pandas.to_numeric(health_data['SHOT_YOU'],errors='coerce')
#  You pulled a knife or gun on someone
health_data['YOU_PULLED_KNIFE_GUN'] = pandas.to_numeric(health_data['YOU_PULLED_KNIFE_GUN'],errors='coerce')
# You shot or stabbed someone
health_data['YOU_SHOT_STAB'] = pandas.to_numeric(health_data['YOU_SHOT_STAB'],errors='coerce')
# how many times did you actually attempt suicide?
health_data['YOU_ATTEMPT_SUICIDE'] = pandas.to_numeric(health_data['YOU_ATTEMPT_SUICIDE'],errors='coerce')
#  During the past 30 days, how many times did you use cocaine?
health_data['COCAINE'] = pandas.to_numeric(health_data['COCAINE'],errors='coerce')

# descriptive statistics for each variable
print('*** DESCRIPTIVE STATISTICS ***')
print('====================================================================================')
print('Stats General Drugs')
descGen = health_data['GENERAL_DRUG'].describe()
print(descGen)
print('====================================================================================')

print('Stats MARIJUANA')
descMar = health_data['MARIJUANA'].describe()
print(descMar)
print('====================================================================================')

print('Stats AGE')
ageDesc = health_data['AGE'].describe()
print(ageDesc)
print('====================================================================================')

print('Stats ALCOHOL')
descAlc = health_data['ALCOHOL'].describe()
print(descAlc)
print('====================================================================================')

print('Stats SMOKE')
descSmo = health_data['SMOKE'].describe()
print(descSmo)
print('====================================================================================')

print('Stats SHOT YOU')
descSho = health_data['SHOT_YOU'].describe()
print(descSho)
print('====================================================================================')

print('Stats YOU PULLED KNIFE OR GUN')
descKni = health_data['YOU_PULLED_KNIFE_GUN'].describe()
print(descKni)
print('====================================================================================')

print('Stats YOU SHOT OR STABBED SOMEONE')
descSta = health_data['YOU_SHOT_STAB'].describe()
print(descSta)
print('====================================================================================')

print('Stats YOU ATTEMPTED SUICIDE')
descSui = health_data['YOU_ATTEMPT_SUICIDE'].describe()
print(descSui)
print('====================================================================================')

print('Stats YOU TOOK COCAINE')
cocaineDesc = health_data['COCAINE'].describe()
print(cocaineDesc)

print('====================================================================================')
print('*** FREQUENCY DISTRIBUTIONS ***')
# generate frequency counts and sort
print('Counts MARIJUANA -  # During the past 30 days, how many times did you use marijuana?')
freqCountMar= health_data["MARIJUANA"].value_counts(sort=True)
print (freqCountMar)
print('====================================================================================')

# percentages for each value
print('Percentages MARIJUANA - Get percentages for each value based on those counts')
percentageCountMar= health_data["MARIJUANA"].value_counts(sort=True, normalize=True)
print (percentageCountMar)
print('====================================================================================')

# generate frequency counts and sort
print('Counts GENERAL_DRUG -  # During the past 30 days, how many times did you use these drugs?')
freqCountGen= health_data["GENERAL_DRUG"].value_counts(sort=True)
print (freqCountGen)
print('====================================================================================')

# percentages for each value
print('Percentages GENERAL_DRUG - Get percentages for each value based on those counts')
percentageCountGen= health_data["GENERAL_DRUG"].value_counts(sort=True, normalize=True)
print (percentageCountGen)
print('====================================================================================')

# generate frequency counts and sort
print('Counts AGE -  What grade {ARE/WERE} you in?')
freqCountAge= health_data["AGE"].value_counts(sort=True)
print (freqCountAge)
print('====================================================================================')

# percentages for each value
print('Percentages AGE - Get percentages for each value based on those counts')
percentageCountAge= health_data["AGE"].value_counts(sort=True, normalize=True)
print (percentageCountAge)
print('====================================================================================')

# generate frequency counts and sort
print('Counts ALCOHOL - During the past 12 months, on how many days did you drink alcohol?')
freqCountAlc= health_data["ALCOHOL"].value_counts(sort=True)
print (freqCountAlc)
print('====================================================================================')

# percentages for each value
print('Percentages ALCOHOL - Get percentages for each value based on those counts')
percentageCountAlc= health_data["ALCOHOL"].value_counts(sort=True, normalize=True)
print (percentageCountAlc)
print('====================================================================================')

# generate frequency counts and sort
print('Counts SMOKE - Did you ever smoke')
freqCountSmo= health_data["SMOKE"].value_counts(sort=True)
print (freqCountSmo)
print('====================================================================================')

# percentages for each value
print('Percentages SMOKE - Get percentages for each value based on those counts')
percentageCountSmo= health_data["SMOKE"].value_counts(sort=True, normalize=True)
print (percentageCountSmo)
print('====================================================================================')

# generate frequency counts and sort
print('Counts SHOT_YOU - In the past 12 months, someone shot you')
freqCountSho= health_data["SHOT_YOU"].value_counts(sort=True)
print (freqCountSho)
print('====================================================================================')

# percentages for each value
print('Percentages SHOT_YOU - Get percentages for each value based on those counts')
percentageCountSho= health_data["SHOT_YOU"].value_counts(sort=True, normalize=True)
print (percentageCountSho)
print('====================================================================================')

# generate frequency counts and sort
print('Counts YOU_PULLED_KNIFE_GUN - In the past 12 months, You pulled a knife or gun on someone')
freqCountKni= health_data["YOU_PULLED_KNIFE_GUN"].value_counts(sort=True)
print (freqCountKni)
print('====================================================================================')

# percentages for each value
print('Percentages YOU_PULLED_KNIFE_GUN - Get percentages for each value based on those counts')
percentageCountKni= health_data["YOU_PULLED_KNIFE_GUN"].value_counts(sort=True, normalize=True)
print (percentageCountKni)
print('====================================================================================')

# generate frequency counts and sort
print('Counts YOU_SHOT_STAB - You shot or stabbed someone ')
freqCountSta= health_data["YOU_SHOT_STAB"].value_counts(sort=True)
print (freqCountSta)
print('====================================================================================')

# percentages for each value
print('Percentages YOU_SHOT_STAB - Get percentages for each value based on those counts')
percentageCountSta= health_data["YOU_SHOT_STAB"].value_counts(sort=True, normalize=True)
print (percentageCountSta)
print('====================================================================================')

# generate frequency counts and sort
print('Counts YOU_ATTEMPT_SUICIDE -  how many times did you actually attempt suicide?')
freqCountSui= health_data["YOU_ATTEMPT_SUICIDE"].value_counts(sort=True)
print (freqCountSui)
print('====================================================================================')

# percentages for each value
print('Percentages YOU_ATTEMPT_SUICIDE - Get percentages for each value based on those counts')
percentageCountSui= health_data["YOU_ATTEMPT_SUICIDE"].value_counts(sort=True, normalize=True)
print (percentageCountSui)
print('====================================================================================')

# generate frequency counts and sort
print('Counts COCAINE -  During the past 30 days, how many times did you use cocaine')
cocaineFreqCount= health_data["COCAINE"].value_counts(sort=True)
print (cocaineFreqCount)
print('====================================================================================')

# percentages for each value
print('Percentages COCAINE - Get percentages for each value based on those counts')
cocainePercentageCount = health_data["COCAINE"].value_counts(sort=True, normalize=True)
print (cocainePercentageCount)

print('====================================================================================')
print('*** REPLACE UNWANTED VALUES ***')
# remove unwanted data - remove columns that are not needed
health_data['MARIJUANA']=health_data['MARIJUANA'].replace(996, numpy.NaN)
health_data['MARIJUANA']=health_data['MARIJUANA'].replace(997, numpy.NaN)
health_data['MARIJUANA']=health_data['MARIJUANA'].replace(998, numpy.NaN)
health_data['MARIJUANA']=health_data['MARIJUANA'].replace(999, numpy.NaN)
health_data['MARIJUANA']=health_data['MARIJUANA'].replace(0, numpy.NaN)

health_data['AGE']=health_data['AGE'].replace(96, numpy.NaN)
health_data['AGE']=health_data['AGE'].replace(97, numpy.NaN)
health_data['AGE']=health_data['AGE'].replace(98, numpy.NaN)
health_data['AGE']=health_data['AGE'].replace(99, numpy.NaN)

health_data['ALCOHOL']=health_data['ALCOHOL'].replace(7, numpy.NaN)
health_data['ALCOHOL']=health_data['ALCOHOL'].replace(96, numpy.NaN)
health_data['ALCOHOL']=health_data['ALCOHOL'].replace(97, numpy.NaN)
health_data['ALCOHOL']=health_data['ALCOHOL'].replace(98, numpy.NaN)

health_data['SMOKE']=health_data['SMOKE'].replace(6, numpy.NaN)
health_data['SMOKE']=health_data['SMOKE'].replace(8, numpy.NaN)
health_data['SMOKE']=health_data['SMOKE'].replace(9, numpy.NaN)

health_data['SHOT_YOU']=health_data['SHOT_YOU'].replace(0, numpy.NaN)
health_data['SHOT_YOU']=health_data['SHOT_YOU'].replace(6, numpy.NaN)
health_data['SHOT_YOU']=health_data['SHOT_YOU'].replace(8, numpy.NaN)
health_data['SHOT_YOU']=health_data['SHOT_YOU'].replace(9, numpy.NaN)

health_data['YOU_PULLED_KNIFE_GUN']=health_data['YOU_PULLED_KNIFE_GUN'].replace(0, numpy.NaN)
health_data['YOU_PULLED_KNIFE_GUN']=health_data['YOU_PULLED_KNIFE_GUN'].replace(6, numpy.NaN)
health_data['YOU_PULLED_KNIFE_GUN']=health_data['YOU_PULLED_KNIFE_GUN'].replace(8, numpy.NaN)
health_data['YOU_PULLED_KNIFE_GUN']=health_data['YOU_PULLED_KNIFE_GUN'].replace(9, numpy.NaN)

health_data['YOU_SHOT_STAB']=health_data['YOU_SHOT_STAB'].replace(0, numpy.NaN)
health_data['YOU_SHOT_STAB']=health_data['YOU_SHOT_STAB'].replace(6, numpy.NaN)
health_data['YOU_SHOT_STAB']=health_data['YOU_SHOT_STAB'].replace(8, numpy.NaN)
health_data['YOU_SHOT_STAB']=health_data['YOU_SHOT_STAB'].replace(9, numpy.NaN)

health_data['YOU_ATTEMPT_SUICIDE']=health_data['YOU_ATTEMPT_SUICIDE'].replace(0, numpy.NaN)
health_data['YOU_ATTEMPT_SUICIDE']=health_data['YOU_ATTEMPT_SUICIDE'].replace(6, numpy.NaN)
health_data['YOU_ATTEMPT_SUICIDE']=health_data['YOU_ATTEMPT_SUICIDE'].replace(7, numpy.NaN)
health_data['YOU_ATTEMPT_SUICIDE']=health_data['YOU_ATTEMPT_SUICIDE'].replace(8, numpy.NaN)

health_data['COCAINE']=health_data['COCAINE'].replace(0, numpy.NaN)
health_data['COCAINE']=health_data['COCAINE'].replace(996, numpy.NaN)
health_data['COCAINE']=health_data['COCAINE'].replace(997, numpy.NaN)
health_data['COCAINE']=health_data['COCAINE'].replace(998, numpy.NaN)
health_data['COCAINE']=health_data['COCAINE'].replace(999, numpy.NaN)

health_data['GENERAL_DRUG']=health_data['GENERAL_DRUG'].replace(0, numpy.NaN)
health_data['GENERAL_DRUG']=health_data['GENERAL_DRUG'].replace(996, numpy.NaN)
health_data['GENERAL_DRUG']=health_data['GENERAL_DRUG'].replace(997, numpy.NaN)
health_data['GENERAL_DRUG']=health_data['GENERAL_DRUG'].replace(998, numpy.NaN)
health_data['GENERAL_DRUG']=health_data['GENERAL_DRUG'].replace(999, numpy.NaN)

print('*** SUM OF REMOVED VALUES ***')
# verify
print((health_data['ALCOHOL']==7).sum())
print((health_data['ALCOHOL']==96).sum())
print((health_data['ALCOHOL']==97).sum())
print((health_data['ALCOHOL']==98).sum())

print((health_data['AGE']==96).sum())
print((health_data['AGE']==97).sum())
print((health_data['AGE']==98).sum())
print((health_data['AGE']==99).sum())

print((health_data['SMOKE']==6).sum())
print((health_data['SMOKE']==8).sum())
print((health_data['SMOKE']==9).sum())

print((health_data['SHOT_YOU']==6).sum())
print((health_data['SHOT_YOU']==8).sum())
print((health_data['SHOT_YOU']==9).sum())

print((health_data['YOU_PULLED_KNIFE_GUN']==6).sum())
print((health_data['YOU_PULLED_KNIFE_GUN']==8).sum())
print((health_data['YOU_PULLED_KNIFE_GUN']==9).sum())

print((health_data['YOU_SHOT_STAB']==0).sum())
print((health_data['YOU_SHOT_STAB']==6).sum())
print((health_data['YOU_SHOT_STAB']==8).sum())
print((health_data['YOU_SHOT_STAB']==9).sum())

print((health_data['YOU_ATTEMPT_SUICIDE']==6).sum())
print((health_data['YOU_ATTEMPT_SUICIDE']==7).sum())
print((health_data['YOU_ATTEMPT_SUICIDE']==8).sum())

print((health_data['COCAINE']==0).sum())
print((health_data['COCAINE']==996).sum())
print((health_data['COCAINE']==997).sum())
print((health_data['COCAINE']==998).sum())
print((health_data['COCAINE']==999).sum())

print('*** VALUES ARE NULL ***')
# make sure values are null
print(health_data['ALCOHOL'].isnull().sum())
print(health_data['MARIJUANA'].isnull().sum())
print(health_data['AGE'].isnull().sum())
print(health_data['SMOKE'].isnull().sum())
print(health_data['SHOT_YOU'].isnull().sum())
print(health_data['YOU_PULLED_KNIFE_GUN'].isnull().sum())
print(health_data['YOU_SHOT_STAB'].isnull().sum())
print(health_data['YOU_ATTEMPT_SUICIDE'].isnull().sum())
print(health_data['COCAINE'].isnull().sum())

print('====================================================================================')

print('*** GROUP BY ***')
## group by Marijuana
print('Marijuana')
groupCountMarijuana = health_data.groupby('MARIJUANA').size()
print(groupCountMarijuana)

# percentage groupby counts
print('percentages with the groupby counts')
percentageGroupCountMarijuana = health_data.groupby('MARIJUANA').size() * 100 / len(health_data)
print(percentageGroupCountMarijuana)
print('====================================================================================')

# group by general drugs
print('General drugs')
groupCountGen = health_data.groupby('GENERAL_DRUG').size()
print(groupCountGen)

# percentage groupby counts
print('percentages with the groupby counts')
percentageGroupCountGen = health_data.groupby('GENERAL_DRUG').size() * 100 / len(health_data)
print(percentageGroupCountGen)
print('====================================================================================')

# group by AGE
print('School grade')
groupCountAge = health_data.groupby('AGE').size()
print(groupCountAge)

# percentage groupby counts
print('percentages with the groupby counts')
percentageGroupCountAge = health_data.groupby('AGE').size() * 100 / len(health_data)
print(percentageGroupCountAge)
print('====================================================================================')

# group by ALCOHOL
print('Did you ever alcohol')
groupCountAlc = health_data.groupby('ALCOHOL').size()
print(groupCountAlc)

# percentage groupby counts
print('percentages with the groupby counts')
percentageGroupCountAlc = health_data.groupby('ALCOHOL').size() * 100 / len(health_data)
print(percentageGroupCountAlc)
print('====================================================================================')

# group by SMOKE
print('Did you ever smoke')
groupCountSmo = health_data.groupby('SMOKE').size()
print(groupCountSmo)

# percentage groupby counts
print('percentages with the groupby counts')
percentageGroupCountSmo = health_data.groupby('SMOKE').size() * 100 / len(health_data)
print(percentageGroupCountSmo)
print('====================================================================================')

print('In the past 12 months, someone shot you - counts grouped by the values')
groupCountSho = health_data.groupby('SHOT_YOU').size()
print(groupCountSho)

# percentage groupby counts
print('percentages with the groupby counts')
percentageGroupCountSho = health_data.groupby('SHOT_YOU').size() * 100 / len(health_data)
print(percentageGroupCountSho)
print('====================================================================================')

# group by YOU_PULLED_KNIFE_GUN
print('You pulled a knife or gun on someone - counts grouped by the values')
groupCountKni = health_data.groupby('YOU_PULLED_KNIFE_GUN').size()
print(groupCountKni)

# percentage groupby counts
print('percentages with the groupby counts')
percentageGroupCountKni = health_data.groupby('YOU_PULLED_KNIFE_GUN').size() * 100 / len(health_data)
print(percentageGroupCountKni)
print('====================================================================================')

# group by YOU_SHOT_STAB
print('You shot or stabbed someone - counts grouped by the values')
groupCountSta = health_data.groupby('YOU_SHOT_STAB').size()
print(groupCountSta)

# percentage groupby counts
print('percentages with the groupby counts')
percentageGroupCountSta = health_data.groupby('YOU_SHOT_STAB').size() * 100 / len(health_data)
print(percentageGroupCountSta)
print('====================================================================================')

# group by YOU_ATTEMPT_SUICIDE 
print('how many times did you actually attempt suicide? - counts grouped by the values')
groupCountSui = health_data.groupby('YOU_ATTEMPT_SUICIDE').size()
print(groupCountSui)

# percentage groupby counts
print('percentages with the groupby counts')
percentageGroupCountSui = health_data.groupby('YOU_ATTEMPT_SUICIDE').size() * 100 / len(health_data)
print(percentageGroupCountSui)
print('====================================================================================')

# group by COCAINE  
print('During the past 30 days, how many times did you use cocaine? - counts grouped by the values')
groupCountCocaine = health_data.groupby('COCAINE').size()
print(groupCountCocaine)
# percentage groupby counts

print('percentages with the groupby counts')
percentageGroupCountCocaine = health_data.groupby('COCAINE').size() * 100 / len(health_data)
print(percentageGroupCountCocaine)
print('====================================================================================')

# recoding
print('*** RECODING ***')
print('====================================================================================')

# recode marijuana
print('During the past 30 days, how many times did you use  LSD, PCP, ecstasy, mushrooms, speed, ice, heroin, or pills? ')
recodeGen= {1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '10', 12: '12', 
            14: '14', 15: '15', 16: '16', 20: '20', 25: '25', 30: '30', 31: '31', 34: '34',
            40: '40', 50: '50', 90: '90', 100: '100', 111: '111', 132: '132'}
health_data['GENERAL_DRUG_RECODE'] = health_data['GENERAL_DRUG'].map(recodeGen)
freqCount_recodeGen= health_data["GENERAL_DRUG_RECODE"].value_counts(sort=True)
print (freqCount_recodeGen)
print('====================================================================================')


# recode marijuana
print('During the past 30 days, how many times did you use marijuana? ')
recodeMar = {1: '1 Time', 2: '2 Times', 3: '3 Times', 4: '4 Times', 5: '5  Times', 6: '6 Times', 7: '7 Times', 8: '8 Times', 9: 'More than 9 times'}
health_data['MARIJUANA_RECODE'] = health_data['MARIJUANA'].map(recodeMar)
freqCount_recodeMar= health_data["MARIJUANA_RECODE"].value_counts(sort=True)
print (freqCount_recodeMar)
print('====================================================================================')

# ages of the students based on grade level
print('Ages of students? - based on grade level')
recodeAge= {7: '12-13 year', 8: '13-14 year', 9: '14-15 year', 10: '15-16 year', 11: '16-17 year', 12: '17-18 year'}
health_data['AGE_RECODE'] = health_data['AGE'].map(recodeAge)
freqCount_recodeAge= health_data["AGE_RECODE"].value_counts(sort=True)
print (freqCount_recodeAge)
print('====================================================================================')

# recode for alochol
print('During the past 12 months, on how many days did you drink alcohol? ')
recodeAlc= {1: 'Every Day', 2: '3-5 days per week', 3: '1-2 days a week', 4: '2-3 days month', 5: 'Once a month', 6: '1-2 days in 12 months'}
health_data['ALCOHOL_RECODE'] = health_data['ALCOHOL'].map(recodeAlc)
freqCount_recodeAlc= health_data["ALCOHOL_RECODE"].value_counts(sort=True)
print (freqCount_recodeAlc)
print('====================================================================================')

# recode for smoking
print('Did you ever smoke?')
recodeSmo= {0: 'NO', 1: 'YES'}
health_data['SMOKE_RECODE'] = health_data['SMOKE'].map(recodeSmo)
freqCount_recodeSmo= health_data["SMOKE_RECODE"].value_counts(sort=True)
print (freqCount_recodeSmo)
print('====================================================================================')

# recode  for someone shot you
print('Someone shot you')
recodeSho= {1: 'YES, Once', 2: 'YES, More than once'}
health_data['SHOT_YOU_RECODE'] = health_data['SHOT_YOU'].map(recodeSho)
freqCount_recodeSho= health_data["SHOT_YOU_RECODE"].value_counts(sort=True)
print (freqCount_recodeSho)
print('====================================================================================')

# recode for you pulled a knife or gun on someone
print('You pulled a knife or gun on someone')
recodeKni= {1: 'YES, Once', 2: 'YES, More than once'}
health_data['YOU_PULLED_KNIFE_GUN_RECODE'] = health_data['YOU_PULLED_KNIFE_GUN'].map(recodeKni)
freqCount_recodeGun= health_data["YOU_PULLED_KNIFE_GUN_RECODE"].value_counts(sort=True)
print (freqCount_recodeGun)
print('====================================================================================')

# recode for you  shot or  stabbed  someone
print('You shot or stabbed someone')
recodeSta= { 1: 'YES, Once', 2: 'YES, More than once'}
health_data['YOU_SHOT_STAB_RECODE'] = health_data['YOU_SHOT_STAB'].map(recodeSta)
freqCount_recodeSta= health_data["YOU_SHOT_STAB_RECODE"].value_counts(sort=True)
print (freqCount_recodeSta)
print('====================================================================================')

# recode for suicide attempts
print('During the past 12 months, how many times did you actually attempt suicide?')
recodeSui= {1: '1 Time', 2: '2 or 3 Times', 3: '4 or 5 Times', 4: '6 or more times'}
health_data['YOU_ATTEMPT_SUICIDE_RECODE'] = health_data['YOU_ATTEMPT_SUICIDE'].map(recodeSui)
freqCount_recodeSui= health_data["YOU_ATTEMPT_SUICIDE_RECODE"].value_counts(sort=True)
print (freqCount_recodeSui)
print('====================================================================================')

# recode for cocaine
print('During the past 30 days, how many times did you use cocaine?')
cocaineRecode= {1: '1 Time'   , 2: '2 Times', 3: '3 Times', 4: '4 Times', 5: '5 Times', 10: '7 - 10 Times',
          14: '11 - 14 Times', 30: '15 - 30 Times', 33: '33 and more Times'}
health_data['COCAINE_RECODE'] = health_data['COCAINE'].map(cocaineRecode)
freqCount_cocaineRecode= health_data["COCAINE_RECODE"].value_counts(sort=True)
print (freqCount_cocaineRecode)
print('====================================================================================')
print('*** NEW DESCRIPTIVE STATISTICS ***')
print('====================================================================================')

print('Stats MARIJUANA')
descMar = health_data['MARIJUANA_RECODE'].describe()
print(descMar)
print('====================================================================================')

print('Stats AGE')
ageDesc = health_data['AGE_RECODE'].describe()
print(ageDesc)
print('====================================================================================')

print('Stats ALCOHOL')
descAlc = health_data['ALCOHOL_RECODE'].describe()
print(descAlc)
print('====================================================================================')

print('Stats SMOKE')
descSmo = health_data['SMOKE_RECODE'].describe()
print(descSmo)
print('====================================================================================')

print('Stats SHOT YOU')
descSho = health_data['SHOT_YOU_RECODE'].describe()
print(descSho)
print('====================================================================================')

print('Stats YOU PULLED KNIFE OR GUN')
descGun = health_data['YOU_PULLED_KNIFE_GUN_RECODE'].describe()
print(descGun)
print('====================================================================================')

print('Stats YOU SHOT OR STABBED SOMEONE')
descSta = health_data['YOU_SHOT_STAB_RECODE'].describe()
print(descSta)
print('====================================================================================')

print('Stats YOU ATTEMPTED  SUICIDE')
descSui = health_data['YOU_ATTEMPT_SUICIDE_RECODE'].describe()
print(descSui)
print('====================================================================================')

print('Stats YOU TOOK COCAINE RECODE')
cocaineDesc = health_data['COCAINE_RECODE'].describe()
print(cocaineDesc)
print('====================================================================================')

# yearly marijuana intake
health_data['TOTAL_MARJ'] = health_data['MARIJUANA'] * 12

# yearly cocaine intake
health_data['TOTAL_COC'] = health_data['COCAINE'] * 12

# yearly general drug intake
health_data['TOTAL_GEN'] = health_data['GENERAL_DRUG'] * 12

# concatination of all drugs
health_data['TOTAL_ALL'] = health_data['GENERAL_DRUG'] + health_data['COCAINE'] + health_data['MARIJUANA'] 

print('*** CUSTOM SPLITS FOR CROSS TAB ***')
#categorise variable based on customised splits using the cut() functions
health_data['COCAINE_USAGE']= pandas.cut(health_data.COCAINE, [0, 1, 2, 3, 4, 5, 10, 14, 30, 33],
           labels=['1 Time','2 Times','3 Times','4 Times','5 Times','7 - 10 Times','11 - 14 Times','15 - 30 Times','33 and more Times'])
cutCocaine = health_data['COCAINE_USAGE'].value_counts(sort=True, dropna=True)
print(cutCocaine)
print('====================================================================================')

print('*** SECONDARY VARIABLE TOTALS ***')
print('====================================================================================')

print('COMBINED TOTAL DRUGS')
# all drugs concatinated 
health_data['TOTAL_DRUG'] = health_data['MARIJUANA'] + health_data['COCAINE'] + health_data['GENERAL_DRUG']

print('TOTAL COUNT MARIJUANA')
# count records
marijuanaCount = health_data['MARIJUANA_RECODE'].count()
#  display the 30 day count
print("30 Day Marijuana count: "+ str(marijuanaCount))
# calculate the  yearly marijuana intake by multiplying by 12 months
yearlyMarijuana = marijuanaCount * 12
# display estimated yearly intake
print('Estimated yearly Marijuana count: ' + str(yearlyMarijuana))
print('====================================================================================')

print('TOTAL COUNT SUICIDE')
# get total count of suicides
suicideCount = health_data['YOU_ATTEMPT_SUICIDE_RECODE'].count()
# display total count of suiciees
print("Yearly suicide count: "+ str(suicideCount))

print('====================================================================================')

print('*** CROSS TAB GENERAL DRUG USAGE ***')
# cross tab for general drugs versus suicide
print(pandas.crosstab(health_data['YOU_ATTEMPT_SUICIDE_RECODE'],health_data['GENERAL_DRUG']))
print('====================================================================================')

print('*** CROSS TAB COCAINE USAGE ***')
#  cross tab for cociane usage
print(pandas.crosstab(health_data['COCAINE_USAGE'],health_data['COCAINE']))
print('====================================================================================')

print('*** CROSS TAB COCAINE AND SUICIDE ***')
# cross tab for cociane usage and suicide numberic cocaine
print(pandas.crosstab(health_data['YOU_ATTEMPT_SUICIDE'],health_data['COCAINE']))
print('====================================================================================')

print('*** CROSS TAB COCAINE AND SUICIDE ***')
# cross tab for cociane usage and suicide categorical cocaine
print(pandas.crosstab(health_data['YOU_ATTEMPT_SUICIDE_RECODE'],health_data['COCAINE_RECODE']))
print('====================================================================================')

print('*** CROSS TAB MARIJUANA AND SUICIDE ***')
# cross tab for marijuana usage and suicide
print(pandas.crosstab(health_data['YOU_ATTEMPT_SUICIDE_RECODE'],health_data['MARIJUANA']))
print('====================================================================================')

print('*** CROSS TAB YOU SHOT OR STABBED SOMEONE AND TRIED TO COMMIT SUICIDE ***')
# cross tab for you shot or stabbed someone and  then attempted suicide
print(pandas.crosstab(health_data['YOU_SHOT_STAB_RECODE'],health_data['YOU_ATTEMPT_SUICIDE_RECODE']))
print('====================================================================================')

print('*** CROSS TAB SOMEONE SHOT YOU AND THEN YOU TRIED TO COMMIT SUICIDE ***')
# cross tab for someone shot you and you then attempted suicide
print(pandas.crosstab(health_data['SHOT_YOU_RECODE'],health_data['YOU_ATTEMPT_SUICIDE_RECODE']))
print('====================================================================================')

print('*** CROSS TAB YOU HAD ALCOHOL AND THEN YOU TRIED TO COMMIT SUICIDE ***')
# cross tab for alcohol consumption and suicide
print(pandas.crosstab(health_data['ALCOHOL_RECODE'],health_data['YOU_ATTEMPT_SUICIDE_RECODE']))
print('====================================================================================')

print('*** CROSS TAB YOU HAD MARIJUANA AND THEN SHOT OR STABBED SOMEONE ***')
# cross tab for marijuana usage and then violance as a result, you shot or stabbed someone
print(pandas.crosstab(health_data['MARIJUANA'],health_data['YOU_SHOT_STAB_RECODE']))
print('====================================================================================')

print('*** QUARTILES ***')
#quartile split qcut function into 4 groups
print('AGE - 4 Categories - quartiles')
health_data['QUARTILE_AGE'] = pandas.qcut(health_data.AGE, 4, labels=['25%tile','50%tile','75%tile','100%tile'])
quartileAge= health_data['QUARTILE_AGE'].value_counts(sort=False, dropna=True)
print(quartileAge)

print('*** MARIJUANA - 4 Categories - quartiles ***')
health_data['MARIJUANA_GROUP'] = pandas.qcut(health_data.MARIJUANA, 5, duplicates='drop')
marijuanaQCUT= health_data['MARIJUANA_GROUP'].value_counts(sort=False, dropna=True)
print(marijuanaQCUT)
print('====================================================================================')

print('*** INITIAL VARIABLE CHARTING ***')
## scatter plot
print('*** SCATTER PLOT ***')
# set size of the figure
seaborn.set(rc={'figure.figsize':(12,6)})
# set color of the figure
seaborn.set(rc={'axes.facecolor':'lightgray', 'figure.facecolor':'lightblue'})
# plot variables numeric
scatDrugs = seaborn.regplot(x="MARIJUANA", y="TOTAL_DRUG", data=health_data)
# label x
plt.xlabel('MARIJUANA')
# label y
plt.ylabel('ALL DRUGS')
# title
plt.title('Scatterplot for the Association between Marijuana as a gateway to other drugs')

print('*** BAR CHART FOR SUICIDE ATTEMPTS BASED ON COCAINE USEAGE ***')
# bivariate strip chart
seaborn.catplot(x='YOU_ATTEMPT_SUICIDE_RECODE',y='COCAINE',data=health_data,kind='strip',ci=None, aspect=2)
plt.xlabel('Suicide')
plt.ylabel('Cocaine')
print('====================================================================================')

print('*** INITIAL UNIVARIATE GRAPHING - COCAINE ***')

health_data['COCAINE_RECODE'] = health_data['COCAINE_RECODE'].astype('category')
seaborn.countplot(x='COCAINE_RECODE', y=None, data=health_data)
plt.title('During the past 30 days, how many times did you use cocaine?')
plt.xlabel('Cocaine use')
plt.ylabel('Counts')
print('====================================================================================')

print('*** BAR CHART FOR AGE BASED ON COCAINE USEAGE ***')

#bivariate bar graph quartile aged and total drugs taken
seaborn.catplot(x='QUARTILE_AGE',y='TOTAL_DRUG',data=health_data,kind='bar',ci=None, aspect=2)
plt.title('Age peak for drug usage through cocaine')
plt.xlabel('AGE QUARTILE')
plt.ylabel('DRUG COUNT')
# print the amount of 12-13 years who have taken cocaine
coke_age = health_data[(health_data['AGE_RECODE']=='12-13 year') & (health_data['COCAINE']>0)]
print(len(coke_age))
print('====================================================================================')

print('*** BAR CHART FOR AGE BASED ON COCAINE USEAGE ***')

#bivariate point graph that shows age with cocaine
seaborn.catplot(x='AGE_RECODE',y='COCAINE',data=health_data,kind='point',ci=None, aspect=2)
plt.title('Age peak for drug usage through cocaine')
plt.xlabel('AGE')
plt.ylabel('COCAINE COUNT')
print('====================================================================================')

print('*** BAR CHART FOR AGE BASED ON MARIJUANA USEAGE ***')
#bivariate point graph that shows age with marijuana
seaborn.catplot(x='AGE_RECODE',y='MARIJUANA',data=health_data,kind='point',ci=None, aspect=2)
plt.title('Age peak for drug usage through marijuana')
plt.xlabel('AGE')
plt.ylabel('MARIJUANA COUNT')
# print the amount of 12-13 years who have taken marijuana
mar_age = health_data[(health_data['AGE_RECODE']=='12-13 year') & (health_data['MARIJUANA']>0)]
print(len(mar_age))
print('====================================================================================')

print('*** BAR CHART FOR AGE BASED ON ALCOHOL USEAGE ***')
#bivariate point graph that shows age with alcohol
seaborn.catplot(x='AGE_RECODE',y='ALCOHOL',data=health_data,kind='point',ci=None, aspect=2)
plt.title('Age peak for ALCOHOL')
plt.xlabel('AGE')
plt.ylabel('ALCOHOL COUNT')
# print the amount of 12-13 years who have taken alcohol
alco_age = health_data[(health_data['AGE_RECODE']=='12-13 year') & (health_data['ALCOHOL']>0)]
print(len(alco_age))
print('====================================================================================')

print('*** BAR CHART FOR AGE BASED ON GENERAL DRUGS ***')
#bivariate point graph that shows age with general drugs
seaborn.catplot(x='AGE_RECODE',y='GENERAL_DRUG',data=health_data,kind='point',ci=None, aspect=2)
plt.title('Age peak for GENERAL DRUGS')
plt.xlabel('AGE')
plt.ylabel('DRUG COUNT')
# print the amount of 12-13 years who have taken general drugs
gen_age = health_data[(health_data['AGE_RECODE']=='12-13 year') & (health_data['GENERAL_DRUG']>0)]
print(len(gen_age))
print('====================================================================================')

print('*** BAR CHART FOR AGE BASED ON SUICIDE ATTEMPTS ***')
#bivariate point graph that shows age with suicide
seaborn.catplot(x='AGE_RECODE',y='YOU_ATTEMPT_SUICIDE',data=health_data,kind='point',ci=None, aspect=2)
plt.title('Age peak for SUICIDE')
plt.xlabel('AGE')
plt.ylabel('SUICIDE ATTEMPTS')
print('====================================================================================')

print('*** BAR CHART FOR AGE BASED ON SMOKING ***')
#bivariate point graph that shows age with smoking
seaborn.catplot(x='AGE_RECODE',y='SMOKE',data=health_data,kind='point',ci=None, aspect=3)
plt.title('Age peak for smoking')
plt.xlabel('AGE')
plt.ylabel('SMOKE COUNT')
print('====================================================================================')

print('*** NESTED GROUPING BAR CHART ***')
# chart showing suicide with cocaine and add age as HUE
seaborn.barplot(x="YOU_ATTEMPT_SUICIDE_RECODE", y="COCAINE", hue="AGE_RECODE", data=health_data)
plt.title('Sucicide attempts based on AGE and Cocaine count usage')
plt.xlabel('Sucicide attempts')
plt.ylabel('Cocaine count')
print('====================================================================================')

print('*** MULTI DIMENSION SUBPLOT CHART DISPLAY ****')
# set size of the figure
seaborn.set(rc={'figure.figsize':(35,20)})
# set color of the figure
seaborn.set(rc={'axes.facecolor':'lightgray', 'figure.facecolor':'lightblue'})
## instansiate a dataframe
df=pandas.DataFrame()
# dataframe label and variable
df['Someone shot you']=health_data['SHOT_YOU_RECODE']

df['You pulled a knife or gun on someone']=health_data['YOU_PULLED_KNIFE_GUN_RECODE']

df['You shot or stabbed someone']=health_data['YOU_SHOT_STAB_RECODE']

df['During the past 12 months, how many times did you actually attempt suicide?']=health_data['YOU_ATTEMPT_SUICIDE_RECODE']

df['During the past 30 days, how many times did you use cocaine?']=health_data['COCAINE_RECODE']

df['Have you ever tried cigarette smoking, even just 1 or 2 puffs?']=health_data['SMOKE_RECODE']

df['During the past 12 months, on how many days did you drink alcohol?']=health_data['ALCOHOL_RECODE']

df['What age are you?']=health_data['AGE_RECODE']

df['During the past 30 days, how many times did you use marijuana?']=health_data['MARIJUANA']

df['During the past 30 days, how many times did you use LSD, PCP, ecstasy, mushrooms, speed, ice, heroin, or pills?']=health_data['GENERAL_DRUG_RECODE']

# create subplots
fig, ax = plt.subplots(3,3)
# plot as per xy co-ordinates
seaborn.countplot(df['Someone shot you'], ax=ax[0,0])

seaborn.countplot(df['You pulled a knife or gun on someone'], ax=ax[0,1])

seaborn.countplot(df['You shot or stabbed someone'], ax=ax[0,2])

seaborn.countplot(df['During the past 12 months, how many times did you actually attempt suicide?'], ax=ax[1,0])

seaborn.countplot(df['During the past 30 days, how many times did you use cocaine?'], ax=ax[1,1])

seaborn.countplot(df['Have you ever tried cigarette smoking, even just 1 or 2 puffs?'], ax=ax[1,2])

seaborn.countplot(df['During the past 12 months, on how many days did you drink alcohol?'], ax=ax[2,0])

seaborn.countplot(df['What age are you?'], ax=ax[2,1])

seaborn.countplot(df['During the past 30 days, how many times did you use LSD, PCP, ecstasy, mushrooms, speed, ice, heroin, or pills?'], ax=ax[2,2])

# show the figure
fig.show()
print('====================================================================================')
seaborn.catplot(x='AGE_RECODE', y='YOU_ATTEMPT_SUICIDE', data=health_data,kind='box',height=4, aspect=1.5)
print('====================================================================================')


print('*** COCAINE USAGE ONCE RECORDS ****')
# refine data to the columns i need
subset_test = health_data[(health_data['COCAINE_RECODE']=='1 Time') & (health_data['YOU_ATTEMPT_SUICIDE_RECODE']=='1 Time')]
print (len(subset_test))
cocaineSub= subset_test['COCAINE_RECODE'].value_counts(sort=True)
print(cocaineSub)
print('====================================================================================')

print('*** IDENTIFIED SINGLE INSTANCE USER ****')
#98715983 # find users that took and tried suicide once and display in a dist table
sub2 = subset_test[['IDENTIFIER', 'COCAINE_RECODE', 'YOU_ATTEMPT_SUICIDE_RECODE' ]]
a = sub2.head(n=100)
print(a)
print('====================================================================================')

print('*** ONE RECORD RELATIONSHIP ***')
# refine to a specific user based on their ID and discover which of the drugs and/or varibables that 
# might be of interest
subset_test = health_data[(health_data['IDENTIFIER']==98715983)]
print (len(subset_test))
sub3 = subset_test[['IDENTIFIER', 'COCAINE_RECODE', 'YOU_ATTEMPT_SUICIDE_RECODE', 'SHOT_YOU_RECODE', 'YOU_SHOT_STAB_RECODE', 'YOU_PULLED_KNIFE_GUN_RECODE', 'SMOKE_RECODE', 'ALCOHOL_RECODE', 'AGE_RECODE', 'MARIJUANA', 'GENERAL_DRUG' ]]
a = sub3.head(n=100)
print(a)
print('====================================================================================')

print('*** ALL USERS THAT TOOK COCAINE ONCE WITH AGE ***')
# refine users to ones that took cocaine once
subset_test_coke = health_data[(health_data['COCAINE_RECODE']=='1 Time')]
print (len(subset_test_coke))
# now display these users in a dist table versus age
sub4 = subset_test_coke[['IDENTIFIER','COCAINE_RECODE', 'AGE_RECODE']]
a = sub4.head(n=100)
print(a)
print('====================================================================================')

print('*** ALL USERS THAT ATTEMPT SUICIDE ONCE WITH AGE ***')
# refine users to ones that tried to commit suicide once
subset_test_suicide = health_data[(health_data['YOU_ATTEMPT_SUICIDE_RECODE']=='1 Time')]
print (len(subset_test_suicide))
# now display these users in a dist table verus age
sub4 = subset_test_suicide[['IDENTIFIER','YOU_ATTEMPT_SUICIDE_RECODE', 'AGE_RECODE']]
a = sub4.head(n=100)
print(a)
print('====================================================================================')

print('**** COCAINE USAGE TWICE ***')
# refine users to ones that took cocaine twice
subset_test_twice = health_data[(health_data['COCAINE_RECODE']=='2 Times')]
print (len(subset_test_twice))
# now display these users in a dist tabe
sub4 = subset_test_twice[['IDENTIFIER','COCAINE_RECODE']]
a = sub4.head(n=100)
print(a)
print('====================================================================================')

# prove that cocaine causes more suicide attempts with just one try
print('*** IN A 12 MONTH PERIOD, HOW MANY PEOPLE TOOK COCAINE ONCE AND ATTEMPTED SUICIDE 1, 2, 3 AND 4? ***')
subset_suicide1 = health_data[(health_data['COCAINE']==1) & (health_data['YOU_ATTEMPT_SUICIDE']==1)]
print(len(subset_suicide1))
print('====================================================================================')

subset_suicide2 = health_data[(health_data['COCAINE']==1) & (health_data['YOU_ATTEMPT_SUICIDE']==2)]
print(len(subset_suicide2))
print('====================================================================================')

subset_suicide3 = health_data[(health_data['COCAINE']==1) & (health_data['YOU_ATTEMPT_SUICIDE']==3)]
print(len(subset_suicide3))
print('====================================================================================')

subset_suicide4 = health_data[(health_data['COCAINE']==1) & (health_data['YOU_ATTEMPT_SUICIDE']==4)]
print(len(subset_suicide4))

print('====================================================================================')

# prove that cocaine causes more suicide attempts with more usage of cocaine
print('*** IN A 12 MONTH PERIOD, HOW MANY PEOPLE TOOK COCAINE MORE THAN ONCE, 2, 3 ... AND ATTEMPTED SUICIDE ONCE? ***')
subset_cocaine1 = health_data[(health_data['COCAINE']==1) & (health_data['YOU_ATTEMPT_SUICIDE']==1)]
print(len(subset_cocaine1))
print('====================================================================================')

subset_cocaine2 = health_data[(health_data['COCAINE']==2) & (health_data['YOU_ATTEMPT_SUICIDE']==1)]
print(len(subset_cocaine2))
print('====================================================================================')

subset_cocaine3 = health_data[(health_data['COCAINE']==3) & (health_data['YOU_ATTEMPT_SUICIDE']==1)]
print(len(subset_cocaine3))
print('====================================================================================')

subset_cocaine4 = health_data[(health_data['COCAINE']==4) & (health_data['YOU_ATTEMPT_SUICIDE']==1)]
print(len(subset_cocaine4))
print('====================================================================================')

subset_cocaine5 = health_data[(health_data['COCAINE']==5) & (health_data['YOU_ATTEMPT_SUICIDE']==1)]
print(len(subset_cocaine5))
print('====================================================================================')

subset_cocaine6 = health_data[(health_data['COCAINE']==10) & (health_data['YOU_ATTEMPT_SUICIDE']==1)]
print(len(subset_cocaine6))
print('====================================================================================')

subset_cocaine7 = health_data[(health_data['COCAINE']==14) & (health_data['YOU_ATTEMPT_SUICIDE']==1)]
print(len(subset_cocaine7))
print('====================================================================================')

subset_cocaine8 = health_data[(health_data['COCAINE']==30) & (health_data['YOU_ATTEMPT_SUICIDE']==1)]
print(len(subset_cocaine8))
print('====================================================================================')

subset_cocaine9 = health_data[(health_data['COCAINE']==33) & (health_data['YOU_ATTEMPT_SUICIDE']==1)]
print(len(subset_cocaine9))

print('====================================================================================')

# count all instances of recoded suicide

print('Count of all suicides in the year')
# sum of recoded freq's
count_of_allSuicides = sum(recodeSui)
print(count_of_allSuicides)
print('====================================================================================')

# average suicide per month
print('Average number of suicides per month')
# divide all suicides in the year by 12 months
average_suicide_monthly = count_of_allSuicides / 12
print(average_suicide_monthly)
print('====================================================================================')

# average suicide per weekly
print('Average number of suicides per week')
# divide suicides into weeks
average_suicide_weekly = count_of_allSuicides / 52
print(average_suicide_weekly)
print('====================================================================================')


#create new secondary variable to hold average suicides per week
print('Secondary Variable - Average weekly suicides')
health_data['AVERAGE_WEEKLY_SUICIDE'] = average_suicide_weekly
health_data['AVERAGE_WEEKLY_SUICIDE'] = pandas.to_numeric(health_data['AVERAGE_WEEKLY_SUICIDE'])
freqCount_secondVarWeek= health_data["AVERAGE_WEEKLY_SUICIDE"].value_counts(sort=True)
print (freqCount_secondVarWeek)
print('====================================================================================')

#create new secondary variable to hold average suicides per month
print('Secondary Variable - Average monthly suicides')
health_data['AVERAGE_MONTHLY_SUICIDE'] = average_suicide_monthly
health_data['AVERAGE_MONTHLY_SUICIDE'] = pandas.to_numeric(health_data['AVERAGE_MONTHLY_SUICIDE'])
freqCount_secondVarMonth= health_data["AVERAGE_MONTHLY_SUICIDE"].value_counts(sort=True)
print (freqCount_secondVarMonth)
print('====================================================================================')

# descriptive statistics - per specific function
print('mean')
mean1 = health_data['MARIJUANA'].mean()
print(mean1)

print('std deviation')
std1 = health_data['MARIJUANA'].std()
print(std1)

print('min')
min1 = health_data['MARIJUANA'].min()
print(min1)

print('max')
max1 = health_data['MARIJUANA'].max()
print(max1)

print('median')
median1 = health_data['MARIJUANA'].median()
print(median1)

print('mode')
mode1 = health_data['MARIJUANA'].mode()
print(mode1)
print('====================================================================================')

print('*** HYPOTHESIS TESTING ***')

# create copy of the data
sub1 = health_data.copy()

# using OLS function for calculating the F-statistic and associated p value
model1 = smf.ols(formula='TOTAL_DRUG ~ C(AGE_RECODE)', data=health_data)
results1 = model1.fit()
print(results1.summary())

print('means for GENERAL DRUG USE by major 13-14 year olds')
#subset of only two variables.
sub2=health_data[['GENERAL_DRUG', 'AGE_RECODE']].dropna()
#Sample size
print(len(sub2))

#Create a new subset which only contains required variables for analysis.
sub3 = health_data[['GENERAL_DRUG', 'MARIJUANA']].dropna()

# using OLS function for calculating the F-statistic and associated p value
model2 = smf.ols(formula='GENERAL_DRUG ~ (MARIJUANA)', data=sub3).fit()
print (model2.summary())

# Multi Comparison
mc1 = multi.MultiComparison(sub3['GENERAL_DRUG'], sub3['MARIJUANA'])
res1 = mc1.tukeyhsd()
print(res1.summary())

