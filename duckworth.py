import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.linear_model import LinearRegression
import pickle
style.use('ggplot')

df = pd.read_csv('Ball_by_Ball.csv')
totalMatches = 577 * 2
df.drop(['Innings_Id', 'Team_Batting_Id', 'Team_Bowling_Id', 'Striker_Id', 'Striker_Batting_Position', 'Non_Striker_Id', 'Bowler_Id', 'Extra_Type', 'Dissimal_Type', 'Fielder_Id'],1,inplace=True)


#Replacing Null and string values to int in each columns of the dataframe
df.Batsman_Scored.replace({
    "Do_nothing" : "0",
    " " : "0"
    }, inplace = True)

df.Extra_Runs.replace({
    " " : "0"
    }, inplace = True)

df.Player_dissimal_Id.replace({
        " " : "0"
    }, inplace=True)

columns = ['Over', 'Runs', 'Wickets']

dataFrame = pd.DataFrame(columns=columns)
totalRuns = 0
#print(df)
#Logic For New Data Frame
initRow = 1
runs = 0
wickets = 0
for index, row in df.iterrows():
   totalRuns += int(row['Batsman_Scored']) + int(row['Extra_Runs'])
   if row['Over_Id'] == initRow:
      runs += int(row['Batsman_Scored']) + int(row['Extra_Runs'])
      if int(row['Player_dissimal_Id']) != 0 :
          wickets += 1
   else:
        dataFrame = dataFrame.append({'Over' : initRow, 'Runs' : runs, 'Wickets' : wickets}, ignore_index=True)
        initRow = int(row['Over_Id'])
        if initRow == 1 :
           runs = int(row['Batsman_Scored']) + int(row['Extra_Runs'])
           if int(row['Player_dissimal_Id']) != 0 :
             wickets = 1
           else :
             wickets = 0
        else :
           runs += int(row['Batsman_Scored']) + int(row['Extra_Runs'])
           if int(row['Player_dissimal_Id']) != 0 :
             wickets += 1

totalRuns = totalRuns/totalMatches
dataFrame = dataFrame.append({'Over' : initRow, 'Runs' : runs, 'Wickets' : wickets}, ignore_index=True)   

#print(dataFrame)


#dataFrame.to_csv('Final1.csv', sep='\t', encoding='utf-8')  
dataFrame = dataFrame.sort_values(by=['Over'], ascending= [True])

#print(dataFrame)

'''
columns2 = ['Over', 'Avg_Runs', 'Avg_Wickets']
dataFrame2 = pd.DataFrame(columns = columns2)

#Logic for Second Data Frame

initOver = 1
initRuns = 0
initWickets = 0
count = 0
for index, row in dataFrame.iterrows():
   if row['Over'] == initOver:
      initRuns += int(row['Runs'])
      initWickets += int(row['Wickets'])
      count += 1
   else:
      initRuns = initRuns/count
      initWickets = initWickets/count
      count = 0
      dataFrame2 = dataFrame2.append({'Over' : initOver, 'Avg_Runs' : initRuns, 'Avg_Wickets' : initWickets}, ignore_index=True)
      initRuns = int(row['Runs'])
      initWickets = int(row['Wickets'])
      initOver += 1
initRuns = initRuns/count
initWickets = initWickets/count
dataFrame2 = dataFrame2.append({'Over' : initOver, 'Avg_Runs' : initRuns, 'Avg_Wickets' : initWickets}, ignore_index=True)
'''
#regression now yuhoo :P

forecast_col= 'Runs'

columns3 = ['Total_Runs', 'Total_Wickets']



initOver = 1
initRuns = 0
initWickets = 0
count = 0

dataFrame3 = pd.DataFrame(columns = columns3)
initOver = 1


#inittialising matrix
c, r = 11, 21
duckLewis = [[0 for x in range(c)] for y in range(r)]

for j, row in dataFrame.iterrows():
   if int(row['Over']) == initOver :
      dataFrame3 = dataFrame3.append({'Total_Runs' : int(row['Runs']), 'Total_Wickets' : int(row['Wickets'])}, ignore_index=True)
   else :
      X = np.array(dataFrame3.drop(['Total_Runs'],1))
      y = np.array(dataFrame3['Total_Runs'])
      clf = LinearRegression(n_jobs = -1)
      clf.fit(X, y)
      for i in range(0,11):
         duckLewis[initOver][i] = clf.predict(i)/totalRuns
         duckLewis[initOver][i] *= 100
        # duckLewis[initOver][i] -= 100
      initOver += 1
      dataFrame3 = pd.DataFrame(columns = columns3)
      dataFram3 = dataFrame3.append({'Total_Runs' : int(row['Runs']), 'Total_Wickets' : int(row['Wickets'])}, ignore_index=True)

X = np.array(dataFrame3.drop(['Total_Runs'],1))
y = np.array(dataFrame3['Total_Runs'])
clf = LinearRegression(n_jobs = -1)
clf.fit(X, y)
     
for i in range(0,11) :
   duckLewis[initOver][i] = clf.predict(i)/totalRuns
   duckLewis[initOver][i] *= 100
   
      

for i in range(1,21) :
   for j in range(0,11) :
      print (duckLewis[i][j], end='')
   print()
   

'''
dataFrame['Runs'].plot()
dataFrame['Wickets'].plot()
plt.legend(loc=4)
plt.xlabel(['Over', 'Wickets'])
plt.ylabel(['Runs'])
plt.show()
'''
   

      
   

























