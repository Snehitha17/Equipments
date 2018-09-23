# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
import cufflinks as cf
cf.go_offline()
from scipy.stats import mode
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense

# Reading the datasets
eqp1 = pd.read_csv('equipments.csv')
eqp2 = pd.read_csv('equipments1.csv')
eqp3 = pd.read_csv('equipments2.csv')

equipments = [eqp1, eqp2, eqp3]
eqp = pd.concat(equipments)
eqp.head()

eqp.shape

eqp.columns

eqp.info()
eqp.describe()

# Plotting the graphs

temp =eqp["PacketDrop_Current_Value"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Current Value')

temp =eqp["PacketDrop_Min_Current_Value"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Minimum Packetdrop Current Value')

temp =eqp["PacketDrop_Severity"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Maximum Packetdrop Current Value')

temp =eqp["PacketDrop_Critical.Threshold"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Maximum Packetdrop Current Value')

temp =eqp["PacketDrop_Warning.Threshold"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Maximum Packetdrop Current Value')

temp = eqp["CRCErrors_Min_Current_Value"].value_counts()
#print("Total number of states : ",len(temp))
trace = go.Bar(
    x = temp.index,
    y = (temp / temp.sum())*100,
)
data = [trace]
layout = go.Layout(
    title = "Minimum Current value of CRC Errors % ",
    xaxis=dict(
        title='Name of type of the error',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of Name of type of the error in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
             )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='Current Values')

temp = eqp["CRCErrors_Max_Current_Value"].value_counts()
#print("Total number of states : ",len(temp))
trace = go.Bar(
    x = temp.index,
    y = (temp / temp.sum())*100,
)
data = [trace]
layout = go.Layout(
    title = "Maximum Current value of CRC Errors % ",
    xaxis=dict(
        title='Name of type of the error',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of Name of type of the error in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
             )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='Current Values')

temp = eqp["CRCErrors_Current_Value"].value_counts()
#print("Total number of states : ",len(temp))
trace = go.Bar(
    x = temp.index,
    y = (temp / temp.sum())*100,
)
data = [trace]
layout = go.Layout(
    title = "Current value of CRC Errors % ",
    xaxis=dict(
        title='Name of type of the error',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of Name of type of the error in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
             )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='Current Values')

temp = eqp["CRCErrors_Severity"].value_counts()
#print("Total number of states : ",len(temp))
trace = go.Bar(
    x = temp.index,
    y = (temp / temp.sum())*100,
)
data = [trace]
layout = go.Layout(
    title = "Severity of CRC Errors % ",
    xaxis=dict(
        title='Name of type of the error',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of Name of type of the error in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
             )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='Current Values')

temp = eqp["SessionUptime_Current_Value"].value_counts()
fig = {
  "data": [
    {
      "values": temp.values,
      "labels": temp.index,
      "domain": {"x": [0, .48]},
      #"name": "Types of sessionuptime",
      #"hoverinfo":"label+percent+name",
      "hole": .7,
      "type": "pie"
    },
    
    ],
  "layout": {
        "title":"Types of sessionuptime",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Session Uptime",
                "x": 0.17,
                "y": 0.5
            }
            
        ]
    }
}
iplot(fig, filename='uptime')

temp1 = eqp["SessionUptime_Min_Current_Value"].value_counts()
temp2 = eqp["SessionUptime_Max_Current_Value"].value_counts()

fig = {
  "data": [
    {
      "values": temp1.values,
      "labels": temp1.index,
      "domain": {"x": [0, .48]},
      "name": "Minimum",
      "hoverinfo":"label+percent+name",
      "hole": .6,
      "type": "pie"
    },
    {
      "values": temp2.values,
      "labels": temp2.index,
      "text":"Maximum",
      "textposition":"inside",
      "domain": {"x": [.52, 1]},
      "name": "Maximum",
      "hoverinfo":"label+percent+name",
      "hole": .6,
      "type": "pie"
    }],
  "layout": {
        "title":"Types of session uptime",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Minimum",
                "x": 0.20,
                "y": 0.5
            },
             {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Maximum",
                "x": 0.8,
                "y": 0.5
            }
        ]
    }
}
iplot(fig, filename='severity')

temp = eqp["SessionUptime_Severity"].value_counts()
fig = {
  "data": [
    {
      "values": temp.values,
      "labels": temp.index,
      "domain": {"x": [0, .48]},
      #"name": "Types of sessionuptime",
      #"hoverinfo":"label+percent+name",
      "hole": .7,
      "type": "pie"
    },
    
    ],
  "layout": {
        "title":"Types of sessionuptime",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Session Uptime",
                "x": 0.17,
                "y": 0.5
            }
            
        ]
    }
}
iplot(fig, filename='uptime')

downlink = eqp[['DownlinkJitter_Current_Value', 'DownlinkJitter_Severity',
       'DownlinkJitter_Warning.Threshold', 'DownlinkJitter_Critical.Threshold',
       'DownlinkJitter_Min_Current_Value', 'DownlinkJitter_Max_Current_Value',
       'DownlinkRSSI_Current_Value', 'DownlinkRSSI_Severity',
       'DownlinkRSSI_Warning.Threshold', 'DownlinkRSSICritical.Threshold',
       'DownlinkRSSI_Min_Current_Value', 'DownlinkRSSI_Max_Current_Value',
       'DownlinkUtilization_Current_Value', 'DownlinkUtilization_Severity',
       'DownlinkUtilization_Min_Current_Value',
       'DownlinkUtilization_Max_Current_Value']]
downlink.head()

downlink.hist(figsize = (30,20))
plt.show()

uplink = eqp[['UplinkJitter_Current_Value','UplinkJitter_Severity', 'UplinkJitter_Warning.Threshold',
       'UplinkJitter_Critical.Threshold', 'UplinkJitter_Min_Current_Value',
       'UplinkJitter_Max_Current_Value', 'UplinkRSSI_Current_Value',
       'UplinkRSSI_Severity', 'UplinkRSSI_Warning.Threshold',
       'UplinkRSSI_Critical.Threshold', 'UplinkRSSI_Min_Current_Value',
       'UplinkRSSI_Max_Current_Value', 'UplinkUtilization_Current_Value',
       'UplinkUtilization_Severity', 'UplinkUtilization_Min_Current_Value',
       'UplinkUtilization_Max_Current_Value']]
uplink.head()

uplink.hist(figsize = (30,20))
plt.show()

temp = eqp["Uptime_Current_Value"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='hist',labels='labels',values='values', title='% of Uptime current value')

temp = eqp["Uptime_Severity"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='hist',labels='labels',values='values', title='% of Uptime Severity')

# checking missing data in training data 
total = eqp.isnull().sum().sort_values(ascending = False)
percent = (eqp.isnull().sum()/eqp.isnull().count()*100).sort_values(ascending = False)
missing_train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_train_data.head(10)

#replace missing value by mode[0]

mode(eqp['UplinkJitter_Max_Current_Value']).mode[0]
var_to_impute = ['UplinkRSSI_Current_Value','Uptime_Min_Current_Value','Uptime_Max_Current_Value','UplinkRSSI_Min_Current_Value',
                 'UplinkRSSI_Max_Current_Value','Uptime_Current_Value','UplinkJitter_Min_Current_Value','UplinkJitter_Max_Current_Value',
                'UplinkJitter_Current_Value']
for var in var_to_impute:
    eqp[var].fillna(mode(eqp[var]).mode[0],inplace = True)
    
from sklearn.preprocessing import LabelEncoder
encoded = eqp.apply(lambda x: LabelEncoder().fit_transform(x) if x.dtype == 'object' else x)
encoded.head()

x = encoded.drop(columns = 'PacketDrop_Current_Value')
y = encoded.PacketDrop_Current_Value
print('X shape:', x.shape)
print('y shape:', y.shape)

# fitting minmaxscaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

x = scaler.fit_transform(x)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)
    
 # Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu', input_dim = 70))

# Adding the second hidden layer
classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 15)
  
