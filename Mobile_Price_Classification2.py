
#Setupp

import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pylab as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import confusion_matrix ,classification_report,precision_score, recall_score ,f1_score, roc_auc_score 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#Import and check the Dataset

mobile_data = pd.read_csv("C:\\Users\\Isuru\\Desktop\\SEM 06\\Applied stat\\Group Project 01\\train.csv")
mobile_data.head()
mobile_data.tail()
mobile_data.info()
mobile_data.isnull()
mobile_data.isnull().sum() #check the missing values

#UNIVARIATE ANALYSIS-----------------

#Battery power

sns.set(rc={'figure.figsize':(5,5)})
ax=sns.displot(data= mobile_data["battery_power"])
plt.title("Histogram of Battery Power",size = 18)
plt.xlabel("Battery Power", size = 16)
plt.ylabel("Frequency",size = 16)
plt.hist(mobile_data["battery_power"] , color = "black")
plt.show() 

#Clock Speed

sns.set(rc={'figure.figsize':(5,5)})
ax=sns.displot(data= mobile_data["clock_speed"])
plt.title("Histogram of Clock Speed",size = 16)
plt.xlabel("Clock Speed", size = 15)
plt.ylabel("Frequency",size = 15)
plt.hist(mobile_data["clock_speed"] , color = "black")
plt.show()

#Front camera - Mega pixels

sns.set(rc={'figure.figsize':(5,5)})
ax=sns.displot(data= mobile_data["fc"],color = "black")
plt.title("Histogram of Front camera-Mega pixels",size = 16)
plt.xlabel("Front camera-Mega pixels", size = 15)
plt.ylabel("Frequency",size = 15)
plt.show()

#Mobile Depth - cm

sns.set(rc={'figure.figsize':(5,5)})
ax=sns.displot(data= mobile_data["m_dep"],color = "black")
plt.title("Histogram of Mobile Depth",size = 16)
plt.xlabel("Mobile Depth in cm", size = 15)
plt.ylabel("Frequency",size = 15)
plt.show()

#Bluetooth(Categorical)

sns.set()
blue_plot=mobile_data['blue'].value_counts().plot(kind='bar',color="black")
plt.xlabel('Bluetooth Availability',size=16)
plt.ylabel('Count',size=16)
plt.title('Bar chart of Bluetooth Availability',size=18)
plt.show() 


#Dual Sim

sns.set()
dual_plot=mobile_data['dual_sim'].value_counts().plot(kind='bar',color="black")
plt.xlabel('Dual Sim Availability',size=16)
plt.ylabel('Count',size=16)
plt.title('Bar chart of Dual Sim Availability',size=18)
plt.show() 

#Battery power boxplot

sns.boxplot(mobile_data['battery_power'],color = "black")
plt.title('Battery Power')
plt.show()


#Bluetooth pie chart

sns.set(rc={"figure.figsize":(3, 12)})
bluetooth = mobile_data['blue'].value_counts()
plt.title('Percentage of Mobiles with Bluetooth', weight='bold')
#labels_blue = ['No Bluetooth', 'Bluetooth']
bluetooth.plot.pie(autopct="%.1f%%")
plt.show()


#Heatmap

plt.figure(figsize=(16,6))
sns.heatmap(mobile_data.corr())
plt.show()

#Relation between price range and battery power

plt.figure(figsize=(12,6))
sns.barplot(x="price_range", y = "battery_power", data = mobile_data)
plt.title('Relation between Price range and Battery power')
plt.show()

#Relation between price range and dual sim

plt.figure(figsize=(12,6))
sns.barplot(x="dual_sim", y = "price_range", data = mobile_data)
plt.title('Relation between Price range and Dual Sim')
plt.show()

#Relation between price range and bluetooth

plt.figure(figsize=(12,6))
sns.barplot(x="blue", y = "price_range", data = mobile_data)
plt.title('Relation between Price range and Bluetooth')
plt.show()

#Relation between price range and clockspeed

plt.figure(figsize=(12,6))
sns.barplot(x="price_range", y = "clock_speed", data = mobile_data)
plt.title('Relation between Price range and Clockspeed')
plt.show()

#Relation between price range and Front camera

plt.figure(figsize=(12,6))
sns.barplot(x="price_range", y = "fc", data = mobile_data)
plt.title('Relation between Price range and Front Camera')
plt.show()

#Relation between price range and Mobile Depth

plt.figure(figsize=(12,6))
sns.barplot(x="price_range", y = "m_dep", data = mobile_data)
plt.title('Relation between Price range and Mobile Depth')
plt.show()

#Price range

sns.set(rc={'figure.figsize':(5,5)})
ax=sns.displot(data= mobile_data["price_range"])
plt.title("Histogram of Price Range",size = 16)
plt.xlabel("Price Range", size = 15)
plt.ylabel("Frequency",size = 15)
plt.hist(mobile_data["price_range"] , color = "black")
plt.show()

#RAM

sns.set(rc={'figure.figsize':(5,5)})
ax=sns.displot(data= mobile_data["ram"])
plt.title("Histogram of RAM",size = 16)
plt.xlabel("ram", size = 15)
plt.ylabel("Frequency",size = 15)
plt.hist(mobile_data["ram"] , color = "black")
plt.show()

#Relation between price range and ram

plt.figure(figsize=(12,6))
sns.barplot(x="price_range", y = "ram", data = mobile_data)
plt.title('Relation between Price range and RAM')
plt.show()

#weight

sns.set(rc={'figure.figsize':(5,5)})
ax=sns.displot(data= mobile_data["mobile_wt"])
plt.title("Histogram of Mobile Weight",size = 16)
plt.xlabel("weight", size = 15)
plt.ylabel("Frequency",size = 15)
plt.hist(mobile_data["mobile_wt"] , color = "red")
plt.show()

#Relation between Price Range and  Pixel Resolution Height

plt.figure(figsize=(12,6))
sns.barplot(x="price_range", y = "px_height", data = mobile_data)
plt.title('Relation between Price range and Pixel Resolution Height')
plt.show()

#Relation between Price Range and  Pixel Resolution width

plt.figure(figsize=(12,6))
sns.barplot(x="price_range", y = "px_width", data = mobile_data)
plt.title('Relation between Price range and Pixel Resolution Width')
plt.show()













