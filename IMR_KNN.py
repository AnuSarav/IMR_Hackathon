
import os
import pandas as pd
import numpy as np

os.chdir("C:/Users/anusa/Downloads/")



TrainRaw = pd.read_csv("training_.csv")
TestRaw = pd.read_csv("test.csv")

# Create Source Column in both Train and Test
TrainRaw['Source'] = "Train"
TestRaw['Source'] = "Test"

# Combine Train and Test
FullRaw = pd.concat([TrainRaw, TestRaw], axis = 0)
FullRaw.shape

FullRaw.isnull().sum()

# Missing Value Impution
#########################

# work_interfere Variable

tempMode= FullRaw['work_interfere'].mode()[0]
FullRaw['work_interfere']= FullRaw['work_interfere'].fillna(tempMode)

# self_employed Variable

tempMode=FullRaw['self_employed'].mode()[0]
FullRaw['self_employed']=FullRaw['self_employed'].fillna(tempMode)


FullRaw.isnull().sum()

# Split of 0s and 1s
#####################

FullRaw.loc[FullRaw['Source']=='Train','treatment'].value_counts()/FullRaw[FullRaw['Source']== 'Train'].shape[0]


FullRaw['treatment']= FullRaw['treatment'].map({'Yes': 1, 'No': 0})


# Remove Customer Id columns
FullRaw.drop(['S.No'], axis = 1, inplace = True)

FullRaw.drop(['Timestamp'], axis = 1, inplace = True)
FullRaw.drop(['state'], axis = 1, inplace = True)
FullRaw.drop(['comments'], axis = 1, inplace = True)

Unique_Value = FullRaw.Gender.unique()

FullRaw['Gender']= FullRaw['Gender'].map({'male':'Male', 'female': 'Female','m':'Male','M':'Male',
                                          'femail':'Female','f':'Female','F':'Female','Woman':'Female',
                                          'woman':'Female','maile':'Male','Mal':'Male','Make':'Male',
                                          'Femake':'Female','Male':'Male','Female':'Female'})
FullRaw['Gender']=FullRaw['Gender'].fillna('Neuter')

FullRaw.Country.unique()

conditions = [
    FullRaw['Country'].str.contains('United States'),
   FullRaw['Country'].str.contains('Canada'),
   FullRaw['Country'].str.contains('United Kingdom'),
   FullRaw['Country'].str.contains('Bulgaria'),
   FullRaw['Country'].str.contains('France'),
   
   FullRaw['Country'].str.contains('Portugal'),
   FullRaw['Country'].str.contains('Netherlands'),
   FullRaw['Country'].str.contains('Switzerland'),
   FullRaw['Country'].str.contains('Poland'),
   FullRaw['Country'].str.contains('Australia'),
   
   FullRaw['Country'].str.contains('Germany'),
   FullRaw['Country'].str.contains('Russia'),
   FullRaw['Country'].str.contains('Mexico'),
   FullRaw['Country'].str.contains('Brazil'),
   FullRaw['Country'].str.contains('Slovenia'),
   
   FullRaw['Country'].str.contains('Costa Rica'),
   FullRaw['Country'].str.contains('Austria'),
   FullRaw['Country'].str.contains('Ireland'),
   FullRaw['Country'].str.contains('Poland'),
   FullRaw['Country'].str.contains('India'),
   
   FullRaw['Country'].str.contains('South Africa'),
   FullRaw['Country'].str.contains('Italy'),
   FullRaw['Country'].str.contains('Sweden'),
   FullRaw['Country'].str.contains('Colombia'),
   FullRaw['Country'].str.contains('Latvia'),
   
   FullRaw['Country'].str.contains('Romania'),
   FullRaw['Country'].str.contains('Belgium'),
   FullRaw['Country'].str.contains('New Zealand'),
   FullRaw['Country'].str.contains('Zimbabwe'),
   FullRaw['Country'].str.contains('Spain'),
   
   FullRaw['Country'].str.contains('Finland'),
   FullRaw['Country'].str.contains('Uruguay'),
   FullRaw['Country'].str.contains('Israel'),
   FullRaw['Country'].str.contains('Bosnia and Herzegovina'),
   FullRaw['Country'].str.contains('Hungary'),
   
   FullRaw['Country'].str.contains('Singapore'),
   FullRaw['Country'].str.contains('Japan'),
   FullRaw['Country'].str.contains('Nigeria'),
   FullRaw['Country'].str.contains('Croatia'),
   FullRaw['Country'].str.contains('Norway'),
   
   FullRaw['Country'].str.contains('Thailand'),
   FullRaw['Country'].str.contains('Denmark'),
   FullRaw['Country'].str.contains('Bahamas,The'),
 FullRaw['Country'].str.contains('Greece'),
   FullRaw['Country'].str.contains('Maldova'),
   
   FullRaw['Country'].str.contains('Georgia'),
   FullRaw['Country'].str.contains('China'),
   FullRaw['Country'].str.contains('Czech Republic'),
   FullRaw['Country'].str.contains('Philippines')]
     
choices = ['North America', 'North America', 'Europe', 'Europe','Europe', 
           
           'Europe', 'Europe', 'Europe', 'Europe', 'Australia',
           
           'Europe', 'Asia','North America', 'South America', 'Europe', 
           
           'South America','Europe', 'Europe','Europe', 'Asia', 'Africa',
           
           'Europe', 'Europe', 'South America', 'Europe','Europe', 
           
           'Europe', 'Australia', 'Africa', 'Europe', 'Europe',
           
           'South America', 'Asia', 'Europe','Europe','Asia', 
           
           'Asia','Africa', 'Europe', 'Europe','Asia',
           
           'Europe', 'North America', 'Europe','Europe', 'Europe', 
           
           'Asia', 'Europe','Asia' ]

FullRaw['Continent'] = np.select(conditions, choices, default='Other')

FullRaw.drop(['Country'], axis = 1, inplace = True)

#Outliers
###########
"""
#Capping the outlier rows with Percentiles
upper_lim = FullRaw['Age'].quantile(.95)
lower_lim = FullRaw['Age'].quantile(.05)
FullRaw.loc[(FullRaw['Age'] > upper_lim),(FullRaw['Age'])] = upper_lim
FullRaw.loc[(FullRaw['Age'] < lower_lim),(FullRaw['Age'])] = lower_lim
"""
"""
#Dropping the outlier rows with Percentiles
"""
#upper_lim = FullRaw['Age'].quantile(.98)
#lower_lim = FullRaw['Age'].quantile(.02)

FullRaw = FullRaw[(FullRaw['Age'] < 100)]


###################
# Sampling
##################

FullRaw2= pd.get_dummies(FullRaw,drop_first= True)
FullRaw2.shape

Train2 = FullRaw2[FullRaw2['Source_Train'] == 1].drop(['Source_Train'], axis = 1).copy()
FinalTest = FullRaw2[FullRaw2['Source_Train'] == 0].drop(['Source_Train'], axis = 1).copy()

Train2.shape
FinalTest.shape


 #Divide Train further into Train and Test by random sampling
from sklearn.model_selection import train_test_split
Train, Test = train_test_split(Train2, train_size=0.8, random_state = 150)

from sklearn.model_selection import train_test_split

Train, Test = train_test_split(Train2, train_size=0.8, random_state = 150)



Train_X = Train.drop(['treatment'], axis = 1).copy()
Train_Y = Train['treatment'].copy()
Test_X = Test.drop(['treatment'], axis = 1).copy()
Test_Y = Test['treatment'].copy()
FinalTest_X = FinalTest.drop(['treatment'], axis = 1).copy()
FinalTest_Y = FinalTest['treatment'].copy()
Train_X.shape
Test_X.shape
FinalTest_X.shape

###################
# Standardization
###################

from sklearn.preprocessing import StandardScaler

Train_Scaling = StandardScaler().fit(Train_X) # Train_Scaling contains means, std_dev of training dataset
Train_X_Std = Train_Scaling.transform(Train_X) # This step standardizes the train data
Test_X_Std  = Train_Scaling.transform(Test_X) # This step standardizes the test data
FinalTest_X_Std  = Train_Scaling.transform(FinalTest_X) # This step standardizes the test data


# Add the column names to Train_X_Std, Test_X_Std
Train_X_Std = pd.DataFrame(Train_X_Std, columns = Train_X.columns)
Test_X_Std = pd.DataFrame(Test_X_Std, columns = Test_X.columns)
FinalTest_X_Std = pd.DataFrame(FinalTest_X_Std, columns = FinalTest_X.columns)


###################
# Model building
###################

from sklearn.neighbors import KNeighborsClassifier
M1 = KNeighborsClassifier(n_neighbors=3).fit(Train_X_Std, Train_Y)

###################
# Model prediction
###################

# Class Prediction
Test_Pred = M1.predict(Test_X_Std)

# Probability Prediction: Probability is just the fraction of 0s and 1s in the K nearest neighbours
Test_Prob = M1.predict_proba(Test_X_Std)
Test_Prob_Df = pd.DataFrame(Test_Prob)
Test_Prob_Df['Class'] = Test_Pred

###################
# Model evaluation
###################

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

Confusion_Mat = confusion_matrix(Test_Y, Test_Pred)
Confusion_Mat

sum(np.diagonal(Confusion_Mat))/Test_X.shape[0]*100 # 60.9
precision_score(Test_Y, Test_Pred) # 60
recall_score(Test_Y, Test_Pred) # 51
f1_score(Test_Y, Test_Pred) # 57
Confusion_Mat[0][1]/sum(Confusion_Mat[0]) # 34


###################
# Grid Search CV
###################
             
from sklearn.model_selection import GridSearchCV

myNN = range(1,14,2) # list(range(1,14,2))
myP = range(1,4,1) # list(range(1,4,1))
my_param_grid = {'n_neighbors': myNN, 'p': myP} # param_grid is a dictionary

Grid_Search_Model = GridSearchCV(estimator = KNeighborsClassifier(), 
                     param_grid=my_param_grid,  
                     scoring='accuracy', 
                     cv=5, n_jobs=-1).fit(Train_X_Std, Train_Y)


Grid_Search_Df = pd.DataFrame.from_dict(Grid_Search_Model.cv_results_)

#	params{'n_neighbors': 13, 'p': 2}

###################      
###################
# Data Re-Distribution (Changing data when there are outliers present in some columns)
# There is a good belief that some times outliers influence and interfere in the way algorithms work
# So, it may be needed to change the way data is originally distributed and feed the changed data
# into an algorithm.
###################
###################           
           
import seaborn as sns
import numpy as np

Train_X.columns
columns_ToConsider = ['Age']

# Histogram using seaborn
#sns.pairplot(Train_X[columns_ToConsider])

# Lets consider Area Mean and apply log transformation
Train_X_Copy = Train_X.copy()
Train_X_Copy["Age"] = np.log(np.where(Train_X_Copy["Age"] == 0, 1, Train_X_Copy["Age"]))

Test_X_Copy = Test_X.copy()
Test_X_Copy["Age"] = np.log(np.where(Test_X_Copy["Age"] == 0, 1, Test_X_Copy["Age"]))

FinalTest_X_Copy = FinalTest_X.copy()
FinalTest_X_Copy["Age"] = np.log(np.where(FinalTest_X_Copy["Age"] == 0, 1, FinalTest_X_Copy["Age"]))


# Histogram using seaborn
#sns.pairplot(Train_X_Copy[columns_ToConsider])


###################
# Standardization
###################

Train_Scaling = StandardScaler().fit(Train_X_Copy)
Train_X_Std = Train_Scaling.transform(Train_X_Copy)
Test_X_Std  = Train_Scaling.transform(Test_X_Copy)
FinalTest_X_Std  = Train_Scaling.transform(FinalTest_X_Copy)
# Add the column names to Train_X_Std, Test_X_Std
Train_X_Std = pd.DataFrame(Train_X_Std, columns = Train_X.columns)
Test_X_Std = pd.DataFrame(Test_X_Std, columns = Test_X.columns)
FinalTest_X_Std = pd.DataFrame(FinalTest_X_Std, columns = FinalTest_X.columns)


###################
# Modeling
###################
Train_X_Std.info()
Train_X_Std.shape
 
Train_X_Std.replace([np.inf, -np.inf], np.nan, inplace=True)


Train_X_Std.fillna(0.12843952118346222, inplace=True)


# Build model
M2 = KNeighborsClassifier(n_neighbors=3).fit(Train_X_Std, Train_Y)

# Class Prediction
Test_Pred = M2.predict(Test_X_Std)

###################
# Model evaluation
###################

Confusion_Mat = confusion_matrix(Test_Y, Test_Pred)
Confusion_Mat

sum(np.diagonal(Confusion_Mat))/Test_X.shape[0]*100 # 60
precision_score(Test_Y, Test_Pred) # 60
recall_score(Test_Y, Test_Pred) # 57
f1_score(Test_Y, Test_Pred) # 59
Confusion_Mat[0][1]/sum(Confusion_Mat[0]) # 35


##############
##############
# Final Prediction
##############
##############

PredSet = pd.DataFrame()
PredSet['S.No'] = TestRaw['S.No']

FinalTest['Test_Pred'] = M2.predict(FinalTest_X)
FinalTest.columns


PredSet['treatment']=FinalTest.Test_Pred
PredSet['treatment']= PredSet['treatment'].map({1 : 'Yes', 0: 'No'})


PredSet.to_csv('IMR_KNN.csv',index=False)

