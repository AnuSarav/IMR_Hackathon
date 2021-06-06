
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

#Dropping the outlier rows with Percentiles
upper_lim = FullRaw['Age'].quantile(.98)
lower_lim = FullRaw['Age'].quantile(.02)

FullRaw = FullRaw[(FullRaw['Age'] < upper_lim) & (FullRaw['Age'] > lower_lim)]


########################
# Dummy variable creation
########################

# Dummy variable creation
FullRaw2 = pd.get_dummies(FullRaw)
FullRaw2.shape


Train2 = FullRaw2[FullRaw2['Source_Train'] == 1].drop(['Source_Train'], axis = 1).copy()
FinalTest = FullRaw2[FullRaw2['Source_Train'] == 0].drop(['Source_Train'], axis = 1).copy()

Train2.shape
FinalTest.shape

########################
# Sampling into X and Y
########################
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

########################################
# Decision Tree Model
########################################

from sklearn.tree import DecisionTreeClassifier

M1 = DecisionTreeClassifier(random_state=123)
M1 = M1.fit(Train_X, Train_Y) # Indep, Dep

########################################
# Model Visualization
########################################

import pydotplus # Shuould give an error the first time!
from sklearn.tree import export_graphviz


dot_data = export_graphviz(M1, out_file=None, feature_names = Train_X.columns) # No error

# Step 3: Draw graphgraph = pydotplus.graph_from_dot_data(dot_data) # Error  


#graph.write_pdf("Treatment_DT_Plot.pdf") 

############################
# Prediction and Validation
############################

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Prediction on testset
Test_Pred = M1.predict(Test_X)

# Classification Model Validation
Confusion_Mat = confusion_matrix(Test_Y, Test_Pred)
Confusion_Mat

# Check the accuracy of the model
sum(np.diagonal(Confusion_Mat))/Test_X.shape[0]*100 # 65.%


precision_score(Test_Y, Test_Pred) # 0.62
recall_score(Test_Y, Test_Pred) # 0.73
f1_score(Test_Y, Test_Pred) # 0.68
Confusion_Mat[0][1]/sum(Confusion_Mat[0]) # 0.42


############
# DT Model 2
############

# Build Model
M2 = DecisionTreeClassifier(random_state=123, min_samples_leaf = 500)
M2 = M2.fit(Train_X, Train_Y)

# Vizualize Model
dot_data = export_graphviz(M2,feature_names = Train_X.columns)
graph = pydotplus.graph_from_dot_data(dot_data)  
graph.write_pdf("Treatment_DT_Plot3.pdf") 

# Prediction on testset
Test_Pred = M2.predict(Test_X)

# Classification Model Validation
Confusion_Mat = confusion_matrix(Test_Y, Test_Pred)
Confusion_Mat # R, C format (Actual = Test_Y, Predicted = Test_Pred)

# Check the accuracy of the model
sum(np.diagonal(Confusion_Mat))/Test_X.shape[0]*100 # 50%

precision_score(Test_Y, Test_Pred) #.65
recall_score(Test_Y, Test_Pred) # 1
f1_score(Test_Y, Test_Pred) # .79
Confusion_Mat[0][1]/sum(Confusion_Mat[0]) # 1


########################################
# Random Forest
########################################

from sklearn.ensemble import RandomForestClassifier

M1_RF = RandomForestClassifier(random_state = 123)
M1_RF = M1_RF.fit(Train_X, Train_Y)
Test_Pred = M1_RF.predict(Test_X)

# Confusion Matrix
Confusion_Mat = confusion_matrix(Test_Y, Test_Pred) # R, C format (Actual = Test_Y, Predicted = Test_Pred)
Confusion_Mat 

# Evaluation
(sum(np.diag(Confusion_Mat))/Test_Y.shape[0])*100 # 72.5%
precision_score(Test_Y, Test_Pred) # .70
recall_score(Test_Y, Test_Pred) # .76
f1_score(Test_Y, Test_Pred) # .73


# Variable importance
M1_RF.feature_importances_

Var_Importance_Df = pd.concat([pd.DataFrame(M1_RF.feature_importances_),
                               pd.DataFrame(Train_X.columns)], axis = 1)

Var_Importance_Df
Var_Importance_Df.columns = ["Value", "Variable_Name"]
Var_Importance_Df.sort_values("Value", ascending = False, inplace = True)
Var_Importance_Df
# Var_Importance_Df.to_csv("Var_Importance_Df.csv", index = False)

import seaborn as sns
plot = sns.scatterplot(x = "Variable_Name", y = "Value", data = Var_Importance_Df) 
# # Task: Rotate the xaxis labels to show up vertically. HINT: Check matplotlib functions on stackoverflow answers

#################
# RF Model with tuning parameters
#################

M2_RF = RandomForestClassifier(random_state=123, n_estimators = 25, 
                               max_features = 5, min_samples_leaf = 500)
M2_RF = M2_RF.fit(Train_X, Train_Y)
Test_Pred = M2_RF.predict(Test_X)

# Confusion Matrix
Confusion_Mat = confusion_matrix(Test_Y, Test_Pred) # R, C format (Actual = Test_Y, Predicted = Test_Pred)
Confusion_Mat 



#################
# Manual Grid Searching
#################

n_estimators_List = [25, 50, 75] # range(25,100,25)
max_features_List = [5, 7, 9] # range(5,11,2)
min_samples_leaf_List = [100, 200] # range(100,300,100)
Counter = 0

Tree_List = []
Num_Features_List = []
Samples_List = []
Accuracy_List = []

Model_Validation_Df = pd.DataFrame()
#Model_Validation_Df2 = pd.DataFrame()
#Model_Validation_Df3 = pd.DataFrame()

for i in n_estimators_List:    
    for j in max_features_List:        
        for k in min_samples_leaf_List:                        
            Counter = Counter + 1
            print(Counter)
#            print(i,j,k)            
            Temp_Model = RandomForestClassifier(random_state=123, n_estimators = i, 
                                                max_features = j, min_samples_leaf = k)
            Temp_Model = Temp_Model.fit(Train_X, Train_Y)
            Test_Pred = Temp_Model.predict(Test_X)                 
            Confusion_Mat = confusion_matrix(Test_Y, Test_Pred)
            Temp_Accuracy = (sum(np.diag(Confusion_Mat))/Test_Y.shape[0])*100            
#            print(i,i,k,Temp_Accuracy)
            
            # Alteranate 1
            Tree_List.append(i)
            Num_Features_List.append(j)
            Samples_List.append(k)
            Accuracy_List.append(Temp_Accuracy)
            
            
            
Model_Validation_Df = pd.DataFrame({'Trees': Tree_List, 'Max_Features': Num_Features_List, 
                                    'Min_Samples': Samples_List, 'Accuracy': Accuracy_List})
    # 25,7,100

########################################
# Random Forest using GridSearchCV
########################################

from sklearn.model_selection import GridSearchCV

my_param_grid = {'n_estimators': [25, 50, 75], 
                 'max_features': [5, 7, 9], 
                 'min_samples_leaf' : [100, 200]} 

Grid_Search_Model = GridSearchCV(estimator = RandomForestClassifier(random_state=123), 
                     param_grid=my_param_grid,  
                     scoring='accuracy', 
                     cv=3).fit(Train_X, Train_Y) # param_grid is a dictionary


Model_Validation_Df4 = pd.DataFrame.from_dict(Grid_Search_Model.cv_results_)
# 9,100,75
##############
#Final Model
##############
M5_RF = RandomForestClassifier(random_state=123, n_estimators = 75, 
                                                max_features = 9, min_samples_leaf = 100)#78.71%

M5_RF = M5_RF.fit(Train_X, Train_Y)
Test_Pred = M5_RF.predict(Test_X)

# Confusion Matrix
Confusion_Mat = confusion_matrix(Test_Y, Test_Pred) # R, C format (Actual = Test_Y, Predicted = Test_Pred)
Confusion_Mat 
(sum(np.diag(Confusion_Mat))/Test_Y.shape[0])*100 # 71
precision_score(Test_Y, Test_Pred) # .71
recall_score(Test_Y, Test_Pred) # .69
f1_score(Test_Y, Test_Pred) # .70



##############
##############
# Final Prediction
##############
##############

PredSet = pd.DataFrame()
PredSet['S.No'] = TestRaw['S.No']

FinalTest['Test_Pred'] = M5_RF.predict(FinalTest_X)
FinalTest.columns


PredSet['treatment']=FinalTest.Test_Pred
PredSet['treatment']= PredSet['treatment'].map({1 : 'Yes', 0: 'No'})


PredSet.to_csv('IMR_DT_RF.csv',index=False)
