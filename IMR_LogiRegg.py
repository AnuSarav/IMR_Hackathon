import os
import numpy as np
import pandas as pd

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
#upper_lim = FullRaw['Age'].quantile(.95)
#lower_lim = FullRaw['Age'].quantile(.05)

FullRaw = FullRaw[(FullRaw['Age'] < 100)]

# Dummy Variable Creation
#########################

FullRaw2= pd.get_dummies(FullRaw,drop_first= True)
FullRaw2.shape

Train2 = FullRaw2[FullRaw2['Source_Train'] == 1].drop(['Source_Train'], axis = 1).copy()
FinalTest = FullRaw2[FullRaw2['Source_Train'] == 0].drop(['Source_Train'], axis = 1).copy()

Train2.shape
FinalTest.shape

# Sampling
###########

 #Divide Train further into Train and Test by random sampling
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


############################
# Add Intercept Column
############################

from statsmodels.api import add_constant
Train_X = add_constant(Train_X)
Test_X = add_constant(Test_X)
FinalTest_X = add_constant(FinalTest_X)
Train_X.shape
Test_X.shape
FinalTest_X.shape

#########################
# VIF check
#########################
from statsmodels.stats.outliers_influence import variance_inflation_factor

temp_Max_VIF = 10
Max_VIF = 10
Train_X_Copy = Train_X.copy()
counter = 1
High_VIF_Column_Names = []

while (temp_Max_VIF >= Max_VIF):
    
    print(counter)
    
    Temp_VIF_Df = pd.DataFrame()
    Temp_VIF_Df['VIF'] = [variance_inflation_factor(Train_X_Copy.values, i) for i in range(Train_X_Copy.shape[1])]
    Temp_VIF_Df['Column_Name'] = Train_X_Copy.columns
    Temp_VIF_Df.dropna(inplace=True) # If there is some calculation error resulting in NAs
    Temp_Column_Name = Temp_VIF_Df.sort_values(["VIF"])[-1:]["Column_Name"].values[0]
    temp_Max_VIF = Temp_VIF_Df.sort_values(["VIF"])[-1:]["VIF"].values[0]
    print(Temp_Column_Name)
    
    if (temp_Max_VIF >= Max_VIF): # This condition will ensure that columns having VIF lower than 5 are NOT dropped
        Train_X_Copy = Train_X_Copy.drop(Temp_Column_Name, axis = 1)    
        High_VIF_Column_Names.append(Temp_Column_Name)
    
    counter = counter + 1

High_VIF_Column_Names


High_VIF_Column_Names.remove('const') # We need to exclude 'const' column from getting dropped/ removed

Train_X = Train_X.drop(High_VIF_Column_Names, axis = 1)
Test_X = Test_X.drop(High_VIF_Column_Names, axis = 1)
FinalTest_X = FinalTest_X.drop(High_VIF_Column_Names, axis = 1)


########################
# Model building
########################

from statsmodels.api import Logit  
M1 = Logit(Train_Y, Train_X) 
M1_Model = M1.fit(method='bfgs') 
M1_Model.summary() 


########################
# Manual model selection.
########################
Cols_To_Drop=[]
Cols_To_Drop.clear()
# Drop Marital_Unknownz
Cols_To_Drop = ["leave_Very easy"]
M2 = Logit(Train_Y, Train_X.drop(Cols_To_Drop, axis = 1)).fit(method='bfgs') # (Dep_Var, Indep_Vars)
M2.summary()


# Drop April_Bill_Amount
Cols_To_Drop.append('phys_health_consequence_Yes')
M3 = Logit(Train_Y, Train_X.drop(Cols_To_Drop, axis = 1)).fit(method='bfgs') # (Dep_Var, Indep_Vars)
M3.summary()

# Drop no_employees_26-100
Cols_To_Drop.append('no_employees_26-100')
M5 = Logit(Train_Y, Train_X.drop(Cols_To_Drop, axis = 1)).fit(method='bfgs') # (Dep_Var, Indep_Vars)
M5.summary()

# Drop Continent_Other  
Cols_To_Drop.append('Continent_Other')
M6 = Logit(Train_Y, Train_X.drop(Cols_To_Drop, axis = 1)).fit(method='bfgs') # (Dep_Var, Indep_Vars)
M6.summary()

# Drop tech_company_Yes 
Cols_To_Drop.append('tech_company_Yes')
M4 = Logit(Train_Y, Train_X.drop(Cols_To_Drop, axis = 1)).fit(method='bfgs') # (Dep_Var, Indep_Vars)
M4.summary()

# Drop mental_vs_physical_No
Cols_To_Drop.append('mental_vs_physical_No')
M7 = Logit(Train_Y, Train_X.drop(Cols_To_Drop, axis = 1)).fit(method='bfgs') # (Dep_Var, Indep_Vars)
M7.summary()

# Drop seek_help_Yes
Cols_To_Drop.append('seek_help_Yes')
M8 = Logit(Train_Y, Train_X.drop(Cols_To_Drop, axis = 1)).fit(method='bfgs') # (Dep_Var, Indep_Vars)
M8.summary()

# # Drop mental_health_interview_Yes
Cols_To_Drop.append('mental_health_interview_Yes')
M9 = Logit(Train_Y, Train_X.drop(Cols_To_Drop, axis = 1)).fit(method='bfgs') # (Dep_Var, Indep_Vars)
M9.summary()

Cols_To_Drop.append('phys_health_interview_No')
M10 = Logit(Train_Y, Train_X.drop(Cols_To_Drop, axis = 1)).fit(method='bfgs') # (Dep_Var, Indep_Vars)
M10.summary()

# Drop remote_work_Yes 
Cols_To_Drop.append('remote_work_Yes')
M11 = Logit(Train_Y, Train_X.drop(Cols_To_Drop, axis = 1)).fit(method='bfgs') # (Dep_Var, Indep_Vars)
M11.summary()

# Drop Continent_Europe
Cols_To_Drop.append('Continent_Europe')
M12 = Logit(Train_Y, Train_X.drop(Cols_To_Drop, axis = 1)).fit(method='bfgs') # (Dep_Var, Indep_Vars)
M12.summary()
"""
# Drop phys_health_interview_No
Cols_To_Drop.append('phys_health_interview_No')
M13 = Logit(Train_Y, Train_X.drop(Cols_To_Drop, axis = 1)).fit(method='bfgs') # (Dep_Var, Indep_Vars)
M13.summary()
"""
# # Drop care_options_Not sure  
Cols_To_Drop.append('care_options_Not sure')
M14 = Logit(Train_Y, Train_X.drop(Cols_To_Drop, axis = 1)).fit(method='bfgs') # (Dep_Var, Indep_Vars)
M14.summary()


Cols_To_Drop.append('phys_health_interview_No')
M15 = Logit(Train_Y, Train_X.drop(Cols_To_Drop, axis = 1)).fit(method='bfgs') # (Dep_Var, Indep_Vars)
M15.summary()

# Drop Gender_Neuter 
Cols_To_Drop.append('Gender_Neuter')
M16 = Logit(Train_Y, Train_X.drop(Cols_To_Drop, axis = 1)).fit(method='bfgs') # (Dep_Var, Indep_Vars)
M16.summary()

# Drop mental_health_consequence_Yes
Cols_To_Drop.append('mental_health_consequence_Yes')
M17 = Logit(Train_Y, Train_X.drop(Cols_To_Drop, axis = 1)).fit(method='bfgs') # (Dep_Var, Indep_Vars)
M17.summary()

# Drop wellness_program_No
Cols_To_Drop.append('wellness_program_No')
M18 = Logit(Train_Y, Train_X.drop(Cols_To_Drop, axis = 1)).fit(method='bfgs') # (Dep_Var, Indep_Vars)
M18.summary()

# # Drop benefits_No   
Cols_To_Drop.append('benefits_No')
M19 = Logit(Train_Y, Train_X.drop(Cols_To_Drop, axis = 1)).fit(method='bfgs') # (Dep_Var, Indep_Vars)
M19.summary()

# Drop anonymity_Yes
Cols_To_Drop.append('anonymity_Yes')
M20 = Logit(Train_Y, Train_X.drop(Cols_To_Drop, axis = 1)).fit(method='bfgs') # (Dep_Var, Indep_Vars)
M20.summary()

# # Drop leave_Very difficult
Cols_To_Drop.append('leave_Very difficult')
M21 = Logit(Train_Y, Train_X.drop(Cols_To_Drop, axis = 1)).fit(method='bfgs') # (Dep_Var, Indep_Vars)
M21.summary()


Cols_To_Drop.append('mental_health_interview_No')
M22 = Logit(Train_Y, Train_X.drop(Cols_To_Drop, axis = 1)).fit(method='bfgs') # (Dep_Var, Indep_Vars)
M22.summary()

# Drop Age 
Cols_To_Drop.append('Age')
M23 = Logit(Train_Y, Train_X.drop(Cols_To_Drop, axis = 1)).fit(method='bfgs') # (Dep_Var, Indep_Vars)
M23.summary()

# Drop Continent_South America
Cols_To_Drop.append('Continent_South America')
M24 = Logit(Train_Y, Train_X.drop(Cols_To_Drop, axis = 1)).fit(method='bfgs') # (Dep_Var, Indep_Vars)
M24.summary()

# Drop self_employed_Yes
Cols_To_Drop.append('self_employed_Yes')
M25 = Logit(Train_Y, Train_X.drop(Cols_To_Drop, axis = 1)).fit(method='bfgs') # (Dep_Var, Indep_Vars)
M25.summary()

# # Drop phys_health_consequence_No   
Cols_To_Drop.append('phys_health_consequence_No')
M26 = Logit(Train_Y, Train_X.drop(Cols_To_Drop, axis = 1)).fit(method='bfgs') # (Dep_Var, Indep_Vars)
M26.summary()

# Drop Continent_Australia
Cols_To_Drop.append('Continent_Australia')
M27 = Logit(Train_Y, Train_X.drop(Cols_To_Drop, axis = 1)).fit(method='bfgs') # (Dep_Var, Indep_Vars)
M27.summary()

# # Drop anonymity_No 
Cols_To_Drop.append('anonymity_No')
M28 = Logit(Train_Y, Train_X.drop(Cols_To_Drop, axis = 1)).fit(method='bfgs') # (Dep_Var, Indep_Vars)
M28.summary()


Cols_To_Drop.append('Continent_Asia')
M29 = Logit(Train_Y, Train_X.drop(Cols_To_Drop, axis = 1)).fit(method='bfgs') # (Dep_Var, Indep_Vars)
M29.summary()

# Drop mental_vs_physical_Yes 
Cols_To_Drop.append('mental_vs_physical_Yes')
M30 = Logit(Train_Y, Train_X.drop(Cols_To_Drop, axis = 1)).fit(method='bfgs') # (Dep_Var, Indep_Vars)
M30.summary()

# Drop wellness_program_Yes 
Cols_To_Drop.append('wellness_program_Yes')
M31 = Logit(Train_Y, Train_X.drop(Cols_To_Drop, axis = 1)).fit(method='bfgs') # (Dep_Var, Indep_Vars)
M31.summary()


# Drop no_employees_100-500 
Cols_To_Drop.append('no_employees_100-500')
M32 = Logit(Train_Y, Train_X.drop(Cols_To_Drop, axis = 1)).fit(method='bfgs') # (Dep_Var, Indep_Vars)
M32.summary()

# Drop leave_Somewhat easy 
Cols_To_Drop.append('leave_Somewhat easy')
M33 = Logit(Train_Y, Train_X.drop(Cols_To_Drop, axis = 1)).fit(method='bfgs') # (Dep_Var, Indep_Vars)
M33.summary()

# Drop no_employees_More than 1000 
Cols_To_Drop.append('no_employees_More than 1000')
M34 = Logit(Train_Y, Train_X.drop(Cols_To_Drop, axis = 1)).fit(method='bfgs') # (Dep_Var, Indep_Vars)
M34.summary()

# Drop no_employees_6-25  
Cols_To_Drop.append('no_employees_6-25')
M35 = Logit(Train_Y, Train_X.drop(Cols_To_Drop, axis = 1)).fit(method='bfgs') # (Dep_Var, Indep_Vars)
M35.summary()

# Drop no_employees_500-1000  
Cols_To_Drop.append('no_employees_500-1000')
M36 = Logit(Train_Y, Train_X.drop(Cols_To_Drop, axis = 1)).fit(method='bfgs') # (Dep_Var, Indep_Vars)
M36.summary()

# Drop seek_help_No 
Cols_To_Drop.append('seek_help_No')
M37 = Logit(Train_Y, Train_X.drop(Cols_To_Drop, axis = 1)).fit(method='bfgs') # (Dep_Var, Indep_Vars)
M37.summary()

# Drop supervisor_Some of them  
Cols_To_Drop.append('supervisor_Some of them')
M38 = Logit(Train_Y, Train_X.drop(Cols_To_Drop, axis = 1)).fit(method='bfgs') # (Dep_Var, Indep_Vars)
M38.summary()

# Drop supervisor_Yes 
Cols_To_Drop.append('supervisor_Yes')
M39 = Logit(Train_Y, Train_X.drop(Cols_To_Drop, axis = 1)).fit(method='bfgs') # (Dep_Var, Indep_Vars)
M39.summary()

# Drop coworkers_Some of them  
Cols_To_Drop.append('coworkers_Some of them')
M40 = Logit(Train_Y, Train_X.drop(Cols_To_Drop, axis = 1)).fit(method='bfgs') # (Dep_Var, Indep_Vars)
M40.summary()

############################
# Prediction and Validation
############################

Train_X = Train_X.drop(Cols_To_Drop, axis = 1)
Test_X = Test_X.drop(Cols_To_Drop, axis = 1) 
FinalTest_X = FinalTest_X.drop(Cols_To_Drop, axis = 1)

Train_X.shape
Test_X.shape
FinalTest_X.shape


Test_X['Test_Prob'] = M40.predict(Test_X)
Test_X.columns
Test_X['Test_Prob'][0:6]
Test_Y[:6]

# Classify 0 or 1 based on 0.5 cutoff
import numpy as np
Test_X['Test_Class'] = np.where(Test_X['Test_Prob'] >= 0.5, 1, 0)
Test_X.columns # A new column called Test_Class should be created


########################
# Confusion matrix
########################
#import pandas as pd
Confusion_Mat = pd.crosstab(Test_X['Test_Class'], Test_Y) 
Confusion_Mat

# Check the accuracy of the model
(sum(np.diagonal(Confusion_Mat))/Test_X.shape[0])*100 # ~78%

########################
# F1 Score
########################

from sklearn.metrics import f1_score, precision_score, recall_score
f1_score(Test_Y, Test_X['Test_Class']) # 78
precision_score(Test_Y, Test_X['Test_Class']) # 75
recall_score(Test_Y, Test_X['Test_Class']) # 81

########################
# AUC and ROC Curve
########################

from sklearn.metrics import roc_curve, auc
# Predict on train data
Train_Prob = M40.predict(Train_X)

# Calculate FPR, TPR and Cutoff Thresholds
fpr, tpr, cutoff = roc_curve(Train_Y, Train_Prob)


# Cutoff Table Creation
Cutoff_Table = pd.DataFrame()
Cutoff_Table['FPR'] = fpr 
Cutoff_Table['TPR'] = tpr
Cutoff_Table['Cutoff'] = cutoff

# Plot ROC Curve
import seaborn as sns
sns.lineplot(Cutoff_Table['FPR'], Cutoff_Table['TPR'])

# Area under curve (AUC)
auc(fpr, tpr) #0.83



############################
# Improve Model Output Using New Cutoff Point
############################

import numpy as np
Cutoff_Table['Distance'] = np.sqrt((1-Cutoff_Table['TPR'])**2 + (0-Cutoff_Table['FPR'])**2) # Euclidean Distance
Cutoff_Table['MaxDiffBetweenTPRFPR'] = Cutoff_Table['TPR'] - Cutoff_Table['FPR'] # Max Diff. Bet. TPR & FPR

# New Cutoff Point Performance (Obtained after studying ROC Curve and Cutoff Table)
cutoffPoint = 0.4895194779489488
 # Max Difference between TPR & FPR

# Classify the test predictions into classes of 0s and 1s
Test_X['Test_Class2'] = np.where(Test_X['Test_Prob'] >= cutoffPoint, 1, 0)

# Confusion Matrix
Confusion_Mat2 = pd.crosstab(Test_X['Test_Class2'], Test_Y) # R, C format
Confusion_Mat2

# Model Evaluation Metrics
sum(np.diagonal(Confusion_Mat2))/Test_X.shape[0]*100 #78
f1_score(Test_Y, Test_X['Test_Class2'], Test_Y) #0.78.5
precision_score(Test_Y, Test_X['Test_Class2'], Test_Y) # 0.75
recall_score(Test_Y, Test_X['Test_Class2'], Test_Y) # 0.81




##############
##############
# Final Prediction
##############
##############




PredSet = pd.DataFrame()
PredSet['S.No'] = TestRaw['S.No']

FinalTest['Test_Prob'] = M40.predict(FinalTest_X)
PredSet['treatment'] = np.where(FinalTest['Test_Prob'] >= 0.49, 'Yes', 'No')
FinalTest.columns




PredSet.to_csv('IMR_LogiRegg.csv',index=False)
