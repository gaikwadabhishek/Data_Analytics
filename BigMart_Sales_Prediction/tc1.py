import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Sales_train = pd.read_csv("Train.csv")
Sales_train["Type"] = "train"
Sales_test = pd.read_csv("Test.csv")
Sales_test["Type"] = "test"

Data_Predicted = Sales_test.loc[:,["Item_Identifier","Outlet_Identifier","Item_MRP"]]  ## will use later while prediction


dframes = [Sales_train,Sales_test]
Sales = pd.concat(dframes,ignore_index = True, sort = False)
print(Sales.shape)


print(Sales['Item_Visibility'])
#Item visibility cannot be zero
Item_Visibility_Mean = Sales['Item_Visibility'].mean(skipna=True)
Sales['Item_Visibility'] = Sales['Item_Visibility'].replace(0,Item_Visibility_Mean)
print(Sales['Item_Visibility'])

print(Sales['Item_Weight'])
#replace missing values by Mean
Item_Weight_Mean = Sales['Item_Weight'].mean(skipna=True)
Sales['Item_Weight'] = Sales['Item_Weight'].replace(np.NaN,Item_Weight_Mean)
print(Sales['Item_Weight'])

import datetime
now = datetime.datetime.now()
now.year

Sales["Outlet_Age"] = now.year - Sales["Outlet_Establishment_Year"]
print(Sales["Outlet_Age"])

#to fill values in outlet_size
Sales.Outlet_Size.value_counts()
Sales.loc[Sales["Outlet_Size"].isnull(),"Outlet_Identifier"].value_counts()
#looking for pattern in missing outlet size

Sales.loc[(Sales["Outlet_Size"].isnull()) & (Sales["Outlet_Identifier"] == "OUT045") ,].describe(include = [object])
Sales.loc[(Sales["Outlet_Size"].isnull()) & (Sales["Outlet_Identifier"] == "OUT017") ,].describe(include = [object])
Sales.loc[(Sales["Outlet_Size"].isnull()) & (Sales["Outlet_Identifier"] == "OUT010") ,].describe(include = [object])

#As for OUT045 and OUT017; Outlet_Location_Type and Outlet_Type are same i.e. Tier 2 and Supermarket Type1, hence we can impute the data as per this

Sales.loc[(Sales["Outlet_Location_Type"]== "Tier 2") & (Sales["Outlet_Type"]=="Supermarket Type1") ,"Outlet_Size"].value_counts()
Sales.loc[(Sales["Outlet_Size"].isnull()) & (Sales["Outlet_Identifier"].isin(["OUT045","OUT017"])) ,"Outlet_Size"] = "Small"
Sales.loc[(Sales["Outlet_Identifier"].isin(["OUT045","OUT017"])) ,"Outlet_Size"].value_counts()
#for output010 there are no similar values we have to use other features to classify the model to predict 
#the outlet_size but as assignment scope Ill just assume OUT010 to be Medium

Sales.loc[(Sales["Outlet_Size"].isnull()) & (Sales["Outlet_Identifier"].isin(["OUT010"])) ,"Outlet_Size"] = "Medium"