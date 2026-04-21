import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRFRegressor
from sklearn import metrics

df = pd.read_csv(r"W:\vscode\SQL\MachineLearningProject\Sales_Prediction\Train.csv")

#print(df.head(6))
#print(df.shape)
#print(df.info())
# categorical values = Item_Identifier Item_Fat_Content  Item_Type Outlet_Identifier Outlet_Size Outlet_Location_Type 
# Outlet_Type 
# restu numerica values

#print(df.isnull().sum())
# avem 1463 nul in item weight si 2410 in outlet_size

df["Item_Weight"] = df["Item_Weight"].fillna(df["Item_Weight"].mean())

#print(df.isnull().sum())
#print(df.head(10))
#print(df["Item_Weight"].mean())

mode_of_outlet_size = df.pivot_table(values="Outlet_Size", columns="Outlet_Type", aggfunc=(lambda x: x.mode()[0]))
miss_values = df["Outlet_Size"].isnull()
df.loc[miss_values, "Outlet_Size"] = df.loc[miss_values, "Outlet_Type"].apply(lambda x: mode_of_outlet_size[x])

print(df.isnull().sum())