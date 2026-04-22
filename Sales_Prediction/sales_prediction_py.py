import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

df = pd.read_csv(r"C:\Program Files (x86)\vcodestuff\MACHINE LEARNING\Sales_Prediction\Train.csv")

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

#print(df.isnull().sum())
#print(df.describe())

#.set()
# plt.figure(figsize=(6,6))
# sb.displot(df["Item_Weight"])
# plt.title("Titlt")
#plt.show()

# sb.set()
# plt.figure(figsize=(6,6))
# sb.displot(df["Item_Visibility"])
# plt.title("Titlt")
# plt.show()

# sb.set()
# plt.figure(figsize=(6,6))
# sb.displot(df["Item_MRP"])
# plt.title("Titlt")
# plt.show()

# sb.set()
# plt.figure(figsize=(6,6))
# sb.displot(df["Item_Outlet_Sales"])
# plt.title("Titlt")
# plt.show()

# sb.set()
# plt.figure(figsize=(6,6))
# sb.countplot(x="Outlet_Establishment_Year", data=df)
# plt.show()

# sb.set()
# plt.figure(figsize=(6,6))
# sb.countplot(x="Item_Fat_Content", data=df)
# plt.show()

# sb.set()
# plt.figure(figsize=(40,6))
# sb.countplot(y="Item_Type", data=df)
# plt.show()

# sb.set()
# plt.figure(figsize=(40,6))
# sb.countplot(x="Outlet_Size", data=df)
# plt.show()

#print(df["Item_Fat_Content"].value_counts())

df.replace({'Item_Fat_Content': {"low fat": "Low Fat", "LF": "Low Fat", "reg": "Regular"}},inplace=True)

#print(df["Item_Fat_Content"].value_counts())

encoder = LabelEncoder()

df["Item_Identifier"] = encoder.fit_transform(df["Item_Identifier"])
df["Item_Fat_Content"] = encoder.fit_transform(df["Item_Fat_Content"])
df["Item_Type"] = encoder.fit_transform(df["Item_Type"])
df["Outlet_Size"] = encoder.fit_transform(df["Outlet_Size"])
df["Outlet_Location_Type"] = encoder.fit_transform(df["Outlet_Location_Type"])
df["Outlet_Type"] = encoder.fit_transform(df["Outlet_Type"])
df["Outlet_Identifier"] = encoder.fit_transform(df["Outlet_Identifier"])

df["Item_Visibility"] = df["Item_Visibility"].replace(0, df["Item_Visibility"].mean())
df["Outlet_Years"] = 2026 - df["Outlet_Establishment_Year"]
#print(df["Item_Visibility"].value_counts())

# columns_to_convert = ["Item_Identifier", "Item_Fat_Content", "Item_Type", "Outlet_Size", "Outlet_Location_Type", "Outlet_Type", "Outlet_Identifier"]

# df = pd.get_dummies(df, columns=columns_to_convert)


#print(df.head(10))

X = df.drop(columns=["Item_Outlet_Sales","Item_Identifier","Outlet_Establishment_Year"], axis=1)
Y = df["Item_Outlet_Sales"]
#print(X)

#print(Y)

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=2)

#print(x_train.shape,x_test.shape)

model = XGBRegressor(n_estimators = 100, max_depth= 3, learning_rate = 0.1, random_state = 2)
model.fit(x_train,y_train)
prediction = model.predict(x_train)
r2_train = metrics.r2_score(y_train,prediction)

#print(f"Acuratetea pentru train este {r2_train}")
test_prediction = model.predict(x_test)
r2_test = metrics.r2_score(y_test,test_prediction)

print(f"Acuratetea pentru test este {r2_test}")
#print(df["Item_MRP"].mean())