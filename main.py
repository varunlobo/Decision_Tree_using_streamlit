
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


header = st.container()
EDA = st.container()
modeltraining = st.container()



with header:
    
    st.title("Building a dynamic webpage using Streamlit in Python.")
    st.subheader("Streamlit is an open source app framework in Python language. \
            It helps us create web apps for data science and machine learning.\
            It is compatible with major Python libraries such as scikit-learn,\
            Keras, PyTorch, NumPy, pandas, Matplotlib etc")
    

with EDA:
    st.header("Let us perform Exploratory Data Analysis of our dataset")
    st.text("Dataset for illustration was downloaded from Kaggle.com")
    
    df=pd.read_csv("employee_data.csv")
    st.subheader("A view of our dataset")
    row = st.slider('Display Rows', 0, 50, 5)
    
    
    st.write(df.head(row))
    
    
    st.subheader("Let us review our data statistics")
    st.write(df.describe())
    
    st.markdown("Age distribution")
    st.bar_chart(df['Age'].value_counts()) 
    
    
    
    st.markdown("Data Distribution by Department")
    fig = plt.figure(figsize=(10, 6))
    sns.countplot(x="Department", data=df)
    st.pyplot(fig)
    
    st.markdown("Data Distribution by Job Role")
    fig = plt.figure(figsize=(10, 4))
    sns.countplot(x="JobRole", data=df)
    plt.xticks(rotation=70)
    st.pyplot(fig)
    
    st.markdown("Data Distribution by Gender")
    fig = plt.figure(figsize=(10, 6))
    sns.countplot(x="Gender", data=df)
    st.pyplot(fig)
    
    st.markdown("Feature Variable boxplot")
    fig= plt.figure(figsize=(15, 15))
    sns.boxplot(x='MonthlyIncome', y='JobRole', data=df)
    st.pyplot(fig)
    
with modeltraining:
    st.header("In this section we will divide our dataset into training \
              and testing datasets")
    
    split = st.slider('Input % for dividing dataset', 0, 100, 50)
    
    split=split/100
    
    st.header("Determine max depth of Decision Tree")
    maxdepth = st.slider('Max depth of decision tree', 0, 15, 5)
    
    y=df['Attrition'].map({'Yes':1, 'No':0})
    
    x=df[['Age', 'NumCompaniesWorked', 'JobSatisfaction' , 'PerformanceRating' ,'MonthlyIncome', 'JobLevel' , 'DistanceFromHome']]
    
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X = scaler.fit_transform(x)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=split, random_state=0)
    
    
    from sklearn.tree import DecisionTreeClassifier
    dtc = DecisionTreeClassifier(criterion='gini', max_depth=maxdepth)
    dtc.fit(X_train,y_train)
    y_predicted = dtc.predict(X_test)
    
    st.header("Error Metrics to Determine the accuracy of our ML model")
    
    from sklearn import metrics 
    st.subheader("Mean Absolute Error")
    st.write(metrics.mean_absolute_error(y_test, y_predicted))

    st.subheader("Root Mean Sq Error")
    st.write(np.sqrt(metrics.mean_squared_error(y_test, y_predicted)))
    
    st.subheader("R2 Score:")
    st.write(dtc.score(X_test, y_test))
    
    