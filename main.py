
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


header = st.container()
EDA = st.container()
modeltraining = st.container()

@st.cache
def get_data():
    
    df=pd.read_csv("employee_data.csv")
    
    return df


with header:
    
    st.title("Implementation of ML classification algorithms using Streamlit in Python.")
    st.markdown("Streamlit is an open source app framework in Python language. \
            It helps us create web apps for data science and machine learning.\
            It is compatible with major Python libraries such as scikit-learn,\
            Keras, PyTorch, NumPy, pandas, Matplotlib etc")
    

with EDA:
    st.header("Exploratory Data Analysis of our dataset")
    st.text("Dataset for illustration was downloaded from Kaggle.com")
    
    df=get_data()
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
    
    st.header("Let us dive into implementation of our ML models.")
    st.subheader("We will implement three different models to compare from. \
             Support Vector Machines (SVM), Decision Trees (DT) and \
                 Logistic Regression")
    
    st.subheader("Let's first divide our dataset into training \
              and testing datasets")
    
    split = st.slider('Input % for dividing dataset', 0, 100, 50)
    
    split=split/100
    

    
    y=df['Attrition'].map({'Yes':1, 'No':0})
    
    x=df[['Age', 'NumCompaniesWorked', 'JobSatisfaction' , 'PerformanceRating' ,'MonthlyIncome', 'JobLevel' , 'DistanceFromHome']]
    
    #Standard Scalar for Normalization
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X = scaler.fit_transform(x)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=split, random_state=0)
    
 
    st.header("Machine Learning (Classification) Algorithm Selection")
    mlselect=st.selectbox("",options=['Support Vector Machines (SVM)', 'Decision Trees (DT)', 'Logistic Regression'])
    
    colA, colB = st.columns(2)
    
    
    #default algo = DT
    
    from sklearn.tree import DecisionTreeClassifier
    dtc = DecisionTreeClassifier(criterion='gini', max_depth=5)
    dtc.fit(X_train,y_train)
    y_predicted = dtc.predict(X_test)  

    
    if mlselect=='Decision Trees (DT)':
        
        # Decision Tree
        colA.subheader("Determine max depth of Decision Tree")
        maxdepth = colA.slider('Max depth of decision tree', 0, 15, 5)
        
        colA.subheader("Hyper Parameter Selection")
        criteriaDT=colA.selectbox("",options=['gini','entropy','log_loss'])
        
        from sklearn.tree import DecisionTreeClassifier
        dtc = DecisionTreeClassifier(criterion=criteriaDT, max_depth=maxdepth)
        dtc.fit(X_train,y_train)
        y_predicted = dtc.predict(X_test)  
    
    if mlselect=='Logistic Regression':
        
        maxiter = colA.slider('Max iteration', 0, 100, 50)
        #Building logistic regression model
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=maxiter)
        model.fit(X_train,y_train)
        y_predicted = model.predict(X_test)
        
    if mlselect=='Support Vector Machines (SVM)':
         
        colA.subheader("Hyper Parameter Selection")
        kernelsvm=colA.selectbox("", options=["linear", "poly", "rbf", "sigmoid"])
         
        from sklearn.svm import SVC #"Support vector classifier"
        model = SVC(kernel=kernelsvm)
        model.fit(X_train,y_train)
        y_predicted = model.predict(X_test)
         
         
        
    # Error Metrics
    colB.subheader("Error Metrics to Determine the accuracy of our ML model")
    
    from sklearn import metrics 
    colB.subheader("Mean Absolute Error")
    colB.write(metrics.mean_absolute_error(y_test, y_predicted))

    colB.subheader("Root Mean Sq Error")
    colB.write(np.sqrt(metrics.mean_squared_error(y_test, y_predicted)))
    
    colB.subheader("R2 Score:")
    colB.write(dtc.score(X_test, y_test))
    
    