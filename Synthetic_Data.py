import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
scaling=StandardScaler()
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
import streamlit as st
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


st.header("Welcome to the Synthetic Datasets Website")
st.write("The purpose of this webiste is to let users to create their own custom datasets whether it's a classification(Binary or Mutli-Binary Target Variable) or Regression(Dependent feature with Continous values).")
st.write()
select=st.radio("Please select the complexity level of your Data",["Easy","Medium","Complex","Custom"],horizontal=True)
if select=="Easy":
    st.write("In this Easy section, Data is Balanced with Binary Categories and there is no noise in the data moreover all the columns are informative that means the features that are present in the data are highly dependent on Dependent variable. The size of the data would be 1000 Rows and 3 Columns.")
    X, y = make_classification(
    n_samples=1000, 
    n_features=3, 
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=2,
    random_state=42)
    names=[]
    for i in range(X.shape[1]):
        names.append("Column {}".format(i))
    X1=pd.DataFrame(X,columns=names)
    y1=pd.DataFrame(y,columns=["Target"])
    data=pd.concat([X1,y1],axis=1)
    st.table(data.head())
    st.write("The occurence of {} Category is {} and the occurence of {} is {}".format(pd.Series(y).value_counts().index[0]
                                                                                ,pd.Series(y).value_counts().values[0]
                                                                                ,pd.Series(y).value_counts().index[1]
                                                                                ,pd.Series(y).value_counts().values[1]))
    
    fig=plt.figure(figsize=(12,8))
    sns.scatterplot(pca.fit_transform(X1)[:,0],pca.fit_transform(X1)[:,1],c=y)
    st.pyplot(fig)
    st.download_button(label="Data is Ready to Download",data=data.to_csv(index=False),file_name="User.csv",mime="csv")
if select=="Medium":
    st.write("In this Medium section, Data is slightly Imbalanced and there would be slightly noise in the data moreover all the columns are slightly informative that means the feature that are present in the data are less dependent on Dependent variable. The size of the data would be 2000 Rows and 5 Columns.")
    X, y = make_classification(
    n_samples=1000, 
    n_features=4, 
    n_redundant=0,
    flip_y=0.20,
    n_clusters_per_class=1,
    weights=[0.60],
    random_state=42)
    names=[]
    for i in range(X.shape[1]):
        names.append("Column {}".format(i))
    X1=pd.DataFrame(X,columns=names)
    y1=pd.DataFrame(y,columns=["Target"])
    data=pd.concat([X1,y1],axis=1)
    st.table(data.head())
    st.write("The occurence of {} Category is {} and the occurence of {} is {}".format(pd.Series(y).value_counts().index[0]
                                                                                ,pd.Series(y).value_counts().values[0]
                                                                                ,pd.Series(y).value_counts().index[1]
                                                                                ,pd.Series(y).value_counts().values[1]))
    fig=plt.figure(figsize=(12,8))
    sns.scatterplot(pca.fit_transform(X1)[:,0],pca.fit_transform(X1)[:,1],c=y)
    st.pyplot(fig)
    st.download_button(label="Data is Ready to Download",data=data.to_csv(index=False),file_name="User.csv",mime="csv")
if select=="Complex":
    st.write("In this Complex section, Data is highly Imbalanced and there would be a great amount of noise moreover majority of the columns are less informative that means the feature that are present in the data are less dependent on Dependent variable. The size of the data would be 5000 Rows and 10 Columns")
    X, y = make_classification(
    n_samples=5000, 
    n_features=9, 
    n_redundant=2,
    flip_y=0.7,
    n_clusters_per_class=2,
    weights=[0.75],
    random_state=42)
    names=[]
    for i in range(X.shape[1]):
        names.append("Column {}".format(i))
    X1=pd.DataFrame(X,columns=names)
    y1=pd.DataFrame(y,columns=["Target"])
    data=pd.concat([X1,y1],axis=1)
    st.table(data.head())
    st.write("The occurence of {} Category is {} and the occurence of {} is {}".format(pd.Series(y).value_counts().index[0]
                                                                                ,pd.Series(y).value_counts().values[0]
                                                                                ,pd.Series(y).value_counts().index[1]
                                                                                ,pd.Series(y).value_counts().values[1]))
    fig=plt.figure(figsize=(12,8))
    sns.scatterplot(pca.fit_transform(X1)[:,0],pca.fit_transform(X1)[:,1],c=y)
    st.pyplot(fig)
    st.download_button(label="Data is Ready to Download",data=data.to_csv(index=False),file_name="User.csv",mime="csv")
if select=="Custom":
    select_values=st.radio("Please select the kind of data that you want to make",["Only Continous Indpendent Features",
                                                                     "Only Categorical Indpendent Features",
                                                                     "Mixed Values( Both Continous and Categorical Features )"],horizontal=True)
    if select_values=="Only Continous Indpendent Features":
        int_feature=st.number_input("Please specify the total number of Independent Features you want in your Data",2,100)
        imb=st.radio("Do you want Imbalance Data",["No","Yes"])
        if imb=="Yes":
            imb_ratio=st.number_input("To what extend you want to make Imbalance Binary Categories",min_value=0.25,max_value=0.75,step=0.25)
            rows=st.number_input("Please specify the total Number of rows that you want to include in your data",100,100000,step=100)
            informative=st.number_input('''Please specify the informative features: this parameter in the scikit-learn function that specifies the number of features that are used to generate the target variable. It is used to control the complexity of the generated sample data. It determines the number of features that are truly informative, that is, the number of features that are used to determine the target variable. The remaining features are noisy and not related to the target, more the number is more informative the data would be ''',1,int_feature-1)
            class_sep=st.number_input('''Please specify to what extent you want to seperate the classes''',0,5)        
            X, y = make_classification(
            n_samples=rows,
            n_features=int_feature,
            n_classes=2,
            n_clusters_per_class=1,
            n_redundant=0,
            n_informative=informative+1,
            class_sep=class_sep,
            weights=[imb_ratio],
            random_state=42)
            tablet=pd.DataFrame(pd.Series(y).value_counts()).reset_index().rename(columns={"index":"Categories",0:"Occurences"})
            names=[]
            for i in range(X.shape[1]):
                names.append("Columns {}".format(i))
            X1=pd.DataFrame(X,columns=names)
            y1=pd.DataFrame(y,columns=["Target"])
            data=pd.concat([X1,y1],axis=1)
    
            fig=plt.figure(figsize=(12,8))
            sns.scatterplot(pca.fit_transform(X1)[:,0],pca.fit_transform(X1)[:,1],c=y)
            fig.legend()
            st.pyplot(fig)
            st.table(tablet)
            st.download_button(label="Data is Ready to Download",data=data.to_csv(index=False),file_name="User.csv",mime="csv")    
        else:
            dep_class=st.number_input("Please specify number of classes, by-default it will be Binary",2,20)
            rows=st.number_input("Please specify the total Number of rows that you want to include in your data",100,100000,step=100)
            informative=st.number_input('''Please specify the informative features: this parameter in the scikit-learn function that specifies the number of features that are used to generate the target variable. It is used to control the complexity of the generated sample data. It determines the number of features that are truly informative, that is, the number of features that are used to determine the target variable. The remaining features are noisy and not related to the target, more the number is more informative the data would be ''',1,int_feature-1)
            class_sep=st.number_input('''Please specify to what extent you want to seperate the classes''',0,5)        
            X, y = make_classification(
            n_samples=rows,
            n_features=int_feature,
            n_classes=dep_class,
            n_clusters_per_class=1,
            n_redundant=0,
            n_informative=informative+1,
            class_sep=class_sep,
            random_state=42)
            names=[]
            for i in range(X.shape[1]):
                names.append("Columns {}".format(i+1))
            X1=pd.DataFrame(X,columns=names)
            y1=pd.DataFrame(y,columns=["Target"])
            data=pd.concat([X1,y1],axis=1)
    
            fig=plt.figure(figsize=(12,8))
            sns.scatterplot(pca.fit_transform(X1)[:,0],pca.fit_transform(X1)[:,1],c=y)
            fig.legend()
            st.pyplot(fig)
            st.download_button(label="Data is Ready to Download",data=data.to_csv(index=False),file_name="User.csv",mime="csv")
    if select_values=="Only Categorical Indpendent Features":
        imb=st.radio("Do you want Imbalance Data",["No","Yes"])
        if imb=="Yes":
            int_feature=st.number_input("Please specify the total number of Independent Features you want in your Data",2,100)
            rows=st.number_input("Please specify the total Number of rows that you want to include in your data",100,100000,step=100)
            class_sep=st.number_input('''Please specify to what extent you want to seperate the classes''',0,5)
            data_range=st.number_input('''What range of values you want in your data''',2,15)
            imb_ratio=st.number_input("To what extend you want to make Imbalance Binary Categories",min_value=0.0,max_value=0.75,step=0.25)
            X, y = make_classification(
            n_samples=rows,
            n_features=int_feature,
            n_classes=2,
            n_clusters_per_class=1,
            n_redundant=0,
            weights=[imb_ratio],
            n_informative=1,
            class_sep=class_sep,
            random_state=42)
            x1=pd.DataFrame(y,columns=["Target"])
            numm=[]
            for i in range(rows):
                numm.append(np.random.randint(0,data_range,size=int_feature))
            names=[]
            for i in range(int_feature):
                names.append("Column {}".format(i+1))
            d=pd.DataFrame(numm,columns=names)
            tablet=pd.concat([d,x1],axis=1)
            st.dataframe(tablet)
            st.header("Total Number of Rows are {} and Total Number of Columns are {}".format(tablet.shape[0],tablet.shape[1]))
            tablet=pd.DataFrame(pd.Series(y).value_counts()).reset_index().rename(columns={"index":"Categories",0:"Occurences"})
            st.table(tablet)
            st.download_button(label="Data is Ready to Download",data=tablet.to_csv(index=False),file_name="User.csv",mime="csv")
        else:
            int_feature=st.number_input("Please specify the total number of Independent Features you want in your Data",2,100)
            rows=st.number_input("Please specify the total Number of rows that you want to include in your data",100,100000,step=100)
            dep_class=st.number_input("Please specify number of classes, by-default it will be Binary",2,20)
            class_sep=st.number_input('''Please specify to what extent you want to seperate the classes''',0,5)
            data_range=st.number_input('''What range of values you want in your data''',2,15)
            imb_ratio=st.number_input("To what extend you want to make Imbalance Binary Categories",min_value=0.0,max_value=0.75,step=0.25)
            X, y = make_classification(
            n_samples=rows,
            n_features=int_feature,
            n_classes=dep_class,
            n_clusters_per_class=1,
            n_redundant=0,
            n_informative=dep_class-1,
            class_sep=class_sep,
            random_state=42)
            x1=pd.DataFrame(y,columns=["Target"])
            numm=[]
            for i in range(rows):
                numm.append(np.random.randint(0,data_range,size=int_feature))
            names=[]
            for i in range(int_feature):
                names.append("Column {}".format(i+1))
            d=pd.DataFrame(numm,columns=names)
            tablet=pd.concat([d,x1],axis=1)
            st.dataframe(tablet)
            st.header("Total Number of Rows are {} and Total Number of Columns are {}".format(tablet.shape[0],tablet.shape[1]))
            tablet=pd.DataFrame(pd.Series(y).value_counts()).reset_index().rename(columns={"index":"Categories",0:"Occurences"})
            st.table(tablet)
            st.download_button(label="Data is Ready to Download",data=tablet.to_csv(index=False),file_name="User.csv",mime="csv")
    if select_values=="Mixed Values( Both Continous and Categorical Features )":
        imb=st.radio("Do you want Imbalance Data",["No","Yes"])
        if imb=="Yes":
            int_feature_con=st.number_input("Please specify the total number of Continous Independent Features you want in your Data",2,100)
            int_feature_cat=st.number_input("Please specify the total number of Categorical Independent Features you want in your Data",2,15)
            rows=st.number_input("Please specify the total Number of rows that you want to include in your data",100,100000,step=100)
            class_sep=st.number_input('''Please specify to what extent you want to seperate the classes''',0,5)
            imb_ratio=st.number_input("To what extend you want to make Imbalance Binary Categories",min_value=0.0,max_value=0.75,step=0.25)
            X, y = make_classification(
            n_samples=rows,
            n_features=int_feature_con,
            n_classes=2,
            n_clusters_per_class=1,
            n_redundant=0,
            weights=[imb_ratio],
            n_informative=1,
            class_sep=class_sep,
            random_state=42)
            numm=[]
            #Continous Values
            data_con=pd.DataFrame(X)
            #Categorical Values
            for i in range(X.shape[0]):
                numm.append(np.random.randint(0,int_feature_cat,size=X.shape[1]))
            data_cat=pd.DataFrame(numm)
            final_data=pd.concat([data_con,data_cat],axis=1)
            final_data.columns=range(final_data.columns.size)
            names=[]
            for i in range(final_data.shape[1]):
                final_data.rename(columns={i:"Column {}".format(i+1)},inplace=True)
            tablet=pd.DataFrame(pd.Series(y).value_counts()).reset_index().rename(columns={"index":"Categories",0:"Occurences"})
            names=[]
            for i in range(final_data.shape[1]):
                final_data.rename(columns={i:"Column {}".format(i+1)},inplace=True)
            st.write("Below table is the final Layout of the Mixed Datapoints")
            st.dataframe(pd.concat([final_data,pd.DataFrame(y,columns=["Target"])],axis=1))
            st.header("Total Number of Rows are {} and Total Number of Columns are {}".format(final_data.shape[0],final_data.shape[1]))
            st.table(tablet)
            st.download_button(label="Data is Ready to Download",data=tablet.to_csv(index=False),file_name="User.csv",mime="csv")
        else:
            int_feature_con=st.number_input("Please specify the total number of Continous Independent Features you want in your Data",3,100)
            int_feature_cat=st.number_input("Please specify the total number of Categorical Independent Features you want in your Data",2,15)
            rows=st.number_input("Please specify the total Number of rows that you want to include in your data",100,100000,step=100)
            dep_class=st.number_input("Please specify number of classes, by-default it will be Binary",2,int_feature_con-1)
            class_sep=st.number_input('''Please specify to what extent you want to seperate the classes''',0,5)
            X, y = make_classification(
            n_samples=rows,
            n_features=int_feature_con,
            n_classes=dep_class,
            n_clusters_per_class=1,
            n_redundant=0,
            n_informative=int_feature_con-1,
            class_sep=class_sep,
            random_state=42)
            numm=[]
            #Continous Values
            data_con=pd.DataFrame(X)
            #Categorical Values
            for i in range(X.shape[0]):
                numm.append(np.random.randint(0,int_feature_cat,size=X.shape[1]))
            data_cat=pd.DataFrame(numm)
            final_data=pd.concat([data_con,data_cat],axis=1)
            final_data.columns=range(final_data.columns.size)
            names=[]
            for i in range(final_data.shape[1]):
                final_data.rename(columns={i:"Column {}".format(i+1)},inplace=True)
            tablet=pd.DataFrame(pd.Series(y).value_counts()).reset_index().rename(columns={"index":"Categories",0:"Occurences"})
            names=[]
            for i in range(final_data.shape[1]):
                final_data.rename(columns={i:"Column {}".format(i+1)},inplace=True)
            st.write("Below table is the final Layout of the Mixed Datapoints")
            st.dataframe(pd.concat([final_data,pd.DataFrame(y,columns=["Target"])],axis=1))
            st.header("Total Number of Rows are {} and Total Number of Columns are {}".format(final_data.shape[0],final_data.shape[1]))
            st.table(tablet)
            st.download_button(label="Data is Ready to Download",data=tablet.to_csv(index=False),file_name="User.csv",mime="csv")