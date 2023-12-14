import pandas as  pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import joblib
import streamlit as st
from sklearn.metrics import f1_score, accuracy_score

df=pd.read_csv("diabetes.csv")

st.dataframe(df.head())

st.sidebar.title("Sommaire🎉")
pages = ["Contexte du projet", "Exploration des données", "Analyse de données", "Modélisation"]

page = st.sidebar.radio("Aller vers la page :", pages)

if page == pages[0] : 
    
    st.write("### Contexte du projet")
    
    st.write(" L'ensemble de données provient de l'Institut national du diabète et des maladies digestives et rénales (NIDDK) et vise à résoudre un problème crucial dans le domaine médical : la prédiction diagnostique du diabète. Le diabète est une maladie chronique affectant la régulation du glucose dans le sang et a des implications significatives sur la santé publique..")
    
    st.write("Nous disposons d'un ensemble de données stocké dans le fichier diabet.csv. Cet ensemble de données comprend plusieurs caractéristiques médicales telles que l'âge, la pression artérielle, le taux de glucose, l'indice de masse corporelle (IMC) et d'autres indicateurs importants. Ces caractéristiques seront utilisées comme variables d'entrée pour le modèle de prédiction..")
    
    st.write("Dans un premier temps, nous explorerons ce dataset. Puis nous l'analyserons visuellement pour en extraire des informations selon certains axes d'étude. Finalement nous implémenterons des modèles de Machine Learning pour prédire si un patient souffre du diabete ou non .")
    
    st.image("ldiabete.jpg")
    
elif page == pages[1]:
    st.write("### Exploration des données")
    
    st.dataframe(df.head())
    
    st.write("Dimensions du dataframe :")
    
    st.write(df.shape)
    st.write("Description des donnees:")
    st.write(df.describe())
    st.write("le nombre d'observation de la variable cible:")
    st.write(df["Outcome"].value_counts())
    if st.checkbox("Afficher les valeurs manquantes") : 
        st.dataframe(df.isna().sum())
        
    if st.checkbox("Afficher les doublons") : 
        st.write(df.duplicated().sum())
        
elif page == pages[2]:
    st.write("### Analyse de données")

    fig3, ax = plt.subplots()
    sns.heatmap(df.corr(), ax=ax,annot=True)
    plt.title("Matrice de corrélation des variables du dataframe")
    st.write(fig3)
    
    fig = sns.displot(x='Outcome', data=df, kde=True)
    plt.title("Distribution de la variable cible Outcome")
    st.pyplot(fig)
    
    fig, ax=plt.subplots(figsize=(15,10))
    sns.boxplot(data=df,width=0.5,ax=ax,fliersize=10)
    plt.title("visualisation des valeurs aberrantes de chaque variable")
    st.pyplot(fig)

    fig=sns.pairplot(df)
    plt.title("visualisation de la relation entre les variable")
    plt.show()
    st.pyplot(fig)
    
elif page == pages[3]:
    st.write("### Modélisation")
    
    df_prep = pd.read_csv("diabetes.csv")
    
    x=df.drop("Outcome",axis=1).values
    y=df.Outcome.values
    
    from sklearn.preprocessing import StandardScaler
#initialiser 
    standar=StandardScaler()
#
    x=standar.fit_transform(x)
    
    #spliter les donnees
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2,random_state=42)
    x_val,x_test, y_val,y_test=train_test_split(x_test, y_test,test_size=0.5,random_state=42)
    
    reg = joblib.load("model_logisticR")
    svm = joblib.load("model_svm")
    knn = joblib.load("model_knn")
    
    
    y_pred_reg=reg.predict(x_val)
    y_pred_rf=svm.predict(x_val)
    y_pred_knn=knn.predict(x_val)
    
    model_choisi = st.selectbox(label = "Modèle", options = ['Logistique Regression', 'SVM', 'KNN'])
    
    def train_model(model_choisi) : 
        if model_choisi == 'Logistique Regression' :
            y_pred = y_pred_reg
        elif model_choisi == 'SVM' :
            y_pred = y_pred_rf
        elif model_choisi == 'KNN' :
            y_pred = y_pred_knn
        f1= f1_score(y_pred,y_val)
        acc=accuracy_score(y_pred, y_val)
        return f1,acc
    
    st.write("Le Score F1  et le taux de Précision (accuracy)", train_model(model_choisi))