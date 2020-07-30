import streamlit as st
import pandas as pd
import numpy as  np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score,recall_score



st.title("Binary Classification Web App")
#just adding a sidebar to make it visually more better
st.sidebar.title("Binary Classification Web App")
#always rerun helps you to live code. u dont need to press "rerun" after every change.

#u can always get documentation about which func you like via  writing it wout parantheses
#ex: st.markdown then click ctrl s and documentation will be shown.

#u can add emojis to the markdown, dont forget that.
st.markdown("Are your mushrooms edible or poisonous? 🍄 ")
st.sidebar.markdown("Are your mushrooms edible or poisonous? 🍄 ")
#what streamlit does, check our project from top the bottom and rather than scanning each line,
#and rebuilding the page each time, it only adds the changes. very smart. so its much faster.

#since we dont want our app to load the data every time we make changes in our code, we could use
#streamlit func (its actually decorator not a func, its a func decorator). what this allows us to
#do is if func doesnt change (load_data) we can cache the output to disk and used cached result
#anytime we rerun. this doesnt make too much dif, but if we get millions,100 millions data then
#it does. u'll have many simultaneous users and many of them does simultaneous changes to their
#hparam settings and each time they do this ur server will be using some precious cpu cycle and
#memory if u loading the data over and over again. which ll cause perf drop.caching to disk is ok

@st.cache(persist=True)
def load_data():
    data=pd.read_csv("mushrooms.csv")
    label=LabelEncoder()
    for col in data.columns:
        data[col]=label.fit_transform(data[col])
    return data

df=load_data()

#u cant see anything bcs we need to sth extra to make our df visible. we just import the data. we
#dont do anything else. We ll deal with this via checkbox... but first lets do some ml steps.

@st.cache(persist=True)
def split(df):
    y=df.type
    x=df.drop(columns=["type"])
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
    return x_train, x_test, y_train, y_test

#its time for build couple metrics for testers to choose. we ll choose from list of metrics.
#for confusion matrix: takes estimator(model), x_test and y_test w labels if u want.
#also if u want to see matplotlib.pyplot visualization in yr webapp u should add st.pyplot()
def plot_metrics(metrics_list):
    if "Confusion Matrix" in metrics_list:
        st.subheader("Confusion Matrix")
        plot_confusion_matrix(model, x_test, y_test,display_labels=class_names)
        st.pyplot()

    if "ROC Curve" in metrics_list:
        st.subheader("ROC Curve")
        plot_roc_curve(model, x_test, y_test)
        st.pyplot()

    if "Precision-Recall Curve" in metrics_list:
        st.subheader("Precision-Recall Curve")
        plot_precision_recall_curve(model, x_test, y_test)
        st.pyplot()

#its time to choose right hparam from dropdown list.


df = load_data()
x_train, x_test, y_train, y_test =split(df)
class_names=["edible","poisonous"]
#its time to add a sidebar subheader to tell users to choose a classifier
st.sidebar.subheader("Choose Classifier")
#widget in a dropdown fashion to show you which ml u want
classifier=st.sidebar.selectbox("Classifier",("SVM","LogisticRegression","RandomForest"))
#so once the user slcs the model from dropdown we ll show here relevant inputs about classifier

#starting w SVM
if classifier == "SVM":
    #we ll let users to slc his/her hparams.
    st.sidebar.subheader("Model Hyperparameters")
    #now we ll provide hparams in some streamlit widget format. C is reg param,lower C underfit.
    #again maybe slider maybe dropdown not sure but this ll add some control on C. key is imp!!!
    #widgets have their own uniq key. basically key set this particular functionality to this
    #unique widget. Idk what this means. u can remove key and try again if u want.
    C=st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C")
    #now we add SVM kernels as radio button. rbf or linear. key could be var name for easier appr
    kernel=st.sidebar.radio("Kernel",("rbf","linear"),key="kernel")
    #and lets add gamma. kernel coef. key could be var name for easier approach. no other reason
    gamma=st.sidebar.radio("Gamma (Kernel Coefficient)",("scale","auto"),key="gamma")

    #here is new widget called "multiselect" u can select more than one item now..
    #adding txt to our multiselect, this vals on the multiselect should be same as plot_metrics
    metrics=st.sidebar.multiselect("What metrics to plot?",("Confusion Matrix","ROC Curve","Precision-Recall Curve"))

    #we need another if block here. bcs on each parameter change we dont want to instantenously
    #update the main page?? bcs user will be updating multiple hparams and also choosing eval
    #metrics for the classifier. so on each user action we dont want to simultaenously update
    #webpage. we want to provide a button, say sth like "classify" and when user presses it
    #will update the app. and we will do these steps for other 2 classifiers as well.

    #when users click this, it ll do certain set of operations.. and if users click this button
    #here is what ll happen.
    if st.sidebar.button("Classify",key="classify"):
        st.subheader("SVM Results")
        model=SVC(C=C,kernel=kernel,gamma=gamma)
        model.fit(x_train,y_train)
        accuracy=model.score(x_test,y_test)
        y_pred=model.predict(x_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision:", precision_score(y_test,y_pred,labels=class_names).round(2))
        st.write("Recall:", recall_score(y_test,y_pred,labels=class_names).round(2))
        plot_metrics(metrics)

#what is extra/dif for logreg:
#we add slider to sidebar for max_iter.
#and for if st.sidebar.button(classify,key=classify) part:
#just add max_iter and remove other unnecessary hparams for our model.
if classifier == "LogisticRegression":

    st.sidebar.subheader("Model Hyperparameters")
    C=st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C")
    max_iter=st.sidebar.slider("Maximum number of iterations",100,500,key="max_iter")

    #kernel=st.sidebar.radio("Kernel",("rbf","linear"),key="kernel")

    #gamma=st.sidebar.radio("Gamma (Kernel Coefficient)",("scale","auto"),key="gamma")

    metrics=st.sidebar.multiselect("What metrics to plot?",("Confusion Matrix","ROC Curve","Precision-Recall Curve"))

    if st.sidebar.button("Classify",key="classify"):
        st.subheader("Logistic Regression Results")
        model=SVC(C=C,max_iter=max_iter)
        model.fit(x_train,y_train)
        accuracy=model.score(x_test,y_test)
        y_pred=model.predict(x_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision:", precision_score(y_test,y_pred,labels=class_names).round(2))
        st.write("Recall:", recall_score(y_test,y_pred,labels=class_names).round(2))
        plot_metrics(metrics)

#changes for RandomForest:
#no C or no max_iter but we have n_estimators, also key name could be anything but it shouldnt
#conflict w other key names.
if classifier == "RandomForest":

    st.sidebar.subheader("Model Hyperparameters")
    #C=st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C")
    #max_iter=st.sidebar.slider("Maximum number of iterations",100,500,key="max_iter")
    #kernel=st.sidebar.radio("Kernel",("rbf","linear"),key="kernel")
    #gamma=st.sidebar.radio("Gamma (Kernel Coefficient)",("scale","auto"),key="gamma")
    n_estimators=st.sidebar.number_input("The number of trees in the forest",100,5000,step=10,key="n_estimators")
    max_depth=st.sidebar.number_input("The maximum depth of the trees",1,20,step=1,key="max_depth")
    bootstrap=st.sidebar.radio("Bootstrap samples when building trees",("True","False"),key="bootstrap")
    metrics=st.sidebar.multiselect("What metrics to plot?",("Confusion Matrix","ROC Curve","Precision-Recall Curve"))

    if st.sidebar.button("Classify",key="classify"):
        st.subheader("Random Forest Results")
        model=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,bootstrap=bootstrap,n_jobs=-1)
        model.fit(x_train,y_train)
        accuracy=model.score(x_test,y_test)
        y_pred=model.predict(x_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision:", precision_score(y_test,y_pred,labels=class_names).round(2))
        st.write("Recall:", recall_score(y_test,y_pred,labels=class_names).round(2))
        plot_metrics(metrics)






#here is the widget to solve this issue.checkbox. when checked we could perform some actions.
#as u can see False means default state of the cbox is unchecked.when u clicked it it becomes
#true thus statement works. .write() will write yr df to the webapp. now if u checked th box ucan
#see the df.

if st.sidebar.checkbox("Show raw data",False):
    st.subheader("Mushroom Data Set (Classification)")
    st.write(df)
