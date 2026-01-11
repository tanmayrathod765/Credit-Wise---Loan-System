#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[11]:


df = pd.read_csv("loan_approval_data.csv")
df.info()
df.isnull().sum()
df.describe()


# In[14]:


numerical_val = df.select_dtypes(include =  ["number"]).columns
categorical_val = df.select_dtypes(include =  ["object"]).columns


# In[15]:


from sklearn.impute import SimpleImputer


# In[19]:


num_imp = SimpleImputer(strategy = "mean")
df[numerical_val] = num_imp.fit_transform(df[numerical_val])
cat_imp = SimpleImputer(strategy = "most_frequent")
df[categorical_val] = cat_imp.fit_transform(df[categorical_val])


# In[21]:


df.isnull().sum()


# In[24]:


classes_count = df["Loan_Approved"].value_counts()
plt.pie(classes_count,labels = ["no" , "yes"],autopct = "%1.1f%%")


# In[32]:


edu_count = df["Loan_Purpose"].value_counts()
ax = sns.barplot(edu_count)
ax.bar_label(ax.containers[0])


# In[37]:


fig, axes = plt.subplots(3, 3)

sns.boxplot(ax=axes[0, 0], data=df, x="Loan_Approved",y="Applicant_Income")
sns.boxplot(ax=axes[0, 1], data=df, x="Loan_Approved",y="Credit_Score")
sns.boxplot(ax=axes[1, 0], data=df, x="Loan_Approved",y="DTI_Ratio")
sns.boxplot(ax=axes[1, 1], data=df, x="Loan_Approved",y="Savings")
sns.boxplot(ax=axes[1, 2], data=df, x="Loan_Approved",y="Gender")

plt.tight_layout()


# In[39]:


sns.histplot(

    data = df,
    x = "Credit_Score",
    hue = "Loan_Approved",
    multiple = "dodge"
    
)


# In[47]:


df = df.drop("Applicant_ID",axis=1)


# In[48]:


df.columns


# In[56]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le = LabelEncoder()
df["Education_Level"] = le.fit_transform(df["Education_Level"])
df["Loan_Approved"] = le.fit_transform(df["Loan_Approved"])

cols = ["Employment_Status", "Marital_Status", "Loan_Purpose", "Property_Area", "Gender", "Employer_Category"]
oho = OneHotEncoder(drop = "first" , sparse_output=False, handle_unknown="ignore")
encoded = oho.fit_transform(df[cols])

encoded_df = pd.DataFrame(encoded, columns=oho.get_feature_names_out(cols), index=df.index)
df = pd.concat([df.drop(columns=cols), encoded_df], axis=1)


# In[61]:


num_cols = df.select_dtypes(include="number")
corr_matrix = num_cols.corr();
num_cols.corr()["Loan_Approved"]


# In[65]:


plt.figure(figsize = (15,8))
sns.heatmap(
    corr_matrix,
    annot = True,
    fmt = ".2f",
    cmap = "coolwarm"  
)


# In[66]:


X = df.drop("Loan_Approved", axis=1)
y = df["Loan_Approved"]


# In[67]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[69]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[72]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

log_model = LogisticRegression()
log_model.fit(X_train_scaled,y_train)

y_pred = log_model.predict(X_test_scaled)
print("Logistic Regression Model")
print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("F1 score: ", f1_score(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("CM: ", confusion_matrix(y_test, y_pred))


# In[74]:


from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors = 5 )
knn_model.fit(X_train_scaled,y_train)

y_pred = knn_model.predict(X_test_scaled)
print("Logistic Regression Model")
print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("F1 score: ", f1_score(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("CM: ", confusion_matrix(y_test, y_pred))


# In[75]:


from sklearn.naive_bayes import GaussianNB
go_model = GaussianNB()
go_model.fit(X_train_scaled,y_train)

y_pred = go_model.predict(X_test_scaled)
print("Logistic Regression Model")
print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("F1 score: ", f1_score(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("CM: ", confusion_matrix(y_test, y_pred))


# In[77]:


df["DTI_Ratio_sq"] = df["DTI_Ratio"]**2
df["Credit_Score_sq"] = df["Credit_Score"] **2

X = df.drop(columns = ["DTI_Ratio","Credit_Score"])
y = df["Loan_Approved"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[78]:


# Logistic regression

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)

y_pred = log_model.predict(X_test_scaled)

# Evaluation
print("Logistic Regression Model")
print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("F1 score: ", f1_score(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("CM: ", confusion_matrix(y_test, y_pred))


# In[79]:


# KNN

from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

y_pred = knn_model.predict(X_test_scaled)

# Evaluation
print("KNN Model")
print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("F1 score: ", f1_score(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("CM: ", confusion_matrix(y_test, y_pred))


# In[80]:


# Naive Bayes

from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)

y_pred = nb_model.predict(X_test_scaled)

# Evaluation
print("Naive Bayes Model")
print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("F1 score: ", f1_score(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("CM: ", confusion_matrix(y_test, y_pred))


# In[ ]:




