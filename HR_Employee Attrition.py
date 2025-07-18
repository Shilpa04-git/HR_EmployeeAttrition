import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC
from imblearn.over_sampling import  RandomOverSampler
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest, chi2
import warnings;
warnings.filterwarnings('ignore')

df=pd.read_csv("C:/shilpa/pythonproject/HR_Employee.csv")
print(df.head())
print(df.shape)
print(df.isnull().sum())
print(df.dtypes)
print(df.nunique())
print(df.Attrition.value_counts()/len(df)*100)
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
print(df['Attrition'])
df=df.drop(['EmployeeNumber'],axis=1)
print(df.head())
df=df.drop(['EmployeeCount', 'Over18', 'StandardHours'],axis=1)
print(df.head())

##EDA and Visualization
pal_2 = sns.color_palette("GnBu",n_colors=2)
def count_Plot(feature, data,xl,yl,axs,hu=None):
    ax = sns.countplot(x='Attrition',palette=pal_2, data=data,hue=hu,ax=axs)
    for p in ax.patches:
         ax.annotate(f'\n{p.get_height()}',(p.get_x()+0.2,p.get_height()),  ha='center', va='center', size=18)
    axs.set(xlabel=xl, ylabel=yl)
def pie_plot(col, df, title, axs):
    co = df[col].value_counts()
    labels = co.index.tolist()  # Extract the category names
    values = co.values          # Extract the corresponding counts

    pal_2 = sns.color_palette("Set2")[0:len(values)]  # Adjust to number of slices
    axs.pie(values, labels=labels, colors=pal_2, autopct='%.0f%%')
    axs.set_title(title)
fig, axes = plt.subplots(1,2, figsize=(16, 6))
count_Plot("Attrition",df,"Employee Attrition in number","amount",axes[0])
pie_plot('Attrition',df,"Employee Attrtion in Percentage",axes[1])
plt.show()

##Uni Variate numerical Feature Analysis
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
sns.histplot(x=df['DailyRate'], kde=True, ax=axes[0, 0], color='lightseagreen')
axes[0, 0].set_xlabel('DailyRate')
sns.histplot(x=df['MonthlyIncome'], kde=True,ax=axes[0,1],color='steelblue')
axes[0,1].set_xlabel('MonthlyIncome')
sns.histplot(x=df['MonthlyRate'], kde=True,ax=axes[0,2],color='lightseagreen')
axes[0,2].set_xlabel('MonthlyRate')
sns.histplot(x=df['Age'], kde=True,ax=axes[1,0],color='lightseagreen')
axes[1,0].set_xlabel('Age')
sns.histplot(x=df['DistanceFromHome'], kde=True,ax=axes[1,1],color='steelblue')
axes[1,1].set_xlabel('DistanceFromHome')
sns.histplot(x=df['YearsAtCompany'], kde=True,ax=axes[1,2],color='steelblue')
axes[1,2].set_xlabel('YearsAtCompany')
plt.tight_layout()
plt.show

##Bi Variate Catagorical Feature Analysis
cata=[]
feat=df.drop(['Attrition'],axis=1)
pal_2 = sns.color_palette("GnBu",n_colors=2)
for colu in feat:
    pa =df[colu].value_counts().count()
    if (pa>1) & (pa<10) :
        cata.append(colu)
for col in cata:        
    fig, axes = plt.subplots(1,2, figsize=(10, 6))
    sns.barplot(x=col,y='Attrition',data=df,ax=axes[0],palette=pal_2)
    st ="No of Attrition in "+col
    count_Plot(col,df,st,"Count",axes[1],'Attrition')
plt.show()


##Multi Variate Feature Analysis
def show_grouped_means(df,by,value):
    print(df.groupby(by)[value].mean().to_frame(),'\n')
def show_grouped_status(df,by,value,func='mean'):
    if func=='mean':
        print(df.groupby(by)[value].mean().to_frame(),'\n')
    elif func=='median':
        print(df.groupby(by)[value].median().to_frame(),'\n')
show_grouped_means(df,by='Gender',value='MonthlyIncome')
show_grouped_status(df,by='Gender',value='MonthlyIncome',func='median')

plots=[
    ('box','Gender','MonthlyIncome'),
    ('box','Gender','Age'),
    ('reg','Age','MonthlyIncome'),
    ('joint','Age','MonthlyIncome'),
    ('swarm','Department','MonthlyIncome'),
    ('bar','Department','DistanceFromHome'),
    ('swarm','Education','HourlyRate'),
    ('violin','EducationField','DailyRate'),
    ('swarm','JobLevel','YearsAtCompany'),
    ('box','BusinessTravel','MonthlyIncome'),
    ('bar','JobRole','DailyRate'),
    ('box','JobSatisfaction','MonthlyIncome'),
    ('swarm','MaritalStatus','Age'),
    ('swarm','MaritalStatus','MonthlyIncome'),
]

for plot_type,x,y in plots:
    plt.figure(figsize=(16,6))
    if plot_type=='box':
        sns.boxplot(x=x,y=y,hue='Attrition',data=df,palette=pal_2)
    elif plot_type=='swarm':
        sns.swarmplot(x=x,y=y,hue='Attrition',data=df,palette=pal_2,size=3)
    elif plot_type=='bar':
        sns.barplot(x=x,y=y,hue='Attrition',data=df,palette=pal_2)
    elif plot_type=='violin':
        sns.violinplot(x=x,y=y,hue='Attrition',data=df,palette=pal_2)
    elif plot_type=='reg':
        sns.regplot(x=x,y=y,data=df,color='lightseagreen')
    elif plot_type=='joint':
        sns.jointplot(x=x,y=y,hue='Attrition',data=df,palette=pal_2)
        plt.show()
        continue
    plt.title(f'{y}vs{x}')
    plt.tight_layout()
   
    
summary_queries = [
    (['Gender'], 'Attrition'),
    (['Gender', 'Attrition'], 'MonthlyIncome'),
    (['Gender', 'Attrition'], 'MonthlyIncome', 'median'),
    (['Gender', 'Attrition'], 'Age'),
    (['Attrition'], 'Age'),
    (['Attrition'], 'MonthlyIncome'),
    (['Department'], 'Attrition'),
    (['Department', 'Attrition'], 'MonthlyIncome'),
    (['Education'], 'Attrition'),
    (['Education', 'Attrition'], 'HourlyRate'),
    (['EducationField'], 'Attrition'),
    (['EducationField', 'Attrition'], 'DailyRate'),
    (['JobLevel'], 'Attrition'),
    (['JobLevel', 'Attrition'], 'YearsAtCompany', 'median'),
    (['BusinessTravel'], 'Attrition'),
    (['BusinessTravel', 'Attrition'], 'MonthlyIncome', 'median'),
    (['JobSatisfaction'], 'Attrition'),
    (['JobSatisfaction', 'Attrition'], 'MonthlyIncome'),
    (['MaritalStatus'], 'Attrition'),
    (['MaritalStatus', 'Attrition'], 'Age'),
    (['MaritalStatus', 'Attrition'], 'MonthlyIncome'),
]

for query in summary_queries:
    if len(query) == 2:
        show_grouped_means(df, query[0], query[1])
    else:
        show_grouped_status(df, query[0], query[1], func=query[2])

               
##Defining Target and Independent Features        
y=df[['Attrition']]
x=df.drop(['Attrition'],axis=1)
print(x.head())

##Remove Features with Zero Variance
def unique_level(m):
    m = m.value_counts().count()
    return m
feature_val_count= pd.DataFrame(x.apply(lambda m: unique_level(m)))
print(feature_val_count)
feature_val_count.columns=['uni_level']
feat_level = feature_val_count.loc[feature_val_count['uni_level']>1]
feat_level_index = feat_level.index
X = x.loc[:,feat_level_index]
print(feat_level_index)

##Separate feature into numerical and categorical
num = x.select_dtypes(include='number')
print(num)
char = x.select_dtypes(include='object')
print(char)
feature_level_val = pd.DataFrame(num.nunique())
feature_level_val.columns = ['unique_level']
print(feature_level_val)


## All catagorical column with numical data
cat_feat = feature_level_val[feature_level_val['unique_level']<=20]
cat_fet_index = cat_feat.index
print(cat_fet_index)
cat_column = num[cat_fet_index]
print(cat_column.columns)

num=num.drop(columns=cat_fet_index)
numerical=num


## All catagorical column after separating
catagorical = pd.concat([char,cat_column],axis=1,join='inner')
print(catagorical)

##convert each category into a number 
categorical_cols = x.select_dtypes(include='object').columns
encoder = OrdinalEncoder()
x[categorical_cols] = encoder.fit_transform(x[categorical_cols])
x_encoded = pd.get_dummies(x, drop_first=True)

##Outlier Analysis of Numerical Features
print(numerical.describe(percentiles=[0.01,0.05,0.10,0.25,0.50,0.75,0.85,0.88,0.9,0.99]))

##Capping and Flooring of outliers
def outlier_cap(x):
    x=x.clip(lower=x.quantile(0.01))
    x=x.clip(upper=x.quantile(0.99))
    return(x)
numerical=numerical.apply(lambda x : outlier_cap(x))
print(numerical)
print(numerical.describe(percentiles=[0.01,0.05,0.10,0.25,0.50,0.75,0.85,0.88,0.9,0.99]))

## Checking corelation between numefical featutes
plt.figure(figsize=(18,18))
cmap =sns.color_palette("GnBu",n_colors=6)
cor =numerical.corr()
sns.heatmap(cor,annot=True,vmax=0.8,cmap=cmap,fmt='.2f',linecolor='green',linewidths=0.7,square=True)
plt.show()

##  function for removing corelated feature
def correlMatrix(data,thres):
    correlated_features = set()
    correlation_matrix = data.corr()
    for i in range(len(correlation_matrix .columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > thres:
                colname = correlation_matrix.columns[i]
                correlated_features.add(colname)
    return correlated_features

##  show corelated column if corelation more than 0.75
correlated_feat = correlMatrix(numerical,0.75)
print(correlated_feat)
numerical.drop(correlated_feat,axis=1,inplace=True)
print(numerical)

## Dropping Gender and PerformanceRating feature
catagorical.drop(columns=['Gender','PerformanceRating'],axis=1,inplace=True)
catagorical = catagorical.astype(object)
print(catagorical.head())

## Create dummy features with n-1 levels for all catagorical column
catag_dum = pd.get_dummies(catagorical,drop_first = True)
print(catag_dum.shape)
print(catag_dum)

## K Best for Selecting Categorical Features using k=20
selector = SelectKBest(chi2, k=20)
selector.fit_transform(catag_dum, y)
cols = selector.get_support(indices=True)
X_ca= catag_dum.iloc[:,cols]
print( catag_dum.columns[cols].tolist())

## joining all inpendent features
X_all=pd.concat([numerical,X_ca],axis=1,join='inner')
print(X_all.shape)
print(X_all)

##Imbalace dataset to balance dataset 
ros = RandomOverSampler(sampling_strategy='minority',random_state=1)
X_s,Y_s = ros.fit_resample(X_all, y)

##Split the Dataset for Training & Testing
X_train,x_test,y_train,y_test=train_test_split(X_s.values,Y_s.values,test_size=0.2,random_state=1)
print("Shape of Training Data",X_train.shape)
print("Shape of Testing Data",x_test.shape)
print("Attrition Rate in Training Data",y_train.mean())
print("Attrition Rate in Testing Data",y_test.mean())

##Data Transformation using Standardrization 
standard_Scaler=StandardScaler()
X_train = standard_Scaler.fit_transform(X_train)  
x_test = standard_Scaler.transform(x_test)

##function for training and evaluate model
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train,y_train)
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    tpr_score = metrics.precision_score(y_train, train_preds)
    trc_score = metrics.recall_score(y_train, train_preds)
    tac_score =metrics.accuracy_score(y_train,train_preds)
    print("For Training Dataset.")   
    print(f'Accuracy: {tac_score:.2f}, Precision: {tpr_score:.2f}, Recall: {trc_score:.2f}')
    pr_score = metrics.precision_score(y_test, test_preds)
    rc_score = metrics.recall_score(y_test, test_preds)
    ac_score = metrics.accuracy_score(y_test, test_preds)
    print("For Testing Dataset")
    print("F1:",metrics.f1_score(y_test, test_preds))
    print(f'Accuracy: {ac_score:.2f}, Precision: {pr_score:.2f}, Recall: {rc_score:.2f}')
    print("\n Classification Report:\n")
    print(classification_report(y_test, test_preds))
    cm = confusion_matrix(y_test, test_preds)
    print(cm)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='GnBu', xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
model = RandomForestClassifier(random_state=42)
print("Model Name : RandomForestClassifier:")
train_and_evaluate_model(model, X_train, y_train, x_test, y_test)

##Hyper-Parameter Optimization using GridSearchCV 
fit_rf = RandomForestClassifier(random_state=1)
start = time.time()
param_dist = {'max_depth': [7,8,9],
              'max_features': ['sqrt'],
              'criterion': ['gini','entropy'],
              'min_samples_split':[8,9,11,12],
              'min_samples_leaf':[8,9,11,13]}
cv_rf = GridSearchCV(fit_rf, cv = 10,
                     param_grid=param_dist, 
                     n_jobs = 3)
cv_rf.fit(X_train, y_train.ravel())
print('Best Parameters for randomforest classifier using grid search: \n', cv_rf.best_params_)

##Decision Tree Model
model_dt = DecisionTreeClassifier(random_state=11, max_depth=3, criterion='gini')
print("Model Name : DecisionTreeClassifier:")
train_and_evaluate_model(model_dt, X_train, y_train, x_test, y_test)


##Hyper-Parameter Optimization using GridSearchCV
dt_model2 = DecisionTreeClassifier(random_state=3)
param_dist = {'max_depth': [7,8,9],
              'min_samples_split':[9,11,15],
              'min_samples_leaf':[9,11,13],
              'criterion': ['gini']}
cv_rf = GridSearchCV(estimator=dt_model2, cv = 10,
                     param_grid=param_dist, 
                     n_jobs = 3)
cv_rf.fit(X_train, y_train)
print('Best Parameters foe decision tree using grid search: \n', cv_rf.best_params_)

##K-Nearest Neighbours Classifier model 
error = []
# Calculating error for K values between 1 and 40
for i in range(1, 40):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(x_test)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12, 6))  
plt.plot(range(1, 40), error, color='seagreen', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=8)
plt.title('Error Rate K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Error')  
plt.show()

classifier = KNeighborsClassifier(n_neighbors=6)  
print("Model Name : KNeighborsClassifier:")
train_and_evaluate_model(classifier, X_train, y_train, x_test, y_test)

##SVC Model 
svc_model = SVC(kernel='rbf', gamma='scale')
print("Model Name : SVC")
train_and_evaluate_model(svc_model,X_train,y_train,x_test,y_test)







