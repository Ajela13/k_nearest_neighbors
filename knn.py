import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import io

df=pd.read_csv("breast_cancer_wisconsin_data_altered.csv") 

# view dimensions of dataset
df.shape
# preview the dataset
df.head()

#Clump Thickness: Assesses if cells are mono- or multi-layered.
#Unif Cell Size (Uniformity of Cell Size): Evaluates the consistency in size of the cells in the sample.
#Unif cell shape (Uniformity of Cell Shape): Estimates the equality of cell shapes and identifies marginal variances.
#Marg Adhesion (Marginal Adhesion): Quantifies how much cells on the outside of the epithelial tend to stick together.
#single_epith_cell_size (Single Epithelial Cell Size): Relates to cell uniformity, determines if epithelial cells are significantly enlarged.
#Bare Nuclei: Calculates the proportion of the number of cells not surrounded by cytoplasm to those that are.
#Blan chrom (Bland Chromatin): Rates the uniform "texture" of the nucleus in a range from fine to coarse.
#Norm nucleoli (Normal Nucleoli): Determines whether the nucleoli are small and barely visible or larger, more visible, and more plentiful.
#Mitoses: Describes the level of mitotic (cell reproduction) activity.

# view summary of dataset and missing values
df.info()

df.describe().round(2)

# drop Id column from dataset

df.drop('id', axis=1, inplace=True)

# check missing values in variables

df.isnull().sum()

# check frequency distribution of `Bare_Nuclei` column

df['bare_nuclei'].value_counts()
# impute missing values with median
for col in df.columns:
    col_median=df[col].median()
    df[col].fillna(col_median, inplace=True)

# view frequency distribution of values in `Class` variable
df['class'].value_counts()

# view percentage of frequency distribution of values in `Class` variable

df['class'].value_counts()/np.float(len(df))

#data visualization

plt.rcParams['figure.figsize']=(30,25)
df.plot(kind='hist', bins=10, subplots=True, layout=(5,2), sharex=False, sharey=False)
plt.show()

correlation = df.corr()

plt.figure(figsize=(18,8))
plt.title('Correlation of Attributes with Class variable')
a = sns.heatmap(correlation, square=True, annot=True, fmt='.2f', linecolor='white')
a.set_xticklabels(a.get_xticklabels(), rotation=90)
a.set_yticklabels(a.get_yticklabels(), rotation=30)           
plt.show()

correlation['class'].sort_values(ascending=False)

#Splitting data into test and train
X = df.drop(['class'], axis=1)

y = df['class']

# split X and y into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# check the shape of X_train and X_test

X_train.shape, X_test.shape

#Feature Scaling

cols = X_train.columns

#StandardScaler removes the mean and scales each feature/variable to unit variance.
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

print("Data type before standard scaler: ",type(X_train))

#We will scale the test data exactly same as how we have scaled the train data
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Data type after standard scaler: ",type(X_train))

#You will notice that data type has changed from "pandas dataframe" to numpy array after the conversion. Hence changing it back to dataframe
X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])

X_train.head()


# import KNeighbors ClaSSifier from sklearn
from sklearn.neighbors import KNeighborsClassifier

# instantiate the model (randomly picking the value of "K" as 3)
knn = KNeighborsClassifier(n_neighbors=3)
# fit the model to the training set
knn.fit(X_train, y_train)

#Predicting on train data
y_pred_train = knn.predict(X_train)

#Predicting on test data
y_pred_test = knn.predict(X_test)


from sklearn.metrics import accuracy_score

print('Model train accuracy score:', accuracy_score(y_train, y_pred_train).round(2))
print('Model test accuracy score:', accuracy_score(y_test, y_pred_test).round(2))

#Finding optimum value of K

#Setup arrays to store training and test accuracies
neighbors = np.arange(1,20)
train_accuracy =np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i,k in enumerate(neighbors):
    #Setup a knn classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=k)
    
    #Fit the model
    knn.fit(X_train, y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)
    
    #Compute accuracy on the test set
    test_accuracy[i] = knn.score(X_test, y_test) 


#Generate plot
plt.figure(figsize=(15,5))
plt.title('k-NN Varying number of neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()

#Accuracy of model has improved from 86% to 90% by optimizing the model parameter "K"

# instantiate the model
knn = KNeighborsClassifier(n_neighbors=9)
# fit the model to the training set
knn.fit(X_train, y_train)

print("Our train accuracy is: ",knn.score(X_train, y_train).round(2))
print("Our test accuracy is: ",knn.score(X_test, y_test).round(2))