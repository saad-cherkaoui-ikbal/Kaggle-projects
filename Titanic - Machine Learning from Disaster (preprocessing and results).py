
# First, title extraction from name
import regex as re
def title_extract_alldata(data) :
    x = []
    for l in data['Name'] :
        x+=re.findall('[A-Z]\w*(?=\.)', l)
    if x.count('L') == 1 :
        x.remove('L')
    for n, i in enumerate(x):
        if x.count(i) < 130 :
            x[n] = 'Other'
    x = pd.DataFrame(x, columns=['Title'])   
    return pd.concat([data.reset_index(drop=True), x], axis=1)

alldata1 = title_extract_alldata(alldata)

def title_extract(data) :
    x = []
    for l in data['Name'] :
        x+=re.findall('[A-Z]\w*(?=\.)', l)
    if x.count('L') == 1 :
        x.remove('L')
    for n, i in enumerate(x):
        if i not in alldata1['Title'].unique() :
            x[n] = 'Other'
    x = pd.DataFrame(x, columns=['Title'])   
    return pd.concat([data.reset_index(drop=True), x], axis=1)
  
  # The first approach is to fit a logistic model on the data and then use the estimated probabilites to compute the weights of the individuals
  # Now that we have the weights, we can eliminate individuals with missing values and estimate a weighted logistic model
  
  # Dealing with missing values : variable Age (Repondération / Weights adjustment)

my_train_data1 = my_train_data.loc[:, ['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']]
age_answered = (~my_train_data1.Age.isnull()).astype(int) # 1 : answered, 0 : didn't answer
my_train_data1 = title_extract(my_train_data1)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def dummy_encode(data, sex, emb, pclass, title) :
    titles = data['Title'].unique()
    le_var = LabelEncoder()
    data.iloc[:, sex] = le_var.fit_transform(data.iloc[:, sex]) # male : 1, female : 0
    data.iloc[:, emb] = le_var.fit_transform(data.iloc[:, emb].astype(str)) # C : 0, Q : 1, S : 2
    data.iloc[:, title] = le_var.fit_transform(data.iloc[:, title])

    ohe_var = OneHotEncoder()
    hot = ohe_var.fit_transform(data.iloc[:, [pclass, emb]].values).toarray()
    
    hot1 = ohe_var.fit_transform(data.iloc[:, [title]].values).toarray()
    hot1 = pd.DataFrame(hot1, columns=list(titles))
    
    zeros_columns = []
    if len(alldata1['Title'].unique())-len(titles) != 0 :
        zeros_columns = list(alldata1['Title'].unique())
        for l in list(titles) :
            zeros_columns.remove(l)    
    zeros = pd.DataFrame(np.zeros((hot1.shape[0], len(alldata1['Title'].unique())-len(titles))), columns = zeros_columns)
    
    hot1 = pd.concat([hot1, zeros], axis=1)
    
    var = pd.concat([data.reset_index(drop=True), pd.DataFrame(hot, columns=['Pclass_1', 'Pclass_2', 'Pclass_3', 'Embarked_0', 'Embarked_1', 
                                                                                          'Embarked_2'])], axis=1).drop(['Pclass', 'Embarked',
                                                                                                                        'Pclass_1','Embarked_0'], axis=1)
    var = pd.concat([var, hot1], axis=1).drop(['Other', 'Name'], axis=1)
    
    return var

my_var = dummy_encode(my_train_data1, 2, 6, 0, 7).drop('Age', axis=1)

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)
classifier.fit(my_var, age_answered)
weights = classifier.predict_proba(my_var)[:, 1] 
weights = pd.DataFrame(1/weights[~my_train_data['Age'].isnull()], columns=['Weight']) # individual weights !!!!
#classifier.score(my_var, age_answered)

# Preprocessing the Data 

# we recover the 'Survived' variable and only keep the individuals with no 'Age' missing value
my_train_data1 = my_train_data.loc[:, ['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']]
my_train_data1 = title_extract(my_train_data1)

train_data = pd.concat([my_train_data['Survived'].reset_index(drop=True), my_train_data1.reset_index(drop=True)], axis=1)
train_data = train_data[~train_data['Age'].isnull()]
train_data = dummy_encode(train_data, 3, 7, 1, 8)

# Feature scaling (Age)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
train_data[['Age']] = sc.fit_transform(train_data[['Age']])

# add the weights to the set
train_data = pd.concat([train_data, weights], axis=1)

# we do the same for the CV and test set

my_CV_data1 = my_CV_data.loc[:, ['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']]
CV_data = title_extract(my_CV_data1)
CV_data = dummy_encode(CV_data, 3, 7, 1, 8) 

my_test_data1 = test.loc[:, ['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']]
test_data = title_extract(my_test_data1)
test_data = dummy_encode(test_data, 2, 6, 0, 7)

# Feature scaling (Age)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
CV_data[['Age']] = sc.fit_transform(CV_data[['Age']])
test_data[['Age']] = sc.fit_transform(test_data[['Age']])


from missingpy import MissForest

# Make an instance and perform the imputation
imputer = MissForest(random_state=0)
my_imp = imputer.fit(train_data.drop(['Survived', 'Weight'], axis=1))

CV_data_missforest = imputer.transform(CV_data.drop('Survived', axis=1))
CV_data_missforest = pd.DataFrame(CV_data_missforest, columns=CV_data.columns[1:])
CV_data_missforest = pd.concat([CV_data.Survived, CV_data_missforest], axis=1)

test_data_missforest = imputer.transform(test_data)
test_data_missforest = pd.DataFrame(test_data_missforest, columns=test_data.columns)

## Now that the individuals in the training set have their new weights, and the missing values in the cross-validation and test set have been imputed
## using the MissForest imputation method, we will now fit the logistic model in R since Python doesn't allow for fitting a weighted model

train_data.to_excel(r'train_data.xlsx', index = False)
CV_data_missforest.to_excel(r'CV_data.xlsx', index = False)
test_data_missforest.to_excel(r'test_data.xlsx', index = False)

## The code for the estimation is in a separate file

# The second approach is the fitting of a Random Forest

# this time we'll keep the individuals with missing values
my_train_data1 = my_train_data.loc[:, ['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']]
my_train_data1 = title_extract(my_train_data1)
train_data = pd.concat([my_train_data['Survived'].reset_index(drop=True), my_train_data1.reset_index(drop=True)], axis=1)
train_data = dummy_encode(train_data, 3, 7, 1, 8)

# Feature scaling (Age)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
train_data[['Age']] = sc.fit_transform(train_data[['Age']])

from missingpy import MissForest

# Make an instance and perform the imputation
imputer = MissForest(random_state=0)
train_data = pd.DataFrame(imputer.fit_transform(train_data.drop(['Survived'], axis=1)), columns=train_data.columns[1:])

# we do the same for the CV and test set

my_CV_data1 = my_CV_data.loc[:, ['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']]
CV_data = title_extract(my_CV_data1)
CV_data = dummy_encode(CV_data, 3, 7, 1, 8) 

my_test_data1 = test.loc[:, ['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']]
test_data = title_extract(my_test_data1)
test_data = dummy_encode(test_data, 2, 6, 0, 7)

# Feature scaling (Age)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
CV_data[['Age']] = sc.fit_transform(CV_data[['Age']])
test_data[['Age']] = sc.fit_transform(test_data[['Age']])

train_data = pd.concat([my_train_data['Survived'].reset_index(drop=True), train_data.reset_index(drop=True)], axis=1)

from missingpy import MissForest

# Make an instance and perform the imputation
imputer = MissForest(random_state=0)
my_imp = imputer.fit(train_data.drop(['Survived'], axis=1))

CV_data_missforest = imputer.transform(CV_data.drop('Survived', axis=1))
CV_data_missforest = pd.DataFrame(CV_data_missforest, columns=CV_data.columns[1:])
CV_data_missforest = pd.concat([CV_data.Survived, CV_data_missforest], axis=1)

test_data_missforest = imputer.transform(test_data)
test_data_missforest = pd.DataFrame(test_data_missforest, columns=test_data.columns)

## The model :
from sklearn.ensemble import RandomForestClassifier

# Make an instance and fit the model on the training set
classifier = RandomForestClassifier(n_estimators = 1000, random_state = 42)
classifier.fit(train_data.iloc[:, 1:10], train_data.Survived)

CV_pred = classifier.predict(CV_data_missforest.iloc[:, 1:10])

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(CV_data_missforest.Survived, CV_pred) # Confusion matrix of the Random Forest model
