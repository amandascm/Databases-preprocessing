from preproc import *

#csv to dataframe
df = pd.read_csv("dataAndResults/titanic/data/titanic_data.csv")

#generate reports
generateReport(df = df, numFilePath = "dataAndResults/titanic/reports/num.csv", catFilePath = "dataAndResults/titanic/reports/", numAttributes = ['PassengerId','Age','SibSp','Parch', 'Fare'], catAttributes = ['Survived','Pclass','Sex', 'Embarked', 'Cabin'])

#conclusion: non
df = dropColumns(df = df, cols = ['PassengerId','Name'])

#separate attributes from labels
X, Y = getXandY(df = df, labelCols = [1], attributesCols = [i for i in range(2,11)])

#cabin attribute: split in two
X['numCabin'] = X[9].apply(lambda x: ''.join(re.findall("[0-9]", x)) if pd.notna(x) else x)
X['catCabin'] = X[9].apply(lambda x: ''.join(re.findall("[A-Z]", x)) if pd.notna(x) else x)
X = dropColumns(df = X, cols = [9])
#numCabin = 8, catCabin = 9

#ticket attribute: split in two
X['numTicket'] = X[6].apply(lambda x: ''.join(re.findall("[0-9]", x)) if pd.notna(x) else x)
X['catTicket'] = X[6].apply(lambda x: ''.join(re.findall("[A-Z]", x)) if pd.notna(x) else x)
X = dropColumns(df = X, cols = [6])
#1 = Pclass, 2 = Sex, 3 = Age, 4 = SIbSp, 5 = parCh, 6 = Fare, 
#7 = Embarked, 8 = numCabin, 9 = catCabin, 10 = numTicket, 11 = catTicket 

#generate reports
#numCabin and numTicket have too many unavailable values
#generateReport(df = X, numFilePath = "dataAndResults/titanic/reports/num.csv", catFilePath = "dataAndResults/titanic/reports/", numAttributes = [3, 4, 5, 6, 8, 10], catAttributes = [1, 2, 7, 9, 11])
#print(len(list(X[[10]].value_counts()))) - we have 679 different values for numTicket and 891 instances with a valid value (100%)
#print(len(list(X[[8]].value_counts()))) - we have 98 different values for numCabin and only 204 instances with a vali value for numCabin (22%)
#print(len(list(X[[9]].value_counts()))) - we have 16 different values fot catCabin and only 204 instances with a valid value for catCabin (22%)
#print(((X[[11]].value_counts())).sum()) - we have 30 different values for catTicket and 891 instances with a valid value (100%)

#discard numCabin column
X = dropColumns(df = X, cols = [8])
#1 = Pclass, 2 = Sex, 3 = Age, 4 = SIbSp, 5 = parCh, 6 = Fare, 
#7 = Embarked, 8 = catCabin, 9 = numTicket, 10 = catTicket

#one hot encoder
X = oneHotEncode(df = X, numCols = [3, 4, 5, 6, 9], catCols = [1, 2, 7, 8, 10])

#normalize dataframe
#replace empty strings with Nan
X = replaceValueWithNan(X, '')
X[[3, 4, 5, 6, 9]] = normalizeDataframe(X[[3, 4, 5, 6, 9]].astype(float))

#implement replacement Nan value politics
X = replacementWithKNN(X)

#PCA
X = getPCA(df = X, n = 10)

#new dataset (X + Y): can't join datasets if they have same columns names (it can happen after encoding categoric attributes, for example)
Y.columns = ['surv']
X.columns = [1,2,3,4,5,6,7,8,9,10]
dataset = X.join(Y)
dataset.to_csv("dataAndResults/titanic/datasets/procTitanic.csv")