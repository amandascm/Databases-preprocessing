from preproc import *

#file to list
data = fileToList(filePath = "dataAndResults/heartdis/data/hungarian.data", separator = " |\n")
data.extend(fileToList(filePath = "dataAndResults/heartdis/data/long-beach-va.data", separator = " |\n"))
data.extend(fileToList(filePath = "dataAndResults/heartdis/data/switzerland.data", separator = " |\n"))

#list to dataframe
df = listToDataframe(lis = data, numCols = 76)

#drop columns and rename them (only 14 attributes were used):
#(age)(sex)(cp)(trestbps)(chol)(fbs)(restecg)(thalach)(exang)(oldpeak)(slope)(ca)(thal)(num) 
df = dropColumns(df = df, cols = [i for i in range(1,77) if i not in [3,4,9,10,12,16,19,32,38,40,41,44,51,58]])

#columns from string to float and int
df = columnsToFloat(df, [10])
df = columnsToInt(df = df, cols = [i for i in range(1,15)])
#print(df.dtypes)

#rename columns
df.columns = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","num"]

#replace -9 with Nan
df = replaceValueWithNan(df, -9)

#generate reports
#categorical attributes: "sex", "cp", "fbs", "restecg", "exang", "slope", "thal", "num"
#numeric attributes: "age", "trestbps", "chol", "thalach", "oldpeak", "ca"
generateReport(df = df, numFilePath = "dataAndResults/heartdis/reports/num.csv", catFilePath = "dataAndResults/heartdis/reports/", numAttributes = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"], catAttributes = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal", "num"])

#separate attributes from labels
X, Y = getXandY(df = df, labelCols = ["num"], attributesCols = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"])

#one hot encoder
X = oneHotEncode(df = X, numCols = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"], catCols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"])

#replace 2, 3 or 4 with 1 in the label column
for i in range(2,5):
    Y = Y.replace(i, 1, regex=True)

#normalize dataframe
X = normalizeDataframe(X)

#implement replacement Nan value politics
X = replacementWithKNN(X)

#PCA
X = getPCA(df = X, n = 10)

#new dataset (X + Y): can't join datasets if they have same columns names (it can happen after encoding categoric attributes, for example)
#Y.columns = ['y0', 'y1', 'y2', 'y3', 'y4']
X.columns = [i for i in range (1,11)]
dataset = X.join(Y)
dataset.to_csv("dataAndResults/heartdis/datasets/procHeartDis.csv")