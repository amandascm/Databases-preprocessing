from preproc import *

#file to list
data = fileToList(filePath = "dataAndResults/heartdis/data/hungarian.data", separator = " |\n")
data.extend(fileToList(filePath = "dataAndResults/heartdis/data/long-beach-va.data", separator = " |\n"))
data.extend(fileToList(filePath = "dataAndResults/heartdis/data/switzerland.data", separator = " |\n"))

#list to dataframe
df = listToDataframe(lis = data, numCols = 76)

#drop columns and rename them
df = dropColumns(df = df, cols = [i for i in range(1,77) if i not in [3,4,9,10,12,16,19,32,38,40,41,44,51,58]])

#columns from string to float and int
df = columnsToFloat(df, [10])
df = columnsToInt(df = df, cols = [i for i in range(1,15)])
#print(df.dtypes)

#generate reports
generateReport(df = df, numFilePath = "dataAndResults/heartdis/reports/num.csv", catFilePath = "dataAndResults/heartdis/reports/", numAttributes = [i for i in range(1,14)], catAttributes = [14])

#separate attributes from labels
X, Y = getXandY(df = df, labelCols = [14], attributesCols = [i for i in range(1,14)])

#replace -9 with Nan
X = replaceValueWithNan(X, -9)

#one hot encoder
Y = oneHotEncode(df = Y, numCols = [], catCols = [14])

#normalize dataframe
X = normalizeDataframe(X)

#implement replacement Nan value politics
X = replacementWithKNN(X)

#PCA
X = getPCA(df = X, n = 10)

#new dataset (X + Y): can't join datasets if they have same columns names (it can happen after encoding categoric attributes, for example)
Y.columns = ['y0', 'y1', 'y2', 'y3', 'y4']
dataset = X.join(Y)
dataset.to_csv("dataAndResults/heartdis/datasets/procHeartDis.csv")