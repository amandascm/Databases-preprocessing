from preproc import *

#csv to dataframe
df = pd.read_csv("dataAndResults/mushrooms/data/mushrooms.csv")

#generate reports
generateReport(df = df, numFilePath = "dataAndResults/mushrooms/reports/num.csv", catFilePath = "dataAndResults/mushrooms/reports/", numAttributes = [], catAttributes = df.columns)

#separate attributes from labels
X, Y = getXandY(df = df, labelCols = ['class'], attributesCols = [i for i in df.columns if i != 'class'])

#replace ? with Nan
X = replaceValueWithNan(X, '?')

#one hot encoder (ignores Nan value)
Y = oneHotEncode(df = Y, numCols = [], catCols = ['class'])
X = oneHotEncode(df = X, numCols = [], catCols = X.columns)

#implement replacement Nan value politics
X = replacementWithKNN(X)

#PCA
X = getPCA(df = X, n = 10)

#new dataset (X + Y): can't join datasets if they have same columns names (it can happen after encoding categoric attributes, for example)
dataset = X.join(Y)
dataset.to_csv("dataAndResults/mushrooms/datasets/procMushrooms.csv")