from preproc import *

#csv to dataframe
df = pd.read_csv("dataAndResults/university/data/College_Data")

#drop columns
df = dropColumns(df = df, cols = [df.columns[0]]) #column with universities's names

#generate reports
generateReport(df = df, numFilePath = "dataAndResults/university/reports/num.csv", catFilePath = "dataAndResults/university/reports/", numAttributes = [df.columns[i] for i in range(1, df.shape[1])], catAttributes = [df.columns[0]])

#separate attributes from labels
X, Y = getXandY(df = df, labelCols = [df.columns[0]], attributesCols = [df.columns[i] for i in range(1, df.shape[1])])

#one hot encoder
Y = oneHotEncode(df = Y, numCols = [], catCols = [Y.columns])

#normalize dataframe
X = normalizeDataframe(X)

#PCA
X = getPCA(df = X, n = 10)

#new dataset (X + Y): can't join datasets if they have same columns names (it can happen after encoding categoric attributes, for example)
dataset = X.join(Y)
dataset.to_csv("dataAndResults/university/datasets/procUniversity.csv")