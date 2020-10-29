from preproc import *

#csv to dataframe
df = pd.read_csv("dataAndResults/breastcancer/data/wdbc.data", header = None)

#drop columns (ID number column to be droped)
df = dropColumns(df = df, cols = [0])

#generate reports
#categorical attributes: diagnosis (M or B)
generateReport(df = df, numFilePath = "dataAndResults/breastcancer/reports/num.csv", catFilePath = "dataAndResults/breastcancer/reports/", numAttributes = [i for i in range(2,32)], catAttributes = [1])

#separate attributes from labels
X, Y = getXandY(df = df, labelCols = [1], attributesCols = [i for i in range(2,32)])

#one hot encoder
Y = oneHotEncode(df = Y, numCols = [], catCols = [1])

#no missing values

#normalize dataframe
X = normalizeDataframe(X)

#PCA
X = getPCA(df = X, n = 10)

#new dataset (X + Y): can't join datasets if they have same columns names (it can happen after encoding categoric attributes, for example)
dataset = X.join(Y)
dataset.to_csv("dataAndResults/breastcancer/datasets/procBreastCancer.csv")