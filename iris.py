from preproc import *

#csv to dataframe
df = pd.read_csv("dataAndResults/iris/data/iris.csv")

#generate reports
generateReport(df = df, numFilePath = "dataAndResults/iris/reports/num.csv", catFilePath = "dataAndResults/iris/reports/", numAttributes = [i for i in df.columns if i != 'Name'], catAttributes = ['Name'])

#separate attributes from labels
X, Y = getXandY(df = df, labelCols = ['Name'], attributesCols = [i for i in df.columns if i != 'Name'])

#one hot encoder
Y = oneHotEncode(df = Y, numCols = [], catCols = Y.columns)

#normalize dataframe
X = normalizeDataframe(X)

#new dataset (X + Y): can't join datasets if they have same columns names (it can happen after encoding categoric attributes, for example)
Y.columns = ["Iris-setosa","Iris-versicolor","Iris-virginica"]
dataset = X.join(Y)
dataset.to_csv("dataAndResults/iris/datasets/procIris.csv")