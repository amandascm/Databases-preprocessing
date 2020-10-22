import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def fileToList(filePath, separator):
	file = open(filePath, "r")
	lis = []
	for line in file:
		els = re.split(separator, line)
		els = list(filter(None, els))
		lis += els
	return lis

def listToDataframe(lis, numCols):
	cols = []
	for i in range(1, numCols+1):
		cols.append(i)
	df = pd.DataFrame(np.array(lis).reshape(-1,numCols), columns = cols)
	return df

def dropColumns(df, cols):
	newCols = []
	for i in range(1, (df.shape[1] - len(cols))+1):
		newCols.append(i)
	for c in cols:
		df.drop(c, inplace=True, axis=1)
	df.columns = newCols
	return df

def columnsToFloat(df, cols):
	for c in cols:
		df[c] = df[c].astype(float)
	return df

def columnsToInt(df, cols):
	for c in cols:
		df[c] = df[c].astype(int)
	return df

def generateReport(df, numFilePath, catFilePath, numAttributes, catAttributes):
	if len(numAttributes) > 0:
		numReport = {'min':df[numAttributes].min(), 'max':df[numAttributes].max(), 'mean':df[numAttributes].mean(), 'median':df[numAttributes].median(), 'var':df[numAttributes].var()}
		pd.DataFrame(data = numReport).to_csv(numFilePath)
	for cat in catAttributes:
		path = catFilePath + "cat" + str(cat) + ".csv"
		df[[cat]] = df[[cat]].astype('category')
		catReport = {'freq':df[[cat]].value_counts()}
		pd.DataFrame(data = catReport).to_csv(path)

def oneHotEncode(df, numCols, catCols):
	dataframe = df.loc[:, numCols]
	for c in catCols:
		toEncode = df.loc[:, c]
		encoded = pd.get_dummies(toEncode)
		newCols = []
		for i in encoded.columns:
			newCols += [str(i) + str(c) + str(c)]
		encoded.columns = newCols
		dataframe = dataframe.join(encoded)
	return dataframe

def replaceValueWithNan(df, valToReplace):
	return df.replace(valToReplace, np.nan, regex=True)

def normalizeDataframe(df):
	minVal = df.min()
	maxVal = df.max()
	rangeVal = maxVal - minVal
	normalized = (df - minVal) / rangeVal
	return normalized

def getXandY(df, labelCols, attributesCols):
	columns = df.shape[1]
	Y = df.loc[:, labelCols]
	X = df.loc[:, attributesCols]
	return X, Y

def getPCA(df, n):
	#pca = PCA(n_components=’mle’)
	pca = PCA(n_components = n)
	newDF = pd.DataFrame(pca.fit_transform(df))
	return newDF

def knn(x, X, n):
	neigh = NearestNeighbors(n_neighbors=n)
	neigh.fit(X)
	return neigh.kneighbors(x)

def replacementWithKNN(df):
	#rows
	for r in range(0, df.shape[0]):
		colsWithoutNan = []
		colsWithNan = []
		#columns
		for c in range(0, (df.shape[1])):
			#couldnt use notna() or isnull() with loc: could with iloc
			if pd.notna(df.iloc[r, c]):
				colsWithoutNan.append(c)
			else:
				colsWithNan.append(c)
		if(len(colsWithNan) > 0):
			for c in colsWithNan:
				#rows that have the same columns without nan values as the actual row (plus one column, which will replace cell value)
				counter = 0
				while(counter < len(colsWithNan)):
					if counter == 0: withoutNan = df.iloc[:, (colsWithoutNan+[c])].dropna()
					else: withoutNan = withoutNan = df.iloc[:, (colsWithoutNan[:-counter]+[c])].dropna()
					withoutNan.reset_index(drop=True, inplace=True)
					if(withoutNan.shape[0] > 0):
						break
					else:
						counter+=1
				#if there are rows to define nearest neighbor
				if counter < len(colsWithNan):
					#if only part of withouNan columns are being used as parameters: update withoutNan array
					if(counter>0):
						newColsWithoutNan = colsWithoutNan[:-counter]
					else:
						newColsWithoutNan = colsWithoutNan
					#actual row with not nan valued attributes
					rowWithoutNan = np.array(df.iloc[r, newColsWithoutNan]).reshape(1, len(newColsWithoutNan))
					
					#rows with same columns without nan values
					rowsWithoutNanColumns = np.array(withoutNan.iloc[:, [i for i in range(0,(len(newColsWithoutNan)))]]).reshape(withoutNan.shape[0], len(newColsWithoutNan))
					
					#get distance from this cell to the nearest neighbor and its index (in 'withoutNan' dataframe)			
					neighbor = knn(rowWithoutNan, rowsWithoutNanColumns, 1)

					#update dataframe with valid value (from the nearest neighbor)
					df.iloc[r, c] = withoutNan.iloc[(neighbor[1][0][0]), (withoutNan.shape[1]-1)]
				#if there aren't rows to find nearest neighbor: use median or default value (0)
				else:
					withoutNan = df.iloc[:, c].dropna()
					if(withoutNan.shape[0] > 0):
						df.iloc[r, c] = withoutNan.iloc[:, c].median()
					else:
						df.iloc[r, c] = 0
	return df