# USAGE
# run classify_multimodal1.py --images med_dataset/images --docs med_dataset/docs

# import the necessary packages
#from __future__ import print_function
from feature.localbinarypatterns import LocalBinaryPatterns
from feature.rgbhistogram import RGBHistogram
from feature.hog import HOG
from feature import dataset
#from feature import imutils
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics
#from sklearn.grid_search import GridSearchCV
import numpy as np
import argparse
import glob
import cv2
import paths1
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.datasets import load_files
from sklearn import metrics

'''
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required = True,
	help = "path to the image dataset")
ap.add_argument("-m", "--docs", required = True,
	help = "path to the document dataset")
args = vars(ap.parse_args())

imagePaths = sorted(paths1.list_images(args["images"]))
docPaths = sorted(paths1.list_images(args["docs"]))
'''	

# initialize the list of data and class label targets
data = []
target = []

##################Text Feature Extraction #######################
x = load_files('med_dataset\docs')

data_set1 = x['data']
data_set2 = x['filenames']
data_set3 = x['target']
data_set4 = x['target_names']

y = load_files('med_dataset\images')
imagePaths = y['filenames']
#print data_set5

vectorizer = CountVectorizer(stop_words='english',decode_error='ignore')
document_term_matrix = vectorizer.fit_transform(data_set1)
#print document_term_matrix.shape
#print vectorizer.vocabulary_
# {u'blue': 0, u'sun': 3, u'bright': 1, u'sky': 2}

tfidf = TfidfTransformer(norm="l2")
tfidf.fit(document_term_matrix)
#tfidf= TfidfTransformer(use_idf=False).fit(document_term_matrix)

#print "IDF:", tfidf.idf_
tf_idf_matrix = tfidf.transform(document_term_matrix)
#print tf_idf_matrix.todense()

############initialize the image descriptors#####################

# initialize the HOG descriptor  1800-D
hog = HOG(orientations = 18, pixelsPerCell = (10, 10),
	cellsPerBlock = (1, 1), transform = True)

# 512-D Histogram
rgb = RGBHistogram([8, 8, 8])

#26-D Histogram
lbp = LocalBinaryPatterns(24, 8)

count = 0
mat =tf_idf_matrix.todense()

########## loop over the image and doc paths####################
#for (imagePath, docPath) in zip(imagePaths, docPaths):
for imagePath in imagePaths:	
	'''
	######### load the document ##############
	file = open(docPath, 'r')

	docs_new = file.read()
	doc_term_matrix = vectorizer.transform(docs_new)
	doc_tf_idf_matrix  = tfidf.transform(doc_term_matrix)
	#print doc_tf_idf_matrix.todense()
	
	y = doc_tf_idf_matrix.todense()
	z = y[1,:] 
	text_array = np.squeeze(np.asarray(z))
	#print text_array.shape
	text_list = text_array.tolist()
	#print text_list
	'''
	
	z = mat[count,:] 
	text_array = np.squeeze(np.asarray(z))
	#print text_array.shape
	text_list = text_array.tolist()
	#print text_list
	
	count+=1
	
	##########load the image################### 
	image = cv2.imread(imagePath)
	# Resize image to 100 x 100 
	image = cv2.resize(image, (100, 100)) 
	
	# RGB Histogram
	rgb_hist = rgb.describe(image)
	
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#LBP feature
	lbp_hist = lbp.describe(gray)
	
	gray = dataset.deskew(gray, 100)
	gray = dataset.center_extent(gray, (100, 100))
	#HOG feature	
	hog_hist = hog.describe(gray)
	
	x = rgb_hist.tolist()
	y = lbp_hist.tolist()
	z = hog_hist.tolist()
	x.extend(y)
	x.extend(z)
	x.extend (text_list)
	combined =np.array(x)
	
	'''
	x = rgb_hist.tolist()
	y = lbp_hist.tolist()
	z = hog_hist.tolist()
	#x.extend(y)
	#y.extend(z)
	#y.extend (text_list)
	combined =np.array(z)
	'''
	####Only text
	#combined = text_array
	
	#print combined.shape
	data.append(combined)
	
	###label
	path_list = imagePath.split(os.sep)
	#print path_list[-2]
	target.append(path_list[-2])
	#target.append(imagePath.split("_")[-2])
	
# grab the unique target names and encode the labels
targetNames = np.unique(target)
le = LabelEncoder()
target = le.fit_transform(target)

# construct the training and testing splits
#(trainData, testData, trainTarget, testTarget) = train_test_split(data, target,
#	test_size = 0.2, random_state = 42)

(trainData, testData, trainTarget, testTarget) = train_test_split(data, target,
	test_size = 0.2)

#################Classification Training ##########################################

###Bayes Classifer
#model = MultinomialNB().fit(trainData, trainTarget)

###Random Forest classifier
#model = RandomForestClassifier(n_estimators = 25, random_state = 84)

###Linear SVM
#model = svm.SVC(gamma=0.001, C=100)
#model = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42).fit(trainData,trainTarget)

model = LinearSVC(C=100.0, random_state=42)  # good for doc

model.fit(trainData, trainTarget)

###Non Linear SVM
#model = svm.NuSVC()
#model.fit(trainData, trainTarget)

######################### evaluate the classifier######################################
'''
docs_new = ['Living cells expressing fusion proteins and immunostained fixed cells', 'Eccentric wall thickening of the sigmoid with filling defect. Note also the ovarian cysts']
test_document_term_matrix = vectorizer.transform(docs_new)
test_tf_idf_matrix  = tfidf.transform(test_document_term_matrix)

predicted = model.predict(test_tf_idf_matrix)
print predicted
'''


predicted = model.predict(testData)
#print predicted

print(classification_report(testTarget, predicted,
	target_names = targetNames))

print metrics.confusion_matrix(testTarget, predicted)
