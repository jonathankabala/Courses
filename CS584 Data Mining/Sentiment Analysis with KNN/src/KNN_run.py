import numpy as np
from scipy import sparse
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords
import pickle

from collections import Counter
import KNN


print("\n************************* Jonathan Mbuya **************************")
print("************************* Miner's Name: Aaelh **************************")

print("\n => Reading data ...")

def my_split(el):
    """
        This function accepts a string and remove the \n (new line) and the \t (tab)
        Returns: returns a string with no new line nore tab
    """
    return el.strip('\n').split("\t")

def read_file(file):
    """
    This function reads the file with data
    
    Returns: returns a list of all the data in the file. 
    
    """
    with open(file, "r") as fn:
        return list(map(my_split, fn.readlines())) 


def read_format_file(file):
    
    """
    This function reads the format file
    
    Returns: returns a list of all the data in the file. 
    
    """
    with open(file, "r") as fn:
        return list(map(str.strip, fn.readlines())) 


# Get the files with data. 
train_file = "../data/train.dat"
predict_file = "../data/test.dat"
#format_file = "../data/format.dat"

# Read the data in the files
train_org = read_file(train_file)
predict_org = read_file(predict_file)
#format_org = read_format_file(format_file)

# Transform the lists to numpy arrays
train_org = np.array(train_org)
predict_org = np.array(predict_org)
#format_org = np.array(format_org)


""" 
	=================================================
	DATA PREPROCESSING
	=================================================
			
"""
print(" => Preprocessing data ... ")

def text_clearn_up(text_array):

	"""
	This method contains code to clean each sample in the train data and the predicted data

	"""

	text_array = text_array.lower()

	text_array = "".join(filter(lambda x: not x.isdigit(), text_array))

    # keep only letters and remove the rest
	text_array = re.sub("[^A-Za-z]+", " ", text_array)

	text_array = text_array.strip()

	return text_array

    

def text_clean_up_vectorized():
	"""
	This method calls the text_clean_up method to clean all the text in the entire document

	"""	
	return np.vectorize(text_clearn_up)


 # Train matrix
train_shape = train_org.shape # Get the shape of the train matrix

train = np.zeros(train_shape).astype(object) # Create the train matrix and initialize it with zeros
train[:,0] = train_org[:,0] # Fill the first column of the train matrix with -1 and +1 from the train_org
train[:,1] = text_clean_up_vectorized()(train_org[:,1]) # Fill the second column of the train matrix with texts from the train_org. We also clean the text using the text_clean_up_vectorized() function

# Features to predict labels
predict = text_clean_up_vectorized()(predict_org).reshape(-1,) # Clean the texts in the predict matrix

# get X (independent variable) and y (dependent variable.)
X, y = train[:,1], train[:,0].astype(int)


def KNN_train_test(vectorizer, X, y, predict, test_predict=False, run_svd=False):

	"""
	This function runs the KNN experiment for the sake of Homework 1.

	Arguments:
		vectorizer: the vertorizer object to tranform the text document into a document matrix (I used TfidfVectorizer from sklearn )
		X: (sparse matrix from scipy) feature vectors (a colum vector of review texts)
		y: (numpy array) classes (+1, -1 to show either a review is positive or negative)
		predict: (sparse matrix from scipy) contains the reviews to predict in a numpy array
		test_predict: If set to True, this function will also predict the label of the test set provided by the professor
		run_svd: If set to True, this function will run TrancateSVD to extract feature. If false, extracted features will be read
				 from the .plk files found in the data folder

	
	"""

	print(" => Running KKN ... ")
	
	vectorizer = vectorizer # creating a vector object
	X_martrix = vectorizer.fit_transform(X) # transfor the features into a document matrix

	# Split the data into train and test sets
	X_train, X_test, y_train, y_test = train_test_split(X_martrix, y, test_size=0.15, random_state=42)

	y_train  = y_train.reshape(-1,1).astype(int) # Tranform y_train into a 2D array
	y_test  = y_test.reshape(-1,1).astype(int) # Tranform y_testinto a 2D array


	print("\t Shapes before Feature extraction")

	print(f"\t\t X_train shape: {X_train.shape}")
	print(f"\t\t X_test shape: {X_test.shape}")
	print(f"\t\t y_train shape: {y_train.shape}")
	print(f"\t\t y_test shape: {y_test.shape} \n")

	
	X_train_svd = None
	X_test_svd = None

	# if run_svd is True, then extract feature from the X_train and y_train using TruncateSVD
	# if run_svd is False, then read these values from the data folder in plk files. 
	if run_svd:

		print(" \tExtracting features with TruncatedSVD...")
		svd = TruncatedSVD(n_components=1500)
		svd.fit(X_train)

		X_train_svd = svd.transform(X_train)
		X_test_svd = svd.transform(X_test)

		X_train_svd = sparse.csr_matrix(X_train_svd)
		X_test_svd = sparse.csr_matrix(X_test_svd)


		pickle.dump(X_train_svd, open(f"data/X_train_svd.plk", "wb"))
		pickle.dump(X_test_svd, open(f"data/X_test_svd.plk", "wb"))

	else:
		X_train_svd = pickle.load(open("data/X_train_svd.plk", "rb"))
		X_test_svd = pickle.load(open("data/X_test_svd.plk", "rb"))
	

	print("\tShapes after Feature extraction")

	print(f"\t\t X_train shape: {X_train_svd.shape}")
	print(f"\t\t X_test shape: {X_test_svd.shape}\n")

	print(" => Training and predicting values...")
	
	print("\tValidating on test data")
	#Train on the test values
	pred_labels, accuracy_test = KNN.knn_train(X_train_svd, X_test_svd, y_train, y_test, k=25, debug=False, get_accuracy=True)



	if test_predict:

		print(" => Predicting unseen labels")
		# Predicting the labels		
		# Predict unseen values (from the test file)
		predicted_labels, accuracy_predict = KNN.knn_train(X_train_svd, X_test_svd, y_train, y_test, k=25, debug=False, get_accuracy=False)
		# Trainform the predicted labels into -1 and +1 to match the format file. 
		prediction_to_submit = np.where(predicted_labels < 0, "-1", "+1")

		#save the predicted labels to the data file. 
		np.savetxt('data/predicted_sumit', prediction_to_submit, fmt="%s")
	
	
	print(" *************************** Finished! **********************************\n")
	


# Create a Term Frequence Inverse Document Frequency object with bi-grams using english stop works from nltk
tfidf = TfidfVectorizer(stop_words=stopwords.words('english'), ngram_range=(1,2))

# Run the KNN algorithm on provided data
KNN_train_test(tfidf, X, y,predict, test_predict=False, run_svd=False) 


