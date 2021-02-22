1. There are 2 pythons files.
	- The KNN.py contains a function that wraps the KNN algorithm I developped. The KNN algorithm is wrapped in the knn_train function. 
	- The KNN_RUN.py does everything from reading and preprocessing the data to running the KNN on the train data and the test data

2. Required libraries (I used latest version of these libraries):
	- numpy
	- scipy
	- re
	- sklearn
	- nltk
	- pickle
	- collections

3. How to run the code:
	- I have simplified how you can ran the code. You only need to run the "python KNN_run.py" in the command prompt/terminal. One thing to note is that I used python 3. 

4. Things to note while running the coding:
	- In the KNN_run.py file, there is a KNN_train_test() function. This function orchastrate everything. 

	- You should see print statements in the command prompt/terminal while running the code to give context of what is going one. 
	- After the code finish running, it will print the accuracy from the test set (should be around 76%)
	- If you set test_predict to True, then a file named predicted_sumit (which can be opened with any text editor) will be created with labels of the test set provided by the professor. This file will be
	  saved under the data folder in the src folder. 
