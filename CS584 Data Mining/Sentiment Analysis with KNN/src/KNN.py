import numpy as np
from scipy import sparse
from collections import Counter 


def tranform_matrix_to_class_labels(matrix):
    """
    Count the majority vote
    """
    
    return Counter(matrix).most_common(1)[0][0]



def knn_train(X_train, X_predict, y_train, y_test, k=3, debug=False, get_accuracy=False):
    """
    This method recieves the training set and the set to predict and produces the predictions.
    
    Input: 
        X_train: The training set matrix of size (m, n): where m is the number of training examples and n is the number of features
        y_predict: The prediction set matrix of size (m, n): where m is the number of training examples and n is the number of features
        y_train: The training labels vector of size (m, 1): where m is thumber of examples in the training set.
        k: The number of neighbors to consider
        
    Output:
        y_labels: a column vector of size m: where m is the number of training examples in the y_predict matrix. 
    """
    y = np.zeros((X_predict.shape[0], 1))
    
    if debug:
        print("\t  Squaring...")

    X_train_l2_norm = X_train.power(2).sum(axis=1) # Element wise square of the X_train and sum over the rows. 
    X_predict_l2_norm = X_predict.power(2).sum(axis=1) # Element wise square of the y_predict and sum over the rows. 
    
    if debug:
        print("\t  Find the quared root...")

    X_train_l2_norm = np.sqrt(X_train_l2_norm).reshape(-1,1) # Element wise squared root of the X_train_l2_norm
    X_predict_l2_norm = np.sqrt(X_predict_l2_norm).reshape(-1,1) # Element wise squared root of the X_train_l2_norm
    
    
    # Do dot products to to get l2 d1 * d2 for all the samples

    if debug:
        print("\t  Computing the dot product...")


    l2_norms = X_train_l2_norm.dot(X_predict_l2_norm.T) # Get the multiplication of l2 norms of the train the predict sets.
    l2_norms = sparse.csr_matrix(l2_norms)
    prod = X_train.dot(X_predict.T) # Multiply and sum up rows in the X_train and the X_predicts matrices    
   
    
    result = np.divide(prod, l2_norms) # Devide by the product of the l2 norms    
    
    
    result = np.squeeze(np.asarray(result)) * y_train # consolidate the dimentions.
    result = result.T # Transport the result to get it in a tabular format
    
    if debug:
        print("\t  Sorting to from most similar to less similar")

    for i in range(result.shape[0]):

        if debug:
            if (i > 0) & ((i % 1000) == 0):
                print(f"\t \t sorted {i} rows out of {result.shape[0]}")

        result[i] = sorted(result[i], key=abs, reverse=True) # sort the values with highest number
        
    all_predictions = result[:, 0:k] # Filter the number of neighbors
    
    predictions = np.where(all_predictions < 0, -1, 1) # change the predictions to positive of negative label
    
    if debug:
        print("\t Predicting labels... ")

    # Get the predicted labels    
    pred_labels = [tranform_matrix_to_class_labels(x) for x in predictions]
    
    # Reshape the predicted labels
    pred_labels = np.array(pred_labels).reshape(-1,1)

    accuracy = 0;
    if get_accuracy:
        # Calculate the accuracy
        accuracy = round(np.sum(pred_labels == y_test) / len(y_test) * 100, 2)
        print(f"\n\t Number of nearest neighbors: {k}")
        print(f"\t The accuracy is {accuracy}%\n")
    
    
    return pred_labels, accuracy
    
    
    