import argparse
import os
import sklearn
from scipy.stats import ttest_rel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier 
import numpy as np
import csv
import argparse
import time
from scipy import stats 

# set the random state for reproducibility 
import numpy as np
np.random.seed(401)

classifiers = [SGDClassifier(max_iter = 10000), GaussianNB(), RandomForestClassifier(max_depth=5, n_estimators=10), MLPClassifier(alpha=0.05), AdaBoostClassifier()]
classifier_names = ["SDGClassifier", "GaussianNB", "RandomForestClassifier", "MLPCLassifier", "AdaBoostClassifier"]

def accuracy(C):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    numer = np.diag(C).sum()
    denom = C.sum()
    return numer/denom


def recall(C):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    numers = np.diag(C).sum()
    denoms = C.sum(axis = 1) #axis 1 is columns
    return numers/denoms

def precision(C):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    numers = np.diag(C).sum()
    denoms = C.sum(axis = 0) #axis 0 is rows
    return numers/denoms

def class31(output_dir, X_train, X_test, y_train, y_test):
    ''' This function performs experiment 3.1
    
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes

    Returns:      
       i: int, the index of the supposed best classifier
    '''
    print("This is part 3.1: ")
    
    with open(f"{output_dir}/a1_3.1.txt", "w") as outf:
        # For each classifier, compute results and write the following output:
        # start an accuracy list and accuracy dictionary to later find the best index
        accuracy_list = [0]*5
        acc_dict = {}

        # iterate through all the classifiers
        for (i,c) in enumerate(classifiers):

            # use each classifier to train
             c.fit(X_train, y_train)
             classifier_name = classifier_names[i]

             # obtain accuracy from the confusion matrix
             # add accuracy to accuracy list and save index to dictionary under the accuracy
             conf_matrix = confusion_matrix(y_test, c.predict(X_test))
             accuracy_curr = accuracy(conf_matrix)
             accuracy_list[i] = accuracy_curr
             acc_dict[accuracy_curr] = i

             # Computer recall and precision
             recall_curr = recall(conf_matrix)
             precision_curr = precision(conf_matrix)

             print("For classifier: ", classifier_name, " Accuracy: ", accuracy_curr)

             outf.write(f'Results for {classifier_name}:\n')  # Classifier name
             outf.write(f'\tAccuracy: {accuracy_curr:.4f}\n')
             outf.write(f'\tRecall: {[round(item, 4) for item in recall_curr]}\n')
             outf.write(f'\tPrecision: {[round(item, 4) for item in precision_curr]}\n')
             outf.write(f'\tConfusion Matrix: \n{conf_matrix}\n\n')
        pass

    # find best accuracy index and return
    best_acc = max(accuracy_list)
    iBest = acc_dict[best_acc]
    print(iBest)
    return iBest


def class32(output_dir, X_train, X_test, y_train, y_test, iBest):
    ''' This function performs experiment 3.2
    
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       iBest: int, the index of the supposed best classifier (from task 3.1)  

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    print("This is experiment 3.2: ")
        
    with open(f"{output_dir}/a1_3.2.txt", "w") as outf:
        classifier = classifiers[iBest]
        # For each number of training examples, compute results and write
        # the following output:
        #     outf.write(f'{num_train}: {accuracy:.4f}\n'))

        # iterate through all specified subsets
        for num_train in [1000, 5000, 10000, 15000, 20000]:
            print("num_train: ", num_train)

            #create subsets for training data
            X_train_sub = X_train[:num_train]
            y_train_sub = y_train[:num_train]

            # call best classifier and find best accuracy 
            classifier.fit(X_train_sub, y_train_sub)
            conf_matrix = confusion_matrix(y_test, classifier.predict(X_test))
            accuracy_curr = accuracy(conf_matrix)

            print(num_train,": ", accuracy_curr)
            outf.write(f'{num_train}: {accuracy_curr:.4f}\n')
        pass

    X_1k, y_1k = X_train[:1000], y_train[:1000]

    return (X_1k, y_1k)


def class33(output_dir, X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3
    
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    print("We are doing part 3.3 now")

    with open(f"{output_dir}/a1_3.3.txt", "w") as outf:
        # Prepare the variables with corresponding names, then uncomment
        # this, so it writes them to outf.
        
        # for each number of features k_feat, write the p-values for
        # that number of features:
        classifier = classifiers[i]
        acc_list = []

        for k_feat in [5, 50]:
            selector = SelectKBest(score_func=f_classif, k=k_feat)
            X_new = selector.fit_transform(X_train, y_train)
            p_values = selector.pvalues_
    
            outf.write(f'{k_feat} p-values: {[format(pval) for pval in p_values]}\n')

        for (train_x, train_y) in [(X_1k, y_1k), (X_train, y_train)]:
            selector = SelectKBest(score_func = f_classif, k = 5)               # k = 5

            x_trans_train = selector.fit_transform(train_x, train_y)          # reduced x train data
            x_trans_test = selector.transform(X_test)

            classifier.fit(x_trans_train, train_y)
            conf_matrix = confusion_matrix(y_test, classifier.predict(x_trans_test))

            acc = accuracy(conf_matrix)
            acc_list.append(acc)


        accuracy_1k = acc_list[0]
        accuracy_full = acc_list[1]

        ind_1k = set(SelectKBest(score_func=f_classif, k=5).fit(X_1k, y_1k).get_support(indices=True))
        ind_full = set(SelectKBest(score_func=f_classif, k=5).fit(X_train, y_train).get_support(indices=True))

        print("ind_1k: ",ind_1k)
        print("ind_full: ",ind_full)

        feature_intersection = ind_1k.intersection(ind_full)

        outf.write(f'Accuracy for 1k: {accuracy_1k:.4f}\n')
        outf.write(f'Accuracy for full dataset: {accuracy_full:.4f}\n')
        outf.write(f'Chosen feature intersection: {feature_intersection}\n')
        outf.write(f'Top-5 at higher: {ind_full}\n')
        pass


def class34(output_dir, X_train, X_test, y_train, y_test, X_data, y_data, i):
    ''' This function performs experiment 3.4
    
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
        '''
    print('We are doing 3.4 now')
    
    with open(f"{output_dir}/a1_3.4.txt", "w") as outf:

        kfold = KFold(n_splits = 5, shuffle = True, random_state = 401)
        kfold_accuracies = np.zeros((5,5))
        # Prepare kfold_accuracies, then uncomment this, so it writes them to outf.
        # for each fold:
        z = 0
        for (train_ind, test_ind) in kfold.split(X_data):
            x_train, y_train = X_data[train_ind], y_data[train_ind]
            x_test, y_test = X_data[test_ind], y_data[test_ind]
            j = 0
            for c in classifiers:
                c.fit(x_train, y_train)
                class_name = classifier_names[j]
                conf_matrix = confusion_matrix(y_test, c.predict(x_test))
                accuracy_curr = accuracy(conf_matrix)
                kfold_accuracies[z][j] = round(accuracy_curr,4)
                print("for class: ", class_name, " K fold #: ",z, " Accuracy: ", accuracy_curr) 
                j = j+1
            print("acc row: ", kfold_accuracies[z])

            outf.write(f'Kfold Accuracies: {kfold_accuracies[z]}\n')
            z = z + 1
        
        class_accs = np.array(kfold_accuracies).transpose()

        p_values = []
        for h in range(0,5):
            if h != i:
                p_values.append(stats.ttest_rel(class_accs[h], class_accs[i]).pvalue)

        p_values = np.array(p_values)
        print("p_vals: ", p_values)

        outf.write(f'p-values: {[round(pval, 4) for pval in p_values]}\n')
        
        pass


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    parser.add_argument(
        "-o", "--output_dir",
        help="The directory to write a1_3.X.txt files to.",
        default=os.path.dirname(os.path.dirname(__file__)))
    args = parser.parse_args()
    
    # TODO: load data and split into train and test.
    # TODO : complete each classification experiment, in sequence.

    feats = np.load(args.input)
    data = feats[feats.files[0]]

    # set the output directory
    output_dir = "part3"

    X_data = data[:, 0:173]
    y_data = data[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.2, random_state = 123, shuffle = True)

    print("X_train length and shape: ", len(X_train), X_train.shape)
    print("y_train length and shape: ", len(y_train), y_train.shape)
    print("x_test length and shape: ", len(X_test), X_test.shape)
    print("y_test length and shape: ", len(y_test), y_test.shape)

    x_1k, y_1k = X_train[:1000], y_train[:1000]
    #RUN EACH OF THESE STEP BY STEP OTHERWISE ITS GONNA TAKE FOREVER MAN

    #i_best = class31(output_dir, X_train, X_test, y_train, y_test)
    
    # we know i_best is ada_boost. thus 
    i_best = 4
    #x_1k, y_1k = class32(output_dir, X_train, X_test, y_train, y_test, i_best)

    #class33(output_dir, X_train, X_test, y_train, y_test, i_best, x_1k, y_1k)
    class34(output_dir, X_train, X_test, y_train, y_test, X_data, y_data, i_best)


    
