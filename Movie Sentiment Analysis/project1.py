"""EECS 445 - Fall 2023.

Project 1
"""

import itertools
import string
import warnings

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from helper import *
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC, LinearSVC

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)

np.random.seed(445)



def extract_word(input_string):
    """Preprocess review text into list of tokens.

    Convert input string to lowercase, replace punctuation with spaces, and split along whitespace.
    Return the resulting array.

    E.g.
    > extract_word("I love EECS 445. It's my favorite course!")
    > ["i", "love", "eecs", "445", "it", "s", "my", "favorite", "course"]

    Input:
        input_string: text for a single review
    Returns:
        a list of words, extracted and preprocessed according to the directions
        above.
    """
    for i in input_string:
        if i in string.punctuation:
            input_string = input_string.replace(i,' ')

    out = [x.lower() for x in input_string.split()]
    for i in out:
        if(i == ''):
            out.remove(i)
    return out

def extract_dictionary(df):
    """Map words to index.

    Reads a pandas dataframe, and returns a dictionary of distinct words
    mapping from each distinct word to its index (ordered by when it was
    found).

    E.g., with input:
        | reviewText                    | label | ... |
        | It was the best of times.     |  1    | ... |
        | It was the blurst of times.   | -1    | ... |

    The output should be a dictionary of indices ordered by first occurence in
    the entire dataset:
        {
           it: 0,
           was: 1,
           the: 2,
           best: 3,
           of: 4,
           times: 5,
           blurst: 6
        }
    The index should be autoincrementing, starting at 0.

    Input:
        df: dataframe/output of load_data()
    Returns:
        a dictionary mapping words to an index
    """
    word_dict = {}
    index = 0
    for x in df.iloc[:, 0]:
        words = extract_word(x)
        for i in words:
            if i=='':
                continue
            if i not in word_dict:
                word_dict[i] = index
                index+=1
    return word_dict

def generate_feature_matrix(df, word_dict):
    """Create matrix of feature vectors for dataset.

    Reads a dataframe and the dictionary of unique words to generate a matrix
    of {1, 0} feature vectors for each review. For each review, extract a token
    list and use word_dict to find the index for each token in the token list.
    If the token is in the dictionary, set the corresponding index in the review's
    feature vector to 1. The resulting feature matrix should be of dimension
    (# of reviews, # of words in dictionary).

    Input:
        df: dataframe that has the text and labels
        word_dict: dictionary of words mapping to indices
    Returns:
        a numpy matrix of dimension (# of reviews, # of words in dictionary)
    """
    number_of_reviews = df.shape[0]
    number_of_words = len(word_dict)
    feature_matrix = np.zeros((number_of_reviews, number_of_words))
    # TODO: Implement this function
    df_texts = df['reviewText']
    for x in range(number_of_reviews):
        df_str = df_texts[x]
        words = extract_word(df_str)
        for i in words:
            if i in word_dict:
                feature_matrix[x][word_dict[i]] = 1

    return feature_matrix

def performance(y_true, y_pred, metric="accuracy"):
    """Calculate performance metrics.

    Performance metrics are evaluated on the true labels y_true versus the
    predicted labels y_pred.

    Input:
        y_true: (n,) array containing known labels
        y_pred: (n,) array containing predicted scores
        metric: string specifying the performance metric (default='accuracy'
                 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                 and 'specificity')
    Returns:
        the performance as an np.float64
    """
    # TODO: Implement this function
    # This is an optional but very useful function to implement.
    # See the sklearn.metrics documentation for pointers on how to implement
    # the requested metrics.
    if metric == "accuracy":
        return metrics.accuracy_score(y_true, y_pred)
    elif metric == "precision":
        return metrics.precision_score(y_true, y_pred)
    elif metric == "sensitivity":
        return metrics.recall_score(y_true, y_pred)
    elif metric == "specificity":
        return metrics.recall_score(y_true, y_pred, pos_label=-1)
    elif metric == "f1-score":
        return metrics.f1_score(y_true, y_pred)
    elif metric == "auroc":
        return metrics.roc_auc_score(y_true, y_pred)

def cv_performance(clf, X, y, k=5, metric="accuracy"):
    """Split data into k folds and run cross-validation.

    Splits the data X and the labels y into k-folds and runs k-fold
    cross-validation: for each fold i in 1...k, trains a classifier on
    all the data except the ith fold, and tests on the ith fold.
    Calculates and returns the k-fold cross-validation performance metric for
    classifier clf by averaging the performance across folds.
    Input:
        clf: an instance of SVC()
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy'
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
    Returns:
        average 'test' performance across the k folds as np.float64
    """
    # TODO: Implement this function
    # HINT: You may find the StratifiedKFold from sklearn.model_selection
    # to be useful
    # Put the performance of the model on each fold in the scores array
    folds = StratifiedKFold(n_splits = k, shuffle = False)
    scores = []
    for train_i, test_i in folds.split(X,y):
        X_train = X[train_i]
        X_test = X[test_i]
        y_train = y[train_i]
        y_test = y[test_i]
        clf.fit(X_train, y_train)
        if metric == 'auroc':
            y_pred = clf.decision_function(X_test)
        else:
            y_pred = clf.predict(X_test)
        score = performance(y_test, y_pred, metric)
        scores.append(score)

    return np.array(scores).mean()

def select_param_linear(
    X, y, k=5, metric="accuracy", C_range=[], loss="hinge", penalty="l2", dual=True
):
    """Search for hyperparameters from the given candidates of linear SVM with
    best k-fold CV performance.

    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.
    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy',
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
        C_range: an array with C values to be searched over
        loss: string specifying the loss function used (default="hinge",
             other option of "squared_hinge")
        penalty: string specifying the penalty type used (default="l2",
             other option of "l1")
        dual: boolean specifying whether to use the dual formulation of the
             linear SVM (set True for penalty "l2" and False for penalty "l1")
    Returns:
        the parameter value for a linear-kernel SVM that maximizes the
        average 5-fold CV performance.
    """
    # TODO: Implement this function
    # HINT: You should be using your cv_performance function here
    # to evaluate the performance of each SVM
    best_performance = -float('inf')
    best_C = None

    for c in C_range:
        clf = LinearSVC(penalty=penalty,loss=loss,dual=dual,C=c, random_state=445)
        score = cv_performance(clf, X, y, k, metric)
        if score > best_performance:
            best_performance = score
            best_C = c
    return best_performance, best_C

def plot_weight(X, y, penalty, C_range, loss, dual):
    """Create a plot of the L0 norm learned by a classifier for each C in C_range.

    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        penalty: string for penalty type to be forwarded to the LinearSVC constructor
        C_range: list of C values to train a classifier on
        loss: string for loss function to be forwarded to the LinearSVC constructor
        dual: whether to solve the dual or primal optimization problem, to be
            forwarded to the LinearSVC constructor
    Returns: None
        Saves a plot of the L0 norms to the filesystem.
    """
    norm0 = []
    # TODO: Implement this part of the function
    # Here, for each value of c in C_range, you should
    # append to norm0 the L0-norm of the theta vector that is learned
    # when fitting an L2- or L1-penalty, degree=1 SVM to the data (X, y)
    for c in C_range:
        clf = LinearSVC(C=c, penalty=penalty, loss=loss, dual=dual, random_state=445)
        clf.fit(X, y)
        l0 = np.sum(clf.coef_ != 0)
        norm0.append(l0)
    plt.plot(C_range, norm0)
    plt.xscale("log")
    plt.legend(["L0-norm"])
    plt.xlabel("Value of C")
    plt.ylabel("Norm of theta")
    plt.title("Norm-" + penalty + "_penalty.png")
    plt.savefig("Norm-" + penalty + "_penalty.png")
    plt.close()

def select_param_quadratic(X, y, k=5, metric="accuracy", param_range=[]):
    """Search for hyperparameters from the given candidates of quadratic SVM
    with best k-fold CV performance.

    Sweeps different settings for the hyperparameters of a quadratic-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.
    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy'
                 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                 and 'specificity')
        param_range: a (num_param, 2)-sized array containing the
            parameter values to search over. The first column should
            represent the values for C, and the second column should
            represent the values for r. Each row of this array thus
            represents a pair of parameters to be tried together.
    Returns:
        The parameter values for a quadratic-kernel SVM that maximize
        the average 5-fold CV performance as a pair (C,r)
    """
    # TODO: Implement this function
    # Hint: This will be very similar to select_param_linear, except
    # the type of SVM model you are using will be different...
    best_C_val, best_r_val = 0.0, 0.0
    best_performance = -float('inf')
    scores = []

    for c,r in param_range:
        clf = SVC(C = c, kernel='poly', degree = 2, coef0=r, gamma='auto')
        score = cv_performance(clf, X, y, k=k, metric=metric)
        scores.append(score)
        if score > best_performance:
            best_performance = score
            best_C_val = c
            best_r_val = r
    return best_C_val, best_r_val, scores

def train_word2vec(fname):
    """
    Train a Word2Vec model using the Gensim library.
    First, iterate through all reviews in the dataframe, run your extract_word() function on each review, and append the result to the sentences list.
    Next, instantiate an instance of the Word2Vec class, using your sentences list as a parameter.
    Return the Word2Vec model you created.
    """
    df = load_data(fname)
    sentences = []
    for review in df['reviewText']:
        sentences.append(extract_word(review))
    model = Word2Vec(sentences=sentences, workers=1)
    return model

def compute_association(fname, w, A, B):
    """
    Inputs:
        - fname: name of the dataset csv
        - w: a word represented as a string
        - A and B: sets that each contain one or more English words represented as strings
    Output: Return the association between w, A, and B as defined in the spec
    """
    model = train_word2vec(fname)

    # First, we need to find a numerical representation for the English language words in A and B
    # TODO: Complete words_to_array(), which returns a 2D Numpy Array where the ith row is the embedding vector for the ith word in the input set.
    def words_to_array(set):
        arr = []
        for word in set:
            word_embedding = model.wv[word]
            arr.append(word_embedding)
        return arr
    # TODO: Complete cosine_similarity(), which returns a 1D Numpy Array where the ith element is the cosine similarity
    #      between the word embedding for w and the ith embedding in the array representation of the input set
    def cosine_similarity(set):
        w_embedding = model.wv[w]
        array = words_to_array(set)
        dot_product = np.dot(np.array(array), w_embedding)
        norm_w = np.linalg.norm(w_embedding)
        norm_set = np.linalg.norm(array,axis=1)
        cos_arr = dot_product/(norm_set*norm_w)
        return cos_arr

    # TODO: Return the association between w, A, and B.
    #      Compute this by finding the difference between the mean cosine similarity between w and the words in A, and the mean cosine similarity between w and the words in B
    association = np.mean(cosine_similarity(A)) - np.mean(cosine_similarity(B))
    return association

def main():
    # Read binary data
    # NOTE: THE VALUE OF dictionary_binary WILL NOT BE CORRECT UNTIL YOU HAVE IMPLEMENTED
    #       extract_dictionary, AND THE VALUES OF X_train, Y_train, X_test, AND Y_test
    #       WILL NOT BE CORRECT UNTIL YOU HAVE IMPLEMENTED extract_dictionary AND
    #       generate_feature_matrix
    fname = "data/dataset.csv"
    X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data(
        fname="data/dataset.csv"
    )
    IMB_features, IMB_labels, IMB_test_features, IMB_test_labels = get_imbalanced_data(
        dictionary_binary, fname="data/dataset.csv"
    )

    # TODO: Questions 2, 3, 4, 5
    #2a
    print(extract_word("It's a test sentence! Does it look CORRECT?"))

    #2b
    print(len(dictionary_binary))

    #2c
    total = 0
    for review in X_train:
        total+=sum(review)
    print("{:.4f}".format(total/len(X_train)))

    appearance_count = []
    for i in range(len(X_train[0])):
        appearance_count.append(sum(X_train[:,i]))
    for key, value in dictionary_binary.items():
        if value == appearance_count.index(max(appearance_count)):
            print(key)

    #3.1b
    print("\nQuestion 3.1b")
    C_range = [0.001,0.01,0.1,1,10,1000]
    metric = ["accuracy", "f1-score","auroc","precision","sensitivity","specificity"]
    for m in metric:
        best_performance, best_C = select_param_linear(X_train,Y_train,5,m,C_range,"hinge")
        print("(",m,", best_C): ","{:.4f}".format(best_performance), "{:.4f}".format(best_C))

    #3.1c
    print("\nQuestion 3.1c")
    clf = LinearSVC(C=0.1, loss='hinge', random_state=445)
    for m in metric:
        scores = []
        clf.fit(X_train, Y_train)
        
        if(m == 'auroc'):
            y_pred = clf.decision_function(X_test)
        else:
            y_pred = clf.predict(X_test)
        score = performance(Y_test, y_pred, m)
        print(m, ": ", "{:.4f}".format(score))

    #3.1d
    C_range = [0.001, 0.01, 0.1, 1]
    plot_weight(X_train, Y_train, penalty='l2', C_range=C_range, loss='hinge', dual=True)

    #3.1e
    clf = LinearSVC(C=0.1, loss='hinge', random_state=445)
    clf.fit(X_train, Y_train)
    coefficients = clf.coef_[0]

    top_positive_indices = coefficients.argsort()[-5:][::-1]
    top_negative_indices = coefficients.argsort()[:5]

    top_positive_words = ['' for _ in range(5)]
    top_negative_words = ['' for _ in range(5)]
    for key, value in dictionary_binary.items():
        for i in range(5):
            if value == top_positive_indices[i]:
                top_positive_words[i] = key
        for i in range(5):
            if value == top_negative_indices[i]:
                top_negative_words[i] = key
    words = top_negative_words + top_positive_words
    values = np.hstack([coefficients[top_negative_indices], coefficients[top_positive_indices]])

    plt.figure(figsize=(12, 6))
    plt.barh(words, values, color=['red']*5 + ['blue']*5)
    plt.xlabel('Coefficient Value')
    plt.ylabel('Word')
    plt.title('Top 5 Negative and Positive Coefficients vs. Words')
    plt.savefig('Top 5 Negative and Positive Coefficients vs. Words.png')
    plt.close()

    #3.2a
    print("\nQuestion 3.2a")
    C_values = [0.001,0.01,0.1,1]
    best_score = 0
    best_c = float('-inf')
    for c in C_values:
        clf = LinearSVC(C=c, penalty='l1', loss='squared_hinge', random_state=445, dual=False)
        score = cv_performance(clf,X_train,Y_train, k =5,metric = "auroc")
        if score > best_score:
            best_score = score
            best_c = c
    print("Best AUROC score:", "{:.4f}".format(best_score), "with C =", "{:.4f}".format(best_c))

    clf = LinearSVC(C=best_c, loss='squared_hinge', penalty='l1', random_state=445,dual=False)
    clf.fit(X_train, Y_train)
    y_pred = clf.decision_function(X_test)
    print("AUROC score for test set:", "{:.4f}".format(metrics.roc_auc_score(Y_test, y_pred)))

    #3.2b
    C_range = [0.001, 0.01, 0.1, 1]
    plot_weight(X_train, Y_train, penalty='l1', C_range=C_range, loss='squared_hinge', dual=False)

    #3.3a
    print("\nQuestion 3.3a")
        #Grid Search
    value_range = np.array([0.01, 0.1, 1, 10, 100, 1000])
    param_range = list(itertools.product(value_range, value_range))
    best_C, best_r, gridSearch_scores = select_param_quadratic(X_train,Y_train,metric='auroc',param_range=param_range)
    clf = SVC(C = best_C, kernel='poly', degree = 2, coef0=best_r, gamma='auto')
    clf.fit(X_train, Y_train)
    y_pred = clf.decision_function(X_test)
    score = metrics.roc_auc_score(Y_test, y_pred)
    print("GridSearch| (best C, best r, AUROC score):", "{:.4f}".format(best_C), "{:.4f}".format(best_r), "{:.4f}".format(score))

        #Random Search
    samples_c = np.array([10**np.random.uniform(-2,3) for _ in range(25)])
    samples_r = np.array([10**np.random.uniform(-2,3) for _ in range(25)])
    param_range_random = np.column_stack((samples_c,samples_r))
    best_C, best_r, randomSearch_scores = select_param_quadratic(X_train,Y_train,metric='auroc',param_range=param_range_random)
    clf = SVC(C = best_C, kernel='poly', degree = 2, coef0=best_r, gamma='auto')
    clf.fit(X_train, Y_train)
    y_pred = clf.decision_function(X_test)
    score = metrics.roc_auc_score(Y_test, y_pred)
    print("RandomSearch| (best C, best r, AUROC score):", "{:.4f}".format(best_C), "{:.4f}".format(best_r), "{:.4f}".format(score))

    #3.3b
    import seaborn as sns
    param_range = np.array(param_range)
    param_range_random = np.round(np.array(param_range_random), decimals = 4)
    value_pairs = [tuple(row) for row in param_range]
    value_pairs_random = [tuple(row) for row in param_range_random]
    df_grid = pd.Series(gridSearch_scores, index=pd.MultiIndex.from_tuples(value_pairs, names=['C', 'r'])).unstack()
    df_random = pd.Series(randomSearch_scores, index=pd.MultiIndex.from_tuples(value_pairs_random, names=['C', 'r'])).unstack()

        #plot gridsearch
    sns.heatmap(df_grid.T, annot=True, cmap='viridis', fmt=".2f")
    plt.xlabel('C')
    plt.ylabel('r')
    plt.title("GridSearchPerformance.png")
    plt.savefig("GridSearchPerformance.png")
    plt.close()

        #plot randomsearch
    plt.figure(figsize=(10, 10))
    sns.heatmap(df_random.T, annot=True, cmap='viridis', fmt=".2f")
    plt.xlabel('C')
    plt.ylabel('r')
    plt.title("RandomSearchPerformance.png")
    plt.savefig("RandomSearchPerformance.png")
    plt.close()

    #4.1c
    print("\nQuestion 4.1c")
    clf = LinearSVC(penalty='l2',C=0.01,loss='hinge', random_state=445, dual=True, class_weight={-1:1,1:10})
    clf.fit(X_train,Y_train)
    
    for m in metric:
        if(m=='auroc'):
            y_pred = clf.decision_function(X_test)
        else:
            y_pred = clf.predict(X_test)
        print(m, ':', "{:.4f}".format(performance(Y_test,y_pred=y_pred,metric=m)))

    #4.2a
    print("\nQuestion 4.2a")
    clf = LinearSVC(penalty='l2',C=0.01,loss='hinge', random_state=445, dual=True, class_weight={-1:1,1:1})
    clf.fit(IMB_features,IMB_labels)
    
    for m in metric:
        if(m=='auroc'):
            y_pred = clf.decision_function(IMB_test_features)
        else:
            y_pred = clf.predict(IMB_test_features)
        print(m, ':', "{:.4f}".format(performance(IMB_test_labels,y_pred=y_pred,metric=m)))
    
    #4.3a
    print("\nQuestion 4.3a")
    value_range = np.array([x for x in range(10) if x>0])
    param_range = list(itertools.product(value_range, value_range))
    best_score = 0
    best_weights = (0,0)
    for w1,w2 in param_range:
        clf = LinearSVC(penalty='l2',C=0.01,loss='hinge', random_state=445, dual=True, class_weight={-1:w1,1:w2})
        score = cv_performance(clf,IMB_features,IMB_labels,k=5,metric='auroc')
        if score > best_score:
            best_score = score
            best_weights = (w1,w2)
    
    print("weights for labels (-1,1):",best_weights,"auroc:","{:.4f}".format(score))
    
    #4.3b
    print("\nQuestion 4.3b")
    clf = LinearSVC(penalty='l2',C=0.01,loss='hinge', random_state=445, dual=True, class_weight={-1:best_weights[0],1:best_weights[1]})
    clf.fit(IMB_features,IMB_labels)
    
    for m in metric:
        if(m=='auroc'):
            y_pred = clf.decision_function(IMB_test_features)
        else:
            y_pred = clf.predict(IMB_test_features)
        print(m, ':', "{:.4f}".format(performance(IMB_test_labels,y_pred=y_pred,metric=m)))

    #4.4
    clf = LinearSVC(penalty='l2',C=0.01,loss='hinge', random_state=445, dual=True, class_weight={-1:1,1:1})
    clf.fit(IMB_features,IMB_labels)
    y_pred = clf.decision_function(IMB_test_features)
    fpr, tpr, _ = metrics.roc_curve(IMB_test_labels, y_pred)
    roc_auc = metrics.auc(fpr, tpr)

    clf_o = LinearSVC(penalty='l2',C=0.01,loss='hinge', random_state=445, dual=True, class_weight={-1:best_weights[0],1:best_weights[1]})
    clf_o.fit(IMB_features,IMB_labels)
    y_pred_o = clf_o.decision_function(IMB_test_features)
    fpr_o, tpr_o, _ = metrics.roc_curve(IMB_test_labels, y_pred_o)
    roc_auc_o = metrics.auc(fpr_o, tpr_o)

    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, color='red', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot(fpr_o, tpr_o, color='green', lw=2, label=f'ROC curve (area = {roc_auc_o:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.title("ROC Curves.png")
    plt.savefig("ROC Curves.png")
    plt.close()

    #5.1a
    print("\nQuestion 5.1a")
    actor,actress = count_actors_and_actresses("data/dataset.csv")
    print("actor:", actor, "actress:", actress)

    #5.1b
    plot_actors_and_actresses("data/dataset.csv", x_label="label")

    #5.1c
    plot_actors_and_actresses("data/dataset.csv", x_label="rating")

    #5.1d
    print("\nQuestion 5.1d")
    clf = LinearSVC(penalty='l2',C=0.1,loss='hinge', random_state=445, dual=True)
    clf.fit(X_train,Y_train)
    actor_index = dictionary_binary['actor']
    actress_index = dictionary_binary['actress']
    actor_coeff = clf.coef_[0,actor_index]
    actress_coeff = clf.coef_[0,actress_index]
    print("actor coefficient:", "{:.4f}".format(actor_coeff), "actress coefficient:", "{:.4f}".format(actress_coeff))

    #5.2a
    print("\nQuestion 5.2a")
    model = train_word2vec("data/dataset.csv")
    actor_embedding = model.wv['actor']
    print("Word Embedding for actor:", actor_embedding)
    print("Dimensionality of the word embedding:", len(actor_embedding))

    #5.2b
    print("\nQuestion 5.2b")
    similar_words = model.wv.most_similar('plot', topn=5)
    for word, score in similar_words:
        print(word, ":","{:.4f}".format(score))

    #5.3a
    print("\nQuestion 5.3a")
    A = ['her','woman','women']
    B = ['him','man','men']
    print("Association:", compute_association('data/dataset.csv','talented',A,B))

    # Read multiclass data
    # TODO: Question 6: Apply a classifier to heldout features, and then use
    #       generate_challenge_labels to print the predicted labels
    from sklearn.feature_extraction.text import TfidfVectorizer

    ## Feature Engineering by using TF-IDF
    def get_multiclass_training_data_tfidf(vectorizer,class_size=750):
        fname = "data/dataset.csv"
        dataframe = load_data(fname)
        neutralDF = dataframe[dataframe["label"] == 0].copy()
        positiveDF = dataframe[dataframe["label"] == 1].copy()
        negativeDF = dataframe[dataframe["label"] == -1].copy()
        X_train = (
            pd.concat(
                [positiveDF[:class_size], negativeDF[:class_size], neutralDF[:class_size]]
            )
            .reset_index(drop=True)
            .copy()
        )
        Y_train = X_train["label"].values.copy()
        vectorizer.fit(X_train)
        dictionary = vectorizer.vocabulary_
        X_train_tfidf = vectorizer.fit_transform(X_train['reviewText'])
        return (X_train_tfidf, Y_train, dictionary)

    def get_heldout_reviews_tfidf(vectorizer):
        fname = "data/heldout.csv"
        dataframe = load_data(fname)
        X = vectorizer.fit_transform(dataframe['reviewText'])
        return X
    (multiclass_features, multiclass_labels, multiclass_dictionary,) = get_multiclass_training_data()
    heldout_features = get_heldout_reviews(multiclass_dictionary)

    # Establish baseline performance score with simple bag of words model
    best_C = -float('inf')
    best_score = 0
    C_range = [0.01, 0.1, 1, 10, 100, 1000]
    for c in C_range:
        clf = LinearSVC(C=c, penalty='l2', loss='squared_hinge', random_state=445, dual=False, multi_class='ovr')
        score = cv_performance(clf,multiclass_features,multiclass_labels, metric='accuracy', k =5)
        if score > best_score:
            best_score = score
            best_C = c
    print("LinearSVC(no modification):", best_C, best_score)
    
    
    # Feature Engineering using N-gram and stop word removal
    stop_words = ['the', 'to', 'and', 'a', 'an', 'in', 'it', 'that', 'on', 'for', 'me','i', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'which', 'who', 'whom']

    for i in [1,2,3,4]:
        vectorizer = TfidfVectorizer(stop_words=stop_words, ngram_range = (1,i))
        (multiclass_features_tfidf, multiclass_labels_tfidf, multiclass_dictionary_tfidf,) = get_multiclass_training_data_tfidf(vectorizer)
        heldout_features_tfidf = get_heldout_reviews_tfidf(vectorizer)
        best_C = -float('inf')
        best_score = 0
        C_range = [0.01, 0.1, 1, 10, 100, 1000]
        for c in C_range:
            clf = LinearSVC(C=c, penalty='l2', loss='squared_hinge', random_state=445, dual=False, multi_class='ovr')
            score = cv_performance(clf,multiclass_features_tfidf,multiclass_labels_tfidf, metric='accuracy', k =5)
            if score > best_score:
                best_score = score
                best_C = c
        print("LinearSVC with ngram =",i,":", best_C, best_score)

    # Transform all data to 2-gram
    vectorizer = TfidfVectorizer(stop_words=stop_words, ngram_range = (1,2), max_features=38000)
    (multiclass_features_tfidf, multiclass_labels_tfidf, multiclass_dictionary_tfidf,) = get_multiclass_training_data_tfidf(vectorizer)
    heldout_features_tfidf = get_heldout_reviews_tfidf(vectorizer)

    # Compare with OvO SVC Model
    clf = SVC(C = 1, kernel='linear', gamma='auto', decision_function_shape= 'ovo')
    print("Linear SVC(ovo): (best C, score):", "1", score)

    # Compare with Quadratic SVC Model
    value_range = np.array([0.1, 1, 10, 100, 1000])
    param_range = list(itertools.product(value_range, value_range))
    best_C, best_r,_ = select_param_quadratic(multiclass_features_tfidf,multiclass_labels_tfidf,metric='accuracy',param_range=param_range)
    clf = SVC(C = best_C, kernel='poly', degree = 2, coef0=best_r, gamma='auto', decision_function_shape= 'ovo')
    print("Quadratic SVC(ovo) GridSearch| (best C, best r, score):", best_C, best_r, score)

    ## Test model (linearSVC)
    clf =  LinearSVC(C=0.1, penalty='l2', loss='squared_hinge', random_state=445, dual=False, multi_class='ovr')
    clf.fit(multiclass_features_tfidf,multiclass_labels_tfidf)
    y_pred = clf.predict(heldout_features_tfidf)
    generate_challenge_labels(y_pred,'junnnli')
if __name__ == "__main__":
    main()
