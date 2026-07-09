import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from utils import * #Importing all the functions from utils.py
from one_hot_encoding import *

#LINER REGRESSION

class LinearRegression:
    
    def __init__(self, n_iters=1000, lr=1e-3, display="adv", lambda_=0):
        self.n_iters = n_iters
        self.lr = lr
        self.weights = None
        self.bias = None
        self.display = display
        self.L = lambda_
        self.cost_list = list()
        self.L2_reg = list()
        self.Y_test = None
        self.l2_norm_without_reg = list()

    def cost_function(self, X, Y):
        m = X.shape[0]
        
        y_pred = np.dot(X,self.weights) + self.bias
        err = y_pred - Y
        a1 = np.dot(err,err)
        a2 = np.sum(a1)
        
        a2 /= 2*m
        
        if self.L == 0:
        
            return a2
        
        else:
        
            w = self.weights
            w = w ** 2
            w = np.sum(w)
            l2_norm = np.sqrt(w) # for L2  Norm
            w *= self.L
            w /= 2*m
            
            #print(f"Without reg: {a2}, with: {a2+w}")
            return (a2 + w), l2_norm

    def fit(self, X, Y, X_test, Y_test,*, for_l2=False):
    
        m,n = X.shape
        self.weights = np.zeros(n)
        self.X = X
        self.Y = Y
        self.bias = 0
        self.Y_test = Y_test
        self.X_test = X_test
        
        start_time = time.time()
        
        for k in range(self.n_iters):
            Y_pred = np.dot(X,self.weights) + self.bias
            err = Y_pred - Y
            
            dw = (1/m) * np.dot(X.T, err)
            
            #Regularization
            if self.L != 0:
            #print(f"regulatization is happening")
                w_reg = (self.L/m) * (self.weights)
                dw += w_reg
            db = (1/m) * np.sum(err)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            if self.L == 0:
                a = self.cost_function(X, Y)
                self.cost_list.append(a)

                if for_l2:
                    l2_norm_without_reg_calc = np.sqrt(np.sum((self.weights)**2))
                    self.l2_norm_without_reg.append(l2_norm_without_reg_calc)

            else:
                a,b = self.cost_function(X, Y)
                
                self.cost_list.append(a)
                self.L2_reg.append(b)

            if self.display == 'adv':
                print(f"Iteration is: {k}, cost funciton is: {a}")
        
        
        end_time = time.time()
        
        if not for_l2:
            print(f"Execution time: {end_time - start_time:.6f} seconds!!!")


    def predict_(self, X):
        y_pred_class = np.dot(X, self.weights) + self.bias
        return y_pred_class

    def plot_l2_norm(self):
    
        dummy = LinearRegression(lr=self.lr, n_iters=self.n_iters, lambda_=0, display="basic")
        dummy.fit(self.X, self.Y, self.X, self.Y, for_l2=True)
        
        fig, ax1 = plt.subplots(1,1, figsize=(10,5))
        
        ax1.plot(dummy.l2_norm_without_reg, color='blue', label="Lambda: 0")
        
        ax1.plot(self.L2_reg, color='black', label=f"Lambda: {self.L}")
        
        ax1.set_xlabel("Iterations ->")
        ax1.set_ylabel("L2_Norm ->")
        ax1.set_title("Effect of regularization on weight norms")
        ax1.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_(self):
    
        print("Plotting")
        
        y_predicted = self.predict_(self.X_test)
        
        if self.display == "adv":
        
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns
            
            # Data points: [(x1, y1, class1), (x2, y2, class2), ...]
            ax1.scatter(self.Y_test, self.Y_test, color='blue', label='actual')
            ax1.scatter(self.Y_test, y_predicted, color='red', label='predicted')
            
            # Labels and title
            ax1.set_xlabel('X-axis')
            ax1.set_ylabel('Y-axis')
            ax1.set_title('Scatter Plot of Data Points')
            
            # Display legend
            ax1.legend()
            # Second subplot: Line plot for cost function
            
            ax2.plot(self.cost_list, label='Cost Function', color='black')
            ax2.set_title("Cost Function Graph")
            ax2.set_xlabel("Iterations")
            ax2.set_ylabel("Cost")
            ax2.legend()
            
            # Show the complete figure
            plt.tight_layout()
            plt.show()


#Logistic Regression

class LogisticRegression:

    def __init__(self, n_iters=1000, lr=1e-3, threshold=0.5, display="adv", return_prob=False): #this return prob will be used in Multi classification
        self.n_iters = n_iters
        self.threshold = threshold
        self.lr = lr
        self.weights = None
        self.bias = None
        self.display = display
        self.cost_list = list()
        self.Y_test = None
        self.p_hist = list() #*precision vs recall history samand
        self.r_hist = list() #precision vs *recall history samand
        self.return_prob = return_prob
        
    def cost_function(self, X, Y):
        m,n = X.shape
        
        z = np.dot(X,self.weights) + self.bias
        Y_pred = sigmoid(z)
        Y_pred = np.clip(Y_pred, 1e-15, 1 - 1e-15)
        
        sum_1 = np.dot(Y.T, np.log(Y_pred))
        
        Y = 1 - Y
        Y_pred = 1 - Y_pred    
        #now y is 1-y and Y_pred is 1-Y_pred
        
        sum_2 = np.dot(Y.T, np.log(Y_pred))
        final_sum = sum_1 + sum_2
        final_sum *= -1
        
        final_sum /= m
        
        return final_sum

    def fit(self, X, Y, X_test, Y_test):
    
        m,n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0
        self.Y_test = Y_test
        self.X_test = X_test
        
        start_time = time.time()
        
        for k in range(self.n_iters):
            z = np.dot(X,self.weights) + self.bias
            Y_pred = sigmoid(z)
            err = Y_pred - Y
            
            dw = (1/m) * np.dot(X.T, err)
            db = (1/m) * np.sum(err)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            a = self.cost_function(X, Y)
            if self.display == "adv":
                if k%10 == 0:
                    print(f"Iteration is: {k}, cost funciton is: {a}")
            
            #APPENDING:
            self.return_prob = False  # false idra false a irtayeth, true idra asta false aguth
            curr_pred = np.array(self.predict_(self.X_test))
            if not self.return_prob:   # false idra true madud jarurat illa, adra true idra false madu jarurat ayethi
                self.return_prob = True
            actual = self.Y_test
            self.p_hist.append(precision(actual, curr_pred))
            self.r_hist.append(recall(actual, curr_pred))
      

            self.cost_list.append(a)

            end_time = time.time()
            print(f"Execution time: {end_time - start_time:.6f} seconds!!!")

    def plot_confusion_matrix(self, actual, pred):
    
        cm = confu_matrix(actual, pred)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Plotting the confusion matrix using imshow
        cax = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        
        # Adding color bar
        plt.colorbar(cax)
        
        # Adding labels to axes
        ax.set_xticks(np.arange(cm.shape[1]))
        ax.set_yticks(np.arange(cm.shape[0]))
        ax.set_xticklabels(['Predicted Negative', 'Predicted Positive'])
        ax.set_yticklabels(['Actual Negative', 'Actual Positive'])
        
        # Labeling the axes
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title('Confusion Matrix')
        
        # Annotating each cell with the numeric value
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        
        # Show plot
        plt.show()

    def plot_(self):
    
        if self.display == "adv":
        
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns
            
            ax1.plot(self.r_hist, self.p_hist, color='orange', label='Model')
            
            # Display legend
            ax1.set_title("Precision")
            ax1.set_xlabel("Recall")
            ax1.set_ylabel("Precision")
            ax1.legend()
            
            # Second subplot: Line plot for cost function
            ax2.plot(self.cost_list, label='Cost Function', color='black')
            ax2.set_title("Cost Function Graph")
            ax2.set_xlabel("Iterations")
            ax2.set_ylabel("Cost")
            ax2.legend()
            
            # Show the complete figure
            plt.tight_layout()
            plt.show()

    def predict_(self, X): 
        z = np.dot(X, self.weights) + self.bias
        pred = sigmoid(z)
        
        if self.return_prob: #I will use this return_prob in multiclassification
        
            return pred
        
        else:
            y_pred_class = [1 if i > self.threshold else 0 for i in pred]
            return np.array(y_pred_class)

def sigmoid(a):
    return 1/(1+np.exp(-a))

def print_log_details(Y_test, Y_pred):
    print("+===============+=============+")
    print("| Metric        | Value       |")
    print("+===============+=============+")
    print(f"| F1            | {f1(Y_test, Y_pred):<12.4f}|")
    print(f"| Accuracy      | {accuracy_(Y_test, Y_pred):<12.4f}|")
    print(f"| Recall        | {recall(Y_test, Y_pred):<12.4f}|")
    print(f"| Precision     | {precision(Y_test, Y_pred):<12.4f}|")
    print("+===============+=============+")

def confu_matrix(actual, pred):

    actual = np.array(actual)
    pred = np.array(pred)
    #coz such == comparisons only work with numpy arrays
    
    a_n_p_n = np.sum( (actual == 0) & (pred == 0) ) #actual neg predicted neg --> [0 , 1, 0, 0, 1, 0, 0, 1, 0] so sum will get us correct count
    #and also true and flase are auto converted to 0s and 1s by np.sum
    a_n_p_p = np.sum( (actual == 0) & (pred == 1) )
    
    a_p_p_p = np.sum( (actual == 1) & (pred == 1) ) #actual pos predicted pos
    a_p_p_n = np.sum( (actual == 1) & (pred == 0) )
    
    return np.array([ [a_n_p_n, a_n_p_p] , [a_p_p_n, a_p_p_p] ])

#Multi Log

def print_log_details(y_test, predictions):

    print("+===============+=============+")
    print("| Metric        | Value       |")
    print("+===============+=============+")
    print(f"| Accuracy      | {accuracy_(y_test, predictions):<12.4f}|")
    print("+===============+=============+")
    
    y_test = y_test.astype(int)
    predictions = predictions.astype(int)
    
    recall_list = recall_multi(y_test, predictions)
    
    precision_list = precision_multi(y_test, predictions)
    f1_list = f1_multi(y_test, predictions)
    
    
    print(f"| Recall        |-------------|")
    for i in range(len(recall_list)):
        print(f"| Class {i}       | {recall_list[i]:<12.4f}|")
    
    print("+===============+=============+")
    print(f"| Precision     |-------------|")
    
    for i in range(len(recall_list)):
        print(f"| Class {i}       | {precision_list[i]:<12.4f}|")
    
    print("+===============+=============+")
    print(f"| F1            |-------------|")
    
    for i in range(len(f1_list)):
        print(f"| Class {i}       | {f1_list[i]:<12.4f}|")
    
    print("+===============+=============+")


class MultiClassification:
  
    def __init__(self, each_lr=0.01, each_iters=1000, display = "adv"):
        self.models = []
        self.display = display
        self.each_lr = each_lr
        self.each_iters = each_iters
    
    def fit(self, X, y):
    
        #print(f'Y: {y}')
        print(f'Uniques: {np.unique(y)}')
        
        self.total_classes = len(np.unique(y))
        
        self.array_of_p_hist = np.empty((self.total_classes, self.each_iters)) #assum total class=2, each iters=10 so u have 2x10 matrix
        self.array_of_r_hist = np.empty((self.total_classes, self.each_iters))  #recall history
        curr_class_count = 0
        
        for y_i in np.unique(y):
        
            x_pos = X[y == y_i] 
            x_neg = X[y != y_i] 
            
            x_final = np.vstack([x_pos, x_neg])
            
            y_pos = np.ones(x_pos.shape[0])
            y_neg = np.zeros(x_neg.shape[0])
            
            y_final = np.hstack([y_pos, y_neg])
            
            
            model = LogisticRegression(return_prob=True, lr=self.each_lr, n_iters=self.each_iters, display=self.display)
            model.fit(x_final, y_final, x_final, y_final)
            
            self.array_of_p_hist[curr_class_count] = model.p_hist
            self.array_of_r_hist[curr_class_count] = model.r_hist
            curr_class_count += 1
            self.models.append([y_i, model])

    def plot_precision_recall_curve(self):
    
        fig, array = plt.subplots(int(np.ceil(self.total_classes/4)), 4, figsize=(14, 13))
        
        # Flatten the array for easier indexing
        array = array.flatten()
        
        # Plot precision-recall curves for each class
        for i in range(self.total_classes):
            random_color = np.random.choice(['red', 'blue', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'])
            array[i].plot(self.array_of_r_hist[i], self.array_of_p_hist[i], label=f"Class {i}", color = random_color)
            array[i].set_title(f"Class {i} Precision-Recall Curve")
            array[i].set_xlabel("Recall")
            array[i].set_ylabel("Precision")
            array[i].legend()
        
        # Remove empty subplots (if any)
        for j in range(self.total_classes, len(array)):
            fig.delaxes(array[j])  # Remove unused axes
        
        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()

    def predict_(self, X): 
        y_pred = [[label, model.predict_(X)] for label, model in self.models]
        
        output = []
        
        for i in range(X.shape[0]):
        
            max_prob = -1
            max_label = None
            
            for k in y_pred:
                current_prob = k[1][i]
                label = k[0]
                
                if current_prob > max_prob:
                    max_prob = current_prob
                    max_label = label
            
            output.append(max_label)
        
        return np.array(output)

# K Means

import random
random.seed(42)

def distance(p1, p2):
    return np.sqrt(np.sum((p1-p2)**2))

def run_elbow_test(X, max_k=3):

    wcss = []
    
    for k in range(1, max_k+1):
        model = KMeans(K = k, max_iters=150, plot_graph=False, display="basic")
        y_pred = model.predict_(X)
        print(f"WCSS for K = {k}: {model.WCSS}")
        if k != max_k:
            print("Calculating for next K.......")
        wcss.append(model.WCSS)
    
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
    
    ax1.plot(wcss, color="blue", marker="o", linestyle="--")
    ax1.set_xlabel("Number of clusters")
    ax1.set_ylabel("WCSS")
    ax1.set_title("Elbow Method")
    
    plt.show()     

class KMeans:

    def __init__(self, K, max_iters, plot_graph, display="adv"):
        self.max_iters = max_iters
        self.display = display
        self.plot_graph = plot_graph
        self.K = K #Number of clusters
        self.WCSS = 0 #Within Cluster Sum of Squares
        
        #storing for ploting:
        self.hist = list()
        # it will store like [ [1st store] , [2nd store]]
        # 1st store -> [[clusters], [K number of centroids]]
        
        #Making clusters, remeber each cluster is a list
        self.clusters = [[] for _ in range(self.K)] # a list which has K lists
        
        #Storing centroids
        self.centroids = []
    
    def wcss_(self):
    
        for idx, cluster in enumerate(self.clusters): 
            dis_clusters_sos = 0 #dis clusters sum of squares
            cluster_centroid = self.centroids[idx]
            for sample_idx in cluster:
                dis_clusters_sos += distance(self.X[sample_idx], cluster_centroid)**2
            self.WCSS += dis_clusters_sos

    def predict_(self, X):
        self.X = X
        self.m, self.n = X.shape
        
        #Set some random sample rows(vectors) as centroid
        random_sample_indexs = np.random.choice(self.m, self.K, replace=False) #(max num, how many u want, replacement)
        
        self.centroids = [self.X[indexs] for indexs in random_sample_indexs]
        
        #Improve clusters:
        
        for k in range(self.max_iters):
        
            if self.display == "adv":
                print(f"In my {k}th iter")
            
            #assign samples to closest centroids
            self.clusters = self.assign_clusters_(self.centroids)
            
            #this will update the clusters
            if self.plot_graph:
                self.store_data()
            
            #calculate new centroids from the clusters
            centroids_old = self.centroids  
            self.centroids = self.get_new_centroids(self.clusters)
            
            if self.is_converged(self.centroids, centroids_old):
                print(f"data is converged")
                break
            
            #this will update the centroids
            if self.plot_graph:
                self.store_data()

        #calculate the within cluster sum of squares
        self.wcss_()
        
        #return respective cluster indexes
        return self.set_labels_to_clusters(self.clusters)
  
#<=======================================================================================>

    def is_converged(self, new, old):
    
        distances = [distance(a,b) for a,b in zip(new, old)]
        return np.sum(distances) == 0 or np.sum(distances) <= 1e-6 #true if statement is true

    def plot_(self, feature_x=0, feature_y=1):
    
        if feature_x < 0 or feature_x == feature_y:
            print(f'Tapp idu!! "-" ')
        
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
        
        for idx, each in enumerate(self.hist):
            ax1.cla()
            clusters = each[0]
            centroids = each[1]
            
            ax1.set_xlabel(f"Feature {feature_x} ->")
            ax1.set_ylabel(f"Feature {feature_y} ->")
            
            
            for indexs in clusters:
                points = [self.X[indexs][:,feature_x], self.X[indexs][:,feature_y]] #why transpose? 
                #see *points what it does is it takes 1st array as x and 2nd as y so if array is [[2,3],[4,5],[6,7]] -> 3x2 
                # we will transpose it to 2x3 that is [[2,4,6],[3,5,7]] [ [x], [y] ]
                ax1.scatter(*points) # *points will unpack on its own and get differnt color for each scatter
            
            for point in centroids:
                points = [point[feature_x],point[feature_y]]
                ax1.scatter(*points, marker="x", color="black")
            
            ax1.set_title(f"Iteration: {idx/2}, changed centroid now changing clusters")
            
            
            plt.draw()
            plt.pause(0.1)
            
            
        plt.show()

    def store_data(self):
        curr_labels = self.clusters #current clusters list
        curr_centroids = self.centroids
        self.hist.append([curr_labels, curr_centroids])
    
    def set_labels_to_clusters(self, clusters):
    
        labels = np.zeros(self.m)
        
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        
        return labels
  
    #Example:
    #clusters -> [[1,5,9], [6,8,0], [2,3,4,7]]
    #so in second iter cluster -> [6,8,0]
    # in loop cluster_idx = 1, that is why 
    #labels will be now [1,0,0,0,0,0,1,0,1]
    #in third iter it will be [1,0,2,2,2,0,1,2,1]

    def get_new_centroids(self, clusters):
        centroids = np.zeros((self.K, self.n))
        #Yadak? coz assum u have K=3 that is 3 clusters so u will have 3 centroid vecotrs adakke u habe K x n matrix
        
        for idx, cluster in enumerate(clusters):
            centroids_mean = np.mean(self.X[cluster], axis=0) #axis = 0 is imp bcoz we need mean along columns that is we need a vecotr of 1 x n
        #also tackled the issue where len(cluster was 0)
            centroids[idx] = centroids_mean #get repective means
        
        return centroids
    
    def assign_clusters_(self, centroids):
        clusters = [[] for _ in range(self.K)] #list of K lists
        
        for idx, sample in enumerate(self.X):
            closest_centroid_idx = self.get_closest_centroid_idx(sample, centroids)
            clusters[closest_centroid_idx].append(idx)
            #basically each cluster stores indexes which are nearer to thier respective clusters
    
        return clusters
    
    def get_closest_centroid_idx(self, sample, centroids):
        distances = [distance(sample, point) for point in centroids]
        return np.argmin(distances)


#Polynomial ->

def add_polynomial_features(data, degree, interactions=True):
    
    if data.ndim == 1:
        data = data[:, np.newaxis]  # Convert 1D array to 2D for consistency

    n_samples, n_features = data.shape
    feature_list = [data]  # Start with the original features

    # For degree >= 2, include squared terms and pairwise interaction terms
    if degree >= 2:
        for i in range(n_features):
            feature_list.append((data[:, i] ** 2)[:, np.newaxis])  # Add squared terms (x1^2, x2^2, etc.)

        if interactions:
            # Add pairwise interaction terms (x1*x2, x1*x3, etc.)
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    interaction_term = data[:, i] * data[:, j]
                    feature_list.append(interaction_term[:, np.newaxis])

    # For degree >= 3, include cubic terms and higher-order interaction terms
    if degree >= 3:
        for i in range(n_features):
            feature_list.append((data[:, i] ** 3)[:, np.newaxis])  # Add cubic terms (x1^3, x2^3, etc.)

        if interactions:
            # Add higher-order interaction terms (e.g., x1^p * x2^q for p+q <= degree)
            for i in range(n_features):
                for j in range(n_features):  # Allow self-interactions and cross-feature interactions
                    if i != j:
                        for p in range(1, degree):  # Iterate over powers for the first feature
                            for q in range(1, degree):  # Iterate over powers for the second feature
                                if p + q <= degree:  # Ensure the total degree doesn't exceed the limit
                                    higher_order_term = (data[:, i] ** p) * (data[:, j] ** q)
                                    feature_list.append(higher_order_term[:, np.newaxis])

    return np.hstack(feature_list)  # Combine all features horizontally


#KNN ->

def distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2)) #ex: x1 = [1,2,3], x2 = [4,5,6] => sqrt( (1-4)^2 + (2-5)^2 + (3-6)^2 )

def max_occurence(arr):
    return np.argmax(np.bincount(arr)) 
    #input array -> [1,2,2,3,3,3,3]
    #np.bincount returns [0,1,2,4] means 0 comes 0 times, 1 comes 1s, 3 comes 4 times
    #np.argmax(of that arrray) returns index of max number in that array here it will be index=3

class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        y_pred = [self.predict_single(x) for x in X]
        return np.array(y_pred)
    
    def predict_single(self, x):
        distances = [distance(x, x_train) for x_train in self.X_train]
        closest_indices = np.argsort(distances)[:self.k] #np.argsort gets u indices  after sorting and u just want k indices
    
        closest_indices_labels = [self.y_train[i] for i in closest_indices]
        return max_occurence(closest_indices_labels)


#Neural ->

class Activation():
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
    
    def forward(self, input):
        #Just applying the activation function to the input
        self.input = input
        return self.activation(self.input)
    
    def backward(self, derivative_matrix, lr):
        #ID -> 5 in notebook
        return np.multiply(derivative_matrix, self.activation_prime(self.input))
    
    #Activation functions classes -->
    
class Sigmoid(Activation):
    def __init__(self):
        super().__init__(self.sigmoid, self.sigmoid_prime)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -709, 709)))

        """
        # Use a stable implementation
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        ) 
        """

    #return 1 / (1 + np.exp(-x)) -> overflow error

    def sigmoid_prime(self, x):
        #print(f"Back prop is going on")
        return self.sigmoid(x) * (1 - self.sigmoid(x))

class Softmax(Activation):
    def __init__(self):
        super().__init__(self.softmax, self.softmax_prime)
    
    def softmax(self, x):
        #get exponent of all the z's
        exps = np.exp(x - np.max(x))
    
    # why i did x - max(x) is because if we have large numbers then the exp will be very large and it will overflow
    #you will take that common e ki power and it will get canceled out so overall we will still get same output
        return exps / np.sum(exps)
    
    def softmax_prime(self, x):
        #print(f"softmax prime was called")
        return self.softmax(x) * (1 - self.softmax(x))

class ReLU(Activation):
    def __init__(self):
        super().__init__(self.relu, self.relu_prime)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_prime(self, x):
        return np.where(x <= 0, 0, 1)

class Tanh(Activation):
    def __init__(self):
        super().__init__(self.tanh, self.tanh_prime)
    
    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_prime(self, x):
        return 1 - np.tanh(x)**2

class Dense:
    def __init__(self, n_neurons, activation):
        self.n_neurons = n_neurons
        self.activation = activation
    
        #Activation functions
        #Relu adds non linearity to the model!!!!!!!!!!
        #Vanishing geadient problem is solved by relu
        if self.activation == 'tanh':
            self.activation = Tanh()
        elif self.activation == 'sigmoid':
            self.activation = Sigmoid()
        elif self.activation == 'softmax':
            self.activation = Softmax()
        elif self.activation == 'relu':
            self.activation = ReLU()

#Errors:

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    smtg =  2 * (y_pred - y_true) / np.size(y_true)
    return smtg

def binary_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)  # Avoid log(0) and log(1)
    return -1*(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_prime(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)  # Avoid log(0) and log(1)
    return (y_pred - y_true) / (y_pred * (1 - y_pred))  

def categorical_cross_entropy(y_true, y_pred):
    #print(f"<-- Y_true: {y_true},\n Y_Pred: [{y_pred}]\n -->")
    #print( -1 * np.sum(y_true * np.log(y_pred)) )
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)  # Avoid log(0) and log(1)
    return -1 * np.sum(y_true * np.log(y_pred))

def categorical_cross_entropy_prime(y_true, y_pred):

    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Avoid log(0)
    return -1*(y_true / y_pred) #retuns a fucking list

class Neural:

    def __init__(self, lr=0.1, n_iters=1000, error_function=None):
        self.network = list()
        self.lr = lr
        self.error_function = error_function
        self.J_history = list()
        self.n_iters = n_iters
    
    def activate_error(self):
        if self.error_function == 'mse':
            self.error = mse
            self.error_prime = mse_prime
        elif self.error_function == 'binary_cross_entropy':
            self.error = binary_cross_entropy
            self.error_prime = binary_cross_entropy_prime
        elif self.error_function == 'categorical_cross_entropy':
            self.error = categorical_cross_entropy
            self.error_prime = categorical_cross_entropy_prime
    
    def Sequential(self, layers):
        prev_features = self.X_train.shape[1]
        for each in layers:
            self.network.append(inner_Dense(prev_features, each.n_neurons)) #creates layer
            self.network.append(each.activation) #creates activation function layer
            prev_features = each.n_neurons 

    def fit(self, X_train, Y_train, X_test, Y_test):
        self.X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))  #Converting to column vector
        self.Y_train = np.reshape(Y_train, (Y_train.shape[0], Y_train.shape[1], 1))  #Converting to column vector
        self.X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))  #Converting to column vector
        self.Y_test = np.reshape(Y_test, (Y_test.shape[0], Y_test.shape[1], 1))  #Converting to column vector

  #actual training
    def train(self):
        for k in range(self.n_iters):
            error = 0
            
            for x, y in zip(self.X_train, self.Y_train):
            
            #Forward propogation
                output = x
                for layer in self.network:
                    output = layer.forward(output)
                    
                self.activate_error()
                error += self.error(y, output)
                #Backward propogation
                derivative_y_final = self.error_prime(y, output)
                for layer in reversed(self.network):
                    derivative_y_final = layer.backward(derivative_y_final, self.lr)
                    
        error /= len(self.X_train)
        self.J_history.append(error)
        
        if k % 2 == 0:
            print('Epoch:', k, 'Error:', error)

    def plot_(self):
        plt.plot(self.J_history, linestyle='-', color='black', label='Error')
        plt.xlabel('Iterations')
        plt.ylabel('Error')
        plt.title('Error vs Iterations')
        plt.legend()
        plt.show()

    def predict(self, Y):
        #Warning here you must pass reshaped Y!!!
        output_list = list()
        for x in Y:
            output = x
            for layer in self.network:
                output = layer.forward(output)
            output_list.append(output)
        return np.array(output_list)

class inner_Dense(): 

    def __init__(self, input_features, output_features):
        self.input = None
        self.weights = np.random.randn(output_features, input_features) * 0.01
        #gives u a matrix of output_features x input_features with mean smtg and std 0.01(before it was 0 and 1 rescpetively)
        
        self.bias = np.random.randn(output_features,1) * 0.01
        
        #by stanford method we need to transpose but here the weights are aleadt transposed one so we can do direct matrix multiplication
        
    def forward(self, input):
        self.input = input
        # ID -> 2 in notebook
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, derivative_matrix, lr):
        #ID -> 3 in notebook
        weights_derivative = np.dot(derivative_matrix, self.input.T)
        # simply bias_derivative = derivative_matrix
        
        self.weights -= lr * weights_derivative
        
        self.bias -= lr * derivative_matrix 
        
        #ID -> 4 in notebook
        return np.dot(self.weights.T, derivative_matrix) #returning the derivative of the input that is DE/DX where E is error and X is input
  
# DECISION TREES

#At every split we need nodes so 
class Node:
    def __init__(self, best_feature=None, left=None, right=None,*,value=None): #trick learnt is that if you use that * it forces u to pass value as Value='smtg'
        self.best_feature = best_feature #best feature of current node on which indexes have been split
        self.left = left #left node
        self.right = right #right node
        self.value = value
    
    def is_leaf_node(self):
        if self.value != None:
            return True
    
#Decision Tree
#assuming all data is hot encoded that is all the values are 0s and 1s
#Notation is that all the helper function end with "_"

class DecisionTree:

    def __init__(self, n_included_features=10, min_samples_split=2, max_depth=100): #i can use n_included_features in randome forest algorthim where we make lot of trees and they select only few features
        self.min_samples_split = min_samples_split
        self.max_depth= max_depth
        self.n_included_features= n_included_features #make it -1 to select all features
        self.root=None

    def fit(self, X, y):
        if self.n_included_features == -1:
            self.n_included_features = X.shape[1]  #Making it select all features
        self.root = self.grow_tree_(X, y)  #Assign node and also left and right nodes

    def grow_tree_(self, X, y, depth=0):
        m, features = X.shape
        
        uniques = len(np.unique(y))
        #print(f'uniques: {uniques}') #uniques will be 2 coz its ( 1 -> is cat, 0-> is not cat )
        
        #Stopping statement->
        if(depth >= self.max_depth or uniques == 1 or m < self.min_samples_split):
            leaf_value = self.most_occuring_(y)
            return Node(value= leaf_value)
        
        selected_feature_indexs = np.random.choice(features, self.n_included_features, replace=False)
        #np.random.choice(range, how many to choose, can they repeat?)
        #print(f'<-- Selected feature indexes: {selected_feature_indexs} -->') -> to debug what and all were choosen
        
        #Find best feature to split->
        
        best_feature = self.best_split_(X, y, selected_feature_indexs)
        
        #Creating child nodes:
        left_indexs, right_indexs = self.split_feature_indexs(X, y, best_feature)
        
        left = self.grow_tree_(X[left_indexs, :], y[left_indexs], depth+1)
        right = self.grow_tree_(X[right_indexs, :], y[right_indexs], depth+1)
        
        return Node(best_feature, left, right) #current splits best feature
        #notice i dint giv value because this is not leaf node

    def best_split_(self, X, y, selected_feature_indexs):
    
        best_gain = -1 
        best_feature = -1 #Let error aries if we dont find best feature
        
        for feature in selected_feature_indexs:
        
            gain = self.calculate_info_gain_(X, y, feature)
            #print(f'gain of feature; {feature}, gain: {gain}') -> to debug which feature has best gain
            
            if gain >= best_gain:
                best_gain = gain
                best_feature = feature
            
        return best_feature

    def entropy_(self, y):
        sum_1s = np.sum(y)
        frac_1s = sum_1s/len(y)
        frac_0s = 1 - frac_1s
        
        if frac_1s == 0 or frac_1s == 1:
            return 0
        else:
            entropy = -1*(frac_1s*np.log2(frac_1s) + frac_0s*np.log2(frac_0s))
        
            return entropy
    
    def split_feature_indexs(self, X, y, feature):
        X_column = X[:, feature]
        left_array = list() #arrays of 1s of that feature
        right_array = list() #arrays of 0s of that feature
        for index in range(len(X_column)):
            if X_column[index] == 1:
                left_array.append(index)
            else:
                right_array.append(index)
    
        return left_array, right_array

    def calculate_info_gain_(self, X, y, feature):
        root_entropy = self.entropy_(y) #Root Entropy
        
        left_indexs, right_indexs = self.split_feature_indexs(X, y, feature)
        
        if len(left_indexs) == 0 or len(right_indexs) == 0:
            return 0 
        
        total = len(y)
        n_left, n_right = len(left_indexs), len(right_indexs)
        frac_left, frac_right = n_left/total, n_right/total
        entropy_left, entropy_right = self.entropy_(y[left_indexs]), self.entropy_(y[right_indexs])
        
        weighted_entropy = frac_left * entropy_left + frac_right * entropy_right
        
        info_gain = root_entropy - weighted_entropy
        
        return info_gain

    def most_occuring_(self, y):
    #calculate fraction of 1 and if fraction is more than 0.5 then our 1 is most occuring or else 0
        sum_0s_1s = np.sum(y)
        
        if len(y) == 0:
            return 0
        
        frac_1s = sum_0s_1s/len(y)
        
        if(frac_1s >= 0.5):
            return 1
        else:
            return 0
    
    def predict(self, X):
        return np.array([self.traverse_tree_(each_row, self.root) for each_row in X]) #basically self.root is the root NODE of the tree
  
    def traverse_tree_(self, row, node):
        if node.is_leaf_node():
            return node.value
        else:
            if row[node.best_feature] == 1: # assume u r at full top node, ofc its not a leaf node, so u have to go left or right, we have the feature at which rooot node was split (i.e node.best_feature) so we will see row[node.best_feature] if its 1 , then it means for tha feature its true and by convention our trues are in left so we gotta go left or other case then right.
                return self.traverse_tree_(row, node.left)
            else:
                return self.traverse_tree_(row, node.right)