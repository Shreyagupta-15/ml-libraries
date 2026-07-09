import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

"""
HOW TO USE->

data = load_data("E:\\MyProjects\\ML\\Library\\final\\LinearReg\\Lineardata_train.csv")

X, Y = break_data(data, target='first')

X_train, Y_train, X_test, Y_test = train_test_split(X, Y, scaleit=True)

model = LinearRegression(lr=9e-2, n_iters=50, display="adv")
model.fit(X_train, Y_train, X_test, Y_test)
model.display_wb()
model.plot_()

"""
#added at end

#converts lists into csv files

def convert_to_csv(list, address):
    df = pd.DataFrame({
    'Result': list,
    })
    df.to_csv(address, index=False)

# added for Neural Multi

def print_neural_multi(y_test, predictions): 
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


# added for Neural Binary 

def print_neural_binary(Y_test, Y_pred):
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

def plot_confusion_matrix(actual, pred):

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

#Multi log reg: 

def f1_multi(y_test, y_pred):
    recall_list = recall_multi(y_test, y_pred)
    precision_list = precision_multi(y_test, y_pred)

    f1_list = list()
    for i in range(len(recall_list)):
        if recall_list[i] + precision_list[i] == 0:
            f1_list.append(0)
        else:
            f1_list.append(2*(recall_list[i] * precision_list[i])/(recall_list[i] + precision_list[i]))

    return f1_list

def precision_multi(y_test, y_pred):
  
    #One Hot Encoding of y_test 

    y_test_classes = int(max(y_test)) + 1
    one_hot_y_test = np.eye(y_test_classes)[y_test]

    #One Hot Encoding of y_pred

    y_pred_classes = int(max(y_pred)) + 1
    one_hot_y_pred = np.eye(y_pred_classes)[y_pred]

    precision_list = list()
    #Total precisions will be same max(y_test) + 1  #why 1 coz it starts from 0
    #Precision for y_i =0, y_i=1, y_i=2, y_i=3, y_i=4, y_i=5, y_i=6, y_i=7, y_i=8 bla bla
    #remeb its starting form y_i=0!!!

    for i in range(y_test_classes):
        precision_list.append(precision(one_hot_y_test[:,i], one_hot_y_pred[:,i]))
        
    return precision_list

def recall_multi(y_test, y_pred):
  
    #One Hot Encoding of y_test 

    y_test_classes = int(max(y_test)) + 1
    one_hot_y_test = np.eye(y_test_classes)[y_test]

    #One Hot Encoding of y_pred

    y_pred_classes = int(max(y_pred)) + 1
    one_hot_y_pred = np.eye(y_pred_classes)[y_pred]

    recall_list = list()
    #Total precisions will be same max(y_test) + 1  #why 1 coz it starts from 0
    #Precision for y_i =0, y_i=1, y_i=2, y_i=3, y_i=4, y_i=5, y_i=6, y_i=7, y_i=8 bla bla
    #remeb its starting form y_i=0!!!

    for i in range(y_test_classes):
        recall_list.append(recall(one_hot_y_test[:,i], one_hot_y_pred[:,i]))
        
    return recall_list


#Log reg-.

def recall(y_test, y_pred):
    y_test = y_test.flatten()
    y_pred = y_pred.flatten()
    
    true_pos = np.sum((y_test == 1) & (y_pred == 1))
    #yadak & ?? ans-> & is used for element-wise logical AND in NumPy arrays.
    false_neg= np.sum((y_test == 1) & (y_pred == 0))
    
    #avoid div by 0
    if true_pos+false_neg == 0:
        return 0
    
    return true_pos/(true_pos+false_neg) 

def precision(y_test, y_pred):
    y_test = y_test.flatten()
    y_pred = y_pred.flatten() #make it 1d array
    
    true_pos = np.sum((y_test == 1) & (y_pred == 1))
    #yadak & ?? ans-> & is used for element-wise logical AND in NumPy arrays.
    false_neg= np.sum((y_test == 0) & (y_pred == 1))
    
    #avoid div by 0
    if true_pos+false_neg == 0:
        return 0
    
    return true_pos/(true_pos+false_neg)

def f1(y_test, y_pred):
  
    recall_ = recall(y_test, y_pred)
    precison_ = precision(y_test, y_pred)
    
    #avoid div by 0
    if recall_ + precison_ == 0:
        return 0
    
    return 2*(recall_ * precison_)/(recall_ + precison_)

# Linear reg ->
def rmse(y_test, y_pred):
    return np.sqrt(mse(y_test, y_pred))

def mae(y_test, y_pred):
    err = np.abs(y_test - y_pred)
    sum = np.sum(err)
    m = err.shape[0]
    
    return 1/m * sum

def mse(y_test, y_pred):
    m = y_test.shape[0]
    err = y_pred - y_test
    err = err**2
    err = np.sum(err)
    err /= 2*m
    
    #print(f"Mean Sqaure Error is: {err}")
    return err

def r2_square(y_test, y_pred):
    err = y_pred - y_test
    err = err**2
    
    mean = np.mean(y_pred)
    
    ss1 = np.sum(err)
    ss2 = y_test - mean
    ss2 = ss2**2
    ss2 = np.sum(ss2)
    
    r2 = 1 - (ss1/ss2)
    
    #print(f'The R2 score is: {r2}')
    return r2

def accuracy_(y_test, y_pred):
    #print(f'Accuracy: {((np.sum(y_pred==y_test)/len(y_test))*100):.4f}%')
    return np.sum(y_test == y_pred)/len(y_test)

def load_data(address):
    df2=pd.read_csv(address)
    df=df2.sample(frac=1, random_state=42).reset_index(drop=True)
    data = np.array(df.iloc[:,:])
    return data

def break_data(df, target='last', has_id=False):
    n = df.shape[1]
    if target == 'last':
        print(f"Assuming the last colum is the output column")
        if has_id:
            X = df[:,1:n-1] #starts with feature 1
        else:
            X = df[:,0:n-1] #starts with feature 0
        Y = df[:,n-1]
    elif target == 'first':
        print(f"Assuming the first colum is the output column")
        X = df[:,1:n]
        Y = df[:,0]
        
    elif target == None:
        if has_id:
            X = df[:,1:n] #starts with feature 1
            return X
        else:
            print(f"There was no need break this")
            X = df[:,:]
            return X
    return X, Y

#Make one scaler function as well

def train_test_split(X,Y,test_size=0.2, scaleit=False): 
    m = X.shape[0] #-> number of rows
    m_test = test_size * m
    m_train = int(m - m_test)
    X_train = X[0:m_train,:]
    X_test = X[m_train:m,:]
    Y_train = Y[0:m_train]
    Y_test = Y[m_train:m]
    
    if scaleit == True:
        mew=np.mean(X_train,axis=0)
        sigma=np.std(X_train, axis=0)
        X_scaled= (X_train-mew)/sigma
        X_scaled_test= (X_test-mew)/sigma
        #returing mew and sigma so that you can rescale manual and test manually
        return X_scaled, Y_train, X_scaled_test, Y_test, mew, sigma
    
    else:
        return X_train, Y_train, X_test, Y_test
