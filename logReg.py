import numpy as np

def sigmoid(z): return 1/(1+np.exp(-z))

def predict_proba(X, weights): return sigmoid(np.dot(X, weights))

def predict_class(X, weights):

    preds_class = [0 if p < 0.5 else 1 for p in predict_proba(X, weights)]
    
    return np.array(preds_class)

def binary_loss(X, Y, weights, alpha=0.00001):
    '''
    Binary Cross-entropy Loss

    '''
    P = predict_proba(X, weights)
    gw = (1.0 / X.shape[0]) * np.dot(X.T, (P - Y))
    gw += (alpha * weights) / X.shape[0]
    loss = np.linalg.norm(gw)
    
    return loss
    


def SGD(X_tr, Y_tr, X_dev=[], Y_dev=[], lr=0.1, 
        alpha=0.00001, epochs=5, 
        tolerance=0.0001, print_progress=True):
    weights = np.zeros(X_tr.shape[1], dtype='float') 
    training_loss_history = [] 
    validation_loss_history = [] 
    pre_dev_loss = 99999 
    for iteration in range(epochs):
        # calc gradient, update weights 
        probas = predict_proba(X_tr, weights)
        gw = (1.0 / X_tr.shape[0]) * np.dot(X_tr.T, (probas - Y_tr))
        gw += (alpha * weights) / X_tr.shape[0] 
        # regularisation 
        weights -= lr * gw 
        # update weights 
        train_loss = np.linalg.norm(gw) 
        cur_dev_loss = binary_loss(X_dev, Y_dev, weights) 
        training_loss_history.append(train_loss) 
        validation_loss_history.append(cur_dev_loss) 
        if print_progress:
            print(f'iter: {iteration}, train_loss: {train_loss}, dev_loss: {cur_dev_loss}') 
        loss_diff = cur_dev_loss - pre_dev_loss 
        pre_dev_loss = cur_dev_loss 
        # early stopping 
        if abs(loss_diff) < tolerance: 
            break 
    return weights, training_loss_history, validation_loss_history

