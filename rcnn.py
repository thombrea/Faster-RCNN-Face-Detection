import os, h5py, glob, re, argparse
import numpy as  np

###############################################################################
# Utility Functions

def rect_regression(rect, anchor):
    '''
    Convert a rectangle into a regression target, with respect to  a given anchor rect
    Returns an np array of distances between rect and anchor
    '''
    return(np.array([(rect[0]-anchor[0])/anchor[2],(rect[1]-anchor[1])/anchor[3],
                     np.log(rect[2]/anchor[2]),np.log(rect[3]/anchor[3])]))

def sigmoid(Xi):
    ''' 
    Sigmoid activation of a layer. Avoids underflows and NaNs.

    Input:
      Xi (N tokens, N1, N2, N dimensions) - excitation
    Output:
      H (N tokens, N1, N2, N dimensions) - sigmoid activation
    '''
    H = np.zeros(Xi.shape)
    H[Xi > -100] = 1/(1+np.exp(-Xi[Xi > -100]))
    return(H)

safe_log_min = np.exp(-100)
def safe_log(X):
    '''
    Compute safe logarithm, again to avoid underflows.
    Input:  
        X = np.array
    Output: 
        Y = np.array, safe logarithm of X
    '''
    Y = np.zeros(X.shape)
    Y[X > safe_log_min] = np.log(X[X > safe_log_min])
    return Y

def conv2(H, W, padding):
    '''
    Compute a 2D convolution.  Compute only the valid outputs, after padding H with 
    specified number of rows and columns before and after. Implementing this with numpy
    to avoid using scipy.

    Input:
      H (N1,N2) - input image
      W (M1,M2) - impulse response, indexed from -M1//2 
      padding (scalar) - number of rows and columns of zeros to pad before and after H

    Output:
      Xi  (N1-M1+1+2*padding,N2-M2+1+2*padding) - output image
         The output image is computed using the equivalent of 'valid' mode,
         after padding H  with "padding" rows and columns of zeros before and after the image.

    Xi[n1,n2] = sum_m1 sum_m2 W[m1,m2] * H[n1-m1,n2-m2]
    
    '''
    N1,N2 = H.shape
    M1,M2 = W.shape
    H_padded = np.zeros((N1+2*padding,N2+2*padding)) # 0 pad H
    H_padded[padding:N1+padding,padding:N2+padding] = H # fill in 0-padded H
    W_flipped_and_flattened = W[::-1,::-1].flatten() # flip W so that we can take an inner product
    Xi = np.empty((N1-M1+1+2*padding,N1-M1+1+2*padding))
    for n1 in range(N1-M1+1+2*padding):
        for n2 in range(N2-M2+1+2*padding):
            Xi[n1,n2] = np.inner(W_flipped_and_flattened, H_padded[n1:n1+M1,n2:n2+M2].flatten())
    return Xi

def conv_layer(H, W, padding):
    '''
    Compute a convolutional layer between input activations H and weights W. Again, implemented
    to avoid using PyTorch.

    Input:
      H (N1,N2,NC (number channels)) - hidden-layer activation from the previous layer
      W (M1,M2,NC,ND (number dimensions)) - convolution weights
      padding (scalar) - number of rows and columns of zeros to pad before and after H

    Output:
      Xi (_,_,ND) - excitations of the next layer
    
    Xi[:,:,d] = sum_c conv2(H[:,:,c], W[:,:,c,d])
    '''
    N1, N2, NC = H.shape
    M1, M2, NC, ND = W.shape
    Xi = np.zeros((N1-M1+1+2*padding, N2-M2+1+2*padding, ND)) # size after convolution
    for d in range(ND):
        for c in range(NC):
            Xi[:,:,d] += conv2(H[:,:,c], W[:,:,c,d], padding) # 2d convolution the input with the weights
    return Xi



###############################################################################
# RCNN Functions

def forwardprop(X, W1, W2):
    '''
    Compute forward propagation of the FasterRCNN network.

    Inputs:
      X (N1,N2,NC) - input features
      W1 (M1,M2,NC,ND) -  weight tensor for the first layer
      W2 (1,1,ND,NA,NY) - weight tensor for the second layer

    Outputs:
      H (N1,N2,ND) - hidden layer activations
      Yhat (N1,N2,NA,NY) - outputs

    Interpretation of the outputs:
      Yhat[n1,n2,a,:4] - regression output, (n1,n2) pixel, a'th anchor
      Yhat[n1,n2,a,4] - classfication output, (n1,n2) pixel, a'th anchor
    '''
    # First layer = 3x3 convolution, followed by ReLU nonlinearity
    N1,N2,NC = np.shape(X)
    t1,t2,ND,NA,NY = np.shape(W2)
    
    E1 = conv_layer(X,W1,1) # excitation after first layer = convolution of X, W1
    H = np.zeros(np.shape(E1))
    
    for n1 in range(N1):
        for n2 in range(N2):
            for d in range(ND):
                H[n1,n2,d] = max(0, E1[n1,n2,d]) # ReLU nonlinearity
    
    # Second layer = 1x1 convolution
    E2 = np.zeros((N1,N2,NA,NY))
    
    #1x1 convolution
    for a in range(NA):
        for k in range(NY):
            for d in range(ND):
                E2[:,:,a,k] += conv2(H[:,:,d], W2[:,:,d,a,k],0)
    
    '''
    #Matrix implementation
    for n1 in range(N1):
        for n2 in range(N2):
            for k in range(NY):
                E2[n1,n2,:,k] = np.transpose(W2[0,0,:,:,k]) @ H[n1,n2,:]
    '''
    
    # Last element = classification, sigmoid
    # Elements 0-last-1 = regression, just output of the layer
    Y_hat = np.zeros(np.shape(E2))
    Y_hat[:,:,:,NY-1] = sigmoid(E2[:,:,:,NY-1])
    Y_hat[:,:,:,0:NY-1] = E2[:,:,:,0:NY-1]
    
    return H, Y_hat
    
def detect(Yhat, number_to_return, anchors):
    '''
    Input:
      Yhat (N1,N2,NA,NY) - neural net outputs for just one image
      number_to_return (scalar) - the number of rectangles to return
      anchors (N1,N2,NA,NY) - the set of standard anchor rectangles
    Output:
      best_rects (number_to_return,4) - [x,y,w,h] rectangles most likely to contain faces.
      You should find the number_to_return rows, from Yhat,
      with the highest values of Yhat[n1,n2,a,4],
      then convert their corresponding Yhat[n1,n2,a,0:4] 
      from regression targets back into rectangles
      (i.e., reverse the process in rect_regression()).
    '''
    
    N1,N2,NA,NY = np.shape(Yhat)
    
    # Find the Yhats with the best probabilities (rectangles most likely to have faces)
    maxes = np.zeros((number_to_return))
    for n1 in range(N1):
        for n2 in range(N2):
            for a in range(NA):
                p = Yhat[n1,n2,a,4]
                if(p > np.amin(maxes)):
                    maxes[np.argmin(maxes)] = p # store max indices
    
    maxes = np.sort(maxes)[::-1] # sort from highest to lowest
    best_rects = np.zeros((number_to_return,4)) # want n rectangles defined by 4 pts
    
    for i in range(number_to_return):
        indeces = np.argwhere(Yhat == maxes[i])[0] #
        regression = Yhat[indeces[0], indeces[1], indeces[2], 0:4]
        rect = reg2rect(regression,anchors[indeces[0], indeces[1], indeces[2], 0:4])
        best_rects[i,:] = rect
    
    return best_rects

def reg2rect(regression, anchor):
    '''
    Undo rect to regression utility function above
    '''
    rect = np.zeros((4))
    
    rect[0] = regression[0] * anchor[2] + anchor[0]
    rect[1] = regression[1] * anchor[3] + anchor[1]
    rect[2] = np.exp(regression[2]) * anchor[2]
    rect[3] = np.exp(regression[3]) * anchor[3]
    
    return rect
    
def loss(Yhat, Y):
    '''
    Compute the two loss terms for the FasterRCNN network, for one image.

    Inputs:
      Yhat (N1,N2,NA,NY) - neural net outputs
      Y (N1,N2,NA,NY) - targets
    Outputs:
      bce_loss (scalar) - 
        binary cross entropy loss of the classification output,
        averaged over all positions in the image, averaged over all anchors 
        at each position.
      mse_loss (scalar) -
        0.5 times the mean-squared-error loss of the regression output,
        averaged over all of the targets (images X positions X  anchors) where
        the classification target is  Y[n1,n2,a,4]==1.  If there are no such targets,
        then mse_loss = 0.
    '''
    N1,N2,NA,NY = np.shape(Yhat)
    
    num = 0
    den = 0
    bce_loss = 0
    mse_loss = 0

    for n1 in range(N1):
        for n2 in range(N2):
            for a in range(NA):
                num += Y[n1,n2,a,4] * np.sum(np.square(Y[n1,n2,a,0:4] - Yhat[n1,n2,a,0:4]))
                den += Y[n1,n2,a,4]
                
                bce_loss += Y[n1,n2,a,4] * safe_log(Yhat[n1,n2,a,4])  + (1 - Y[n1,n2,a,4]) * safe_log(1 - Yhat[n1,n2,a,4])
    
    bce_loss /= (N1 * N2 * NA * -1.0)
    mse_loss = num/den * 0.5
    
    return bce_loss, mse_loss

def backprop(Y, Yhat, H, W2):
    '''
    Compute back-propagation in the Faster-RCNN network.
    Inputs:
      Y (N1,N2,NA,NY) - training targets
      Yhat (N1,N2,NA,NY) - network outputs
      H (N1,N2,ND) - hidden layer activations
      W2 (1,1,ND,NA,NY) - second-layer weights
    Outputs:
      GradXi1 (N1,N2,ND) - derivative of loss w.r.t. 1st-layer excitations
      GradXi2 (N1,N2,NA,NY) - derivative of loss w.r.t. 2nd-layer excitations
    '''
    N1,N2,NA,NY = np.shape(Yhat)
    ND = np.shape(H)[2]
    
    GradXi1 = np.zeros(np.shape(H))
    GradXi2 = np.zeros(np.shape(Yhat)) 
    
    GradXi2 = Yhat-Y
    
    dNonLin = np.zeros(np.shape(H))
    dNonLin[H != 0] = 1
    
    for n1 in range(N1):
        for n2 in range(N2):
            for d in range(ND):
                for a in range(NA):
                    for y in range(NY):
                        GradXi1[n1,n2,d] += GradXi2[n1,n2,a,y] * W2[0,0,d,a,y] * dNonLin[n1,n2,d]
    
    return GradXi1, GradXi2

def weight_gradient(X, H, GradXi1, GradXi2, M1, M2):
    '''
    Compute weight gradient in the Faster-RCNN network.
    Inputs:
      X (N1,N2,NC) - network inputs
      H (N1,N2,ND) - hidden-layer activations
      GradXi1 (N1,N2,ND) - gradient of loss w.r.t. layer-1 excitations
      GradXi2 (N1,N2,NA,NY) - gradient of loss w.r.t. layer-2 excitations
      M1 - leading dimension of W1
      M2 - second dimension of W1
    Outputs:
      dW1 (M1,M2,NC,ND) - gradient of loss w.r.t. layer-1 weights
      dW2 (1,1,ND,NA,NY) -  gradient of loss w.r.t. layer-2 weights
    '''
    NC = np.shape(X)[2]
    ND = np.shape(H)[2]
    N1,N2,NA,NY = np.shape(GradXi2)
    
    dW1 = np.zeros((M1,M2,NC,ND))
    dW2 = np.zeros((1,1,ND,NA,NY))
    
    t1 = int(M1/2) #recenter m1,m2 around 0
    t2 = int(M2/2)
    
    for m1 in range(M1):
        for m2 in range(M2):
            for c in range(NC):
                for d in range(ND):
                    for n1 in range(N1):
                        for n2 in range(N2):
                            if(n1-(m1-t1) >= 0 and n1-(m1-t1) < N1 and n2-(m2-t2) >=0 and n2-(m2-t2) < N2):
                                dW1[m1,m2,c,d] += GradXi1[n1-(m1-t1),n2-(m2-t2),d] * X[n1,n2,c]
                                
    
    for d in range(ND):
        for a in range(NA):
            for k in range(NY):
                for n1 in range(N1):
                    for n2 in range(N2):
                        dW2[0,0,d,a,k] += GradXi2[n1,n2,a,k] * H[n1,n2,d]
    
    return dW1, dW2

def weight_update(W1,W2,dW1,dW2,learning_rate):
    '''
    Input: 
      W1 (M1,M2,NC,ND) = first layer weights
      W2 (1,1,ND,NA,NY) = second layer weights
      dW1 (M1,M2,NC,ND) = first layer weights
      dW2 (1,1,ND,NA,NY) = second layer weights
      learning_rate = scalar learning rate
    Output:
      new_W1 (M1,M2,NC,ND) = first layer weights
      new_W2 (1,1,ND,NA,NY) = second layer weights
    '''
    new_W1 = W1 - learning_rate * dW1
    new_W2 = W2 - learning_rate * dW2
    
    return new_W1, new_W2

