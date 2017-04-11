import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

def softmax(y):
    return np.exp(y)/np.sum(np.exp(y))



class RNN():
    def __init__(self, x_dim, h_dim, y_dim):
        self.Wxh = np.random.uniform(low=-2/(x_dim+2*h_dim), high=\
        2/(x_dim+2*h_dim), size=(h_dim,x_dim))
        self.Whh = np.random.uniform(low=-2/(x_dim+2*h_dim), high=\
        2/(x_dim+2*h_dim), size=(h_dim,h_dim))
        self.Bh = np.zeros((h_dim,1))
        self.Why = np.random.uniform(low=-2/(y_dim+h_dim), high=\
        2/(y_dim+h_dim), size=(y_dim,h_dim))
        self.By = np.zeros((y_dim,1))
        self.classes = x_dim
        self.h_dim = h_dim

    def feedfwd(self, inputs, h=None):
        if h is None:
            h = np.zeros_like(self.Bh)
        hidden_states = [h]
        outputs = []
        predictions = []
        for i in inputs:
                a = np.zeros((self.classes,1))
                a[i,0] = 1
                h = sigmoid(np.dot(self.Wxh,a)+np.dot(self.Whh,h)+self.Bh)
                y = softmax(np.dot(self.Why,h)+self.By)
                hidden_states.append(h)
                outputs.append(y)
                predictions.append(np.argmax(y))
        return hidden_states, outputs, predictions

    def backprop(self, inputs, targets, truncate=None, alpha=0.01):
        if truncate is None:
            truncate = len(inputs)-1
        assert truncate < len(inputs)

        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dBh, dBy = np.zeros_like(self.Bh), np.zeros_like(self.By)
        dhnext = np.zeros_like(self.Bh)

        states, outputs, ps = self.feedfwd(inputs)

        loss = 0
        for i in xrange(1, truncate):
            X = np.array(inputs[-i])
            H = np.array(states[-i])
            Hprev = states[-i-1]
            dy = np.copy(outputs[-i])
            loss += -np.log(dy[targets[i],0])
            dy[targets[-i],0] -= 1
            dWhy += np.dot(dy,H.T)
            dBy += dy
            dH = dhnext + np.dot(self.Why.T,dy)
            dBh += dH*H*(1-H)
            dWxh += np.dot(dBh, X.T)
            dWhh += np.dot(dBh, Hprev.T)
            dhnext = np.dot(self.Whh, (dH*H*(1-H)))
        for dparam in [dWxh, dWhh, dWhy, dBh, dBy]:
            np.clip(dparam, -5, 5, out=dparam)
        self.Wxh -= alpha*dWxh/truncate
        self.Whh -= alpha*dWhh/truncate
        self.Why -= alpha*dWhy/truncate
        self.Bh -= alpha*dBh/truncate
        self.By -= alpha*dBy/truncate
        return loss



X = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'*10
X = list(X)
X_set = list(set(X))
char_to_index = {x:i for i,x in enumerate(X_set)}
index_to_char = {i:x for i,x in enumerate(X_set)}
Xi = [char_to_index[x] for x in X]
Y = 'BCDEFGHIJKLMNOPQRSTUVWXYZA'*10
Y = list(Y)
Yi = [char_to_index[y] for y in Y]
#
classes = len(X_set)
R = RNN(classes,classes*2,classes)

for i in xrange(150000):
    loss = R.backprop(Xi,Yi, truncate=60)
    if i % 5000 == 0:
        print 'Iteration {0}:'.format(i)
        O = R.feedfwd(Xi)
        print ''.join([index_to_char[i] for i in O[2]])
        print loss

O = R.feedfwd(Xi)
print 'Iteration {0}:'.format(150000)
print ''.join(Y)
print ''.join([index_to_char[i] for i in O[2]])
#
# print char_to_index
