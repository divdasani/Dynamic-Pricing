import autograd.numpy as np
from autograd import value_and_grad
from scipy.stats import gamma

def model(w,x):
    return np.dot(x[0].T,w)

def least_squares(w,x,y):    
    cost = np.sum((model(w,x) - y)**2)
    return cost

def gradient_descent(g,alpha,max_its,w,x,y):
    gradient = value_and_grad(g)

    weight_history = []      
    cost_history = []

    for k in range(1,max_its+1):
        cost_eval,grad_eval = gradient(w,x,y)
        
        weight_history.append(w)
        cost_history.append(cost_eval)

        w = w - alpha*grad_eval

    weight_history.append(w)
    cost_history.append(g(w,x,y))
    cost_history = np.asarray(cost_history)/x.shape[2]
    
    return weight_history,cost_history

def generate_sample(sample_size, gen_probs, weights):
    sample = np.random.rand(sample_size,len(gen_probs))
    for i in range(sample.shape[1]):
        sample[:,i] = [1 if x < gen_probs[i] else 0 for x in sample[:,i]]
    sample = np.vstack((np.ones((1,sample_size)),sample.T)).T
    value = []
    for i in range(sample_size):
        a = np.dot(weights,sample[i])
        val = gamma.ppf(np.random.uniform(),a,scale = 1/np.sqrt(a))
        value.append(np.round(val,2))
    return sample, value

def demand_curve(axis, value):
    demand = []
    sample_size = len(value)
    for price in axis:
        demand.append(sum(1 if x>=price else 0 for x in value)/sample_size)
    return demand

logistic_demand = lambda a,x0,x: 1/(1+np.exp(np.abs(a)*(x-x0)))

def APP_s(axis, value, c):
    app_s = []
    sample_size = len(value)
    demand = demand_curve(axis, value)
    for i in range(len(axis)):
        app_s.append(max(axis[i]-c,0)*demand[i])
    return app_s

def APP_d(w,x,y,c,d):
    est = d*model(w,x)
    ind = list(range(len(est)))
    rev = sum(max(est[i],c) if y[0][i] >= max(est[i],c) else 0 for i in ind)
    return rev/x.shape[2]
