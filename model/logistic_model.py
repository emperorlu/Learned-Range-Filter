import numpy as np
 
def sigmoid(z):
    '''
    σ函数
    '''
    a = 1/(1+np.exp(-z))
    
    return a
 
def initialize_parameters(dim):
    '''
    初始化w和b参数
    dim: w的维度，在logistic回归中也等于输入特征的数量
    '''
    w = np.random.randn(dim,1)
    b = 0
    
    return w,b
 
def propagate(w,b,X,Y):
    '''
    正向和反向传播
    
    X:训练集的特征向量 形状为(num_px*num_px*3,m) num_px表示图像的尺寸,m表示训练样本的个数
    Y:训练集的标签 形状为(1,m)
    
    cost:logistic回归的成本函数
    dw:成本函数对w的导数
    db:成本函数对b的导数
    '''
    m = X.shape[1]
    A = sigmoid(np.dot(w.T,X)+b)
    cost = -1/m*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))
    
    dw = 1/m*np.dot(X,(A-Y).T)
    db = 1/m*np.sum(A-Y)
    
    cost = np.squeeze(cost)
    grads = {"dw":dw,
             "db":db}
    
    return grads,cost
 
def optimize(w,b,X,Y,num_iterations,learning_rate,print_cost = False):
    '''
    梯度下降算法
    
    num_iterations：迭代次数(梯度下降的次数) 
    learning_rate: 学习率α
    
    costs 记录迭代过程中的代价函数值，用来绘图，便于判断梯度下降是否正确
    '''
    costs = []
    
    for i in range(num_iterations):
        grads,cost = propagate(w,b,X,Y)
        
        dw = grads["dw"]
        db = grads["db"]
        
        w = w - learning_rate*dw
        b = b - learning_rate*db
        
        if i%100 == 0:
            costs.append(cost)
        
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" %(i, cost))
            
    params = {"w":w,
              "b":b}
    
    grads = {"dw":dw,
             "db":db}
    
    return params,grads,costs
 
def predict(w,b,X):
    '''
    根据训练得到的w,b进行预测
    '''
    m = X.shape[1]
    A = sigmoid(np.dot(w.T,X)+b)
    Y_prediction = np.zeros((1,m))
    
    for i in range(m):
        Y_prediction[0,i]=A[0,i]>0.5
    
    return Y_prediction
 
def model(X_train,Y_train,X_test,Y_test,num_iterations = 2000,learning_rate = 0.5,print_cost = False):
    '''
    整合每个函数
    
    X_train 训练集的特征向量
    Y_train 训练集的标签
    X_test 测试集样本的特征向量
    Y_test 测试集样本的标签
    '''
    w,b = initialize_parameters(X_train.shape[0])
    parameters,grads,costs = optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost)
    
    w = parameters["w"]
    b = parameters["b"]
    
    Y_prediction_train = predict(w,b,X_train)
    Y_prediction_test = predict(w,b,X_test)
    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d