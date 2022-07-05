# SVM implementation by python in scatter pots

* import the models

    import numpy as np
    import joblib
    from sklearn import svm
    import matplotlib.pyplot as plt

* pre-define the scatter pots

    x = [[20, 10], [4, 5], [18, 9], [10, 7], [9, 1], [13, 18]]
    x = np.array(x)
    y = [1, 1, 0, 0, 1, 0]
    y = np.array(y)

* training model

    model = svm.SVC(C=10, kernel='linear')
    model.fit(x, y)

* draw the scatter pots with their predicted genre

    X=[[20, 10], [4, 5], [18, 9], [10, 7], [9, 1], [13, 18]]
    
    # predict
    a = [[100, 100]]
    for i in range(len(X)):
        a_pre = model.predict([X[i]])
        if a_pre[0]==0:
            plt.scatter(np.array(X)[i:i+1,0],np.array(X)[i:i+1,1],s=40,c='b')
        else:
            plt.scatter(np.array(X)[i:i+1,0],np.array(X)[i:i+1,1],s=40,c='r')

* corresponding support vector

    Support_vector = model.support_vectors_
    print("Support_vector:", Support_vector)

* the parameters of Linear SVM

    w = model.coef_
    print("w:", w)
    b = model.intercept_
    print("b:", b)

* draw the result graphic

    if w[0, 1] != 0:
        xx = np.arange(0, 20, 0.1)
        # optimal boundary
        yy = -w[0, 0]/w[0, 1] * xx - b/w[0, 1]
        plt.scatter(xx, yy, s=4)
        # support vector
        # b1 = Support_vector[0, 1] + w[0, 0]/w[0, 1] * Support_vector[0, 0]
        # b2 = Support_vector[1, 1] + w[0, 0]/w[0, 1] * Support_vector[1, 0]
        # yy1 = -w[0, 0] / w[0, 1] * xx + b1
        # plt.scatter(xx, yy1, s=4)
        # yy2 = -w[0, 0] / w[0, 1] * xx + b2
        # plt.scatter(xx, yy2, s=4)
    else:
        xx = np.ones(100) * (-b) / w[0, 0]
        yy = np.arange(0, 10, 0.1)
        plt.scatter(xx, yy)
    plt.show()
