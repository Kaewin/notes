def find_best_k_f1(X_train, y_train, X_test, y_test, min_k=1, max_k=25):
    best_k = 0
    best_score = 0.0
    for k in range(min_k, max_k+1, 2):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test)
        f1 = f1_score(y_test, preds)
        if f1 > best_score:
            best_k = k
            best_score = f1
    
    print("Best Value for k: {}".format(best_k))
    print("F1-Score: {}".format(best_score))
    return best_k

def find_best_k_ll(X_train, y_train, X_test, y_test, min_k=1, max_k=50):
    best_k = 0
    best_score = 0.125
    for k in range(min_k, max_k+1, 2):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        preds = knn.predict_proba(X_test)
        log_loss = -cross_val_score(knn, X_train, y_train, scoring="neg_log_loss").mean()
        if log_loss < best_score:
            best_k = k
            best_score = log_loss
    
    print("Best Value for k: {}".format(best_k))
    print("Log Loss: {}".format(best_score))
    return best_k