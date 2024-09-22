from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
from tools import plotROC, calculate_all_prediction, calculae_lable_prediction, calculate_label_recall, readData, plotROC_two
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.feature_selection import RFECV

def gbm(n_classes):
    X,y = readData()
    model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=10, random_state=42)

    rfecv = RFECV(estimator=model,  # Learner
                  min_features_to_select=2,  # The minimum number of features selected
                  step=1,  # The number of features removed
                  cv=StratifiedKFold(10),  # Number of cross-validations
                  scoring='accuracy',  # Evaluation criteria for learners
                  verbose=0,
                  n_jobs=-1
                  ).fit(X, y)

    X_RFECV = rfecv.transform(X)
    print("RFECV feature selection results")
    print("The number of valid features : %d" % rfecv.n_features_)
    print("All feature levels : %s" % list(rfecv.ranking_))

    selected_features = X.columns[rfecv.support_]
    print("RFE selected compounds")
    print(list(selected_features))
    print("Cross-validation score")
    print(rfecv.cv_results_["mean_test_score"])

    U=X[selected_features]
    X_train, X_test, Y_train, Y_test = train_test_split(U, y, test_size=0.2)
    model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=10, random_state=42)

    model.fit(X_train, Y_train)
    score = model.score(X_test, Y_test)
    print("Feature sorting results")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(X_train.shape[1]):
        print("%2d) %-*s %f" % \
              (f + 1, 30, selected_features[indices[f]], importances[indices[f]]))
    predictions=model.predict(X_test)
    cm = confusion_matrix(y_true=Y_test, y_pred=predictions)
    calculate_all_prediction(cm)
    calculae_lable_prediction(cm)
    calculate_label_recall(cm)
    scores = cross_val_score(model, X_train, Y_train, cv=10)  #cv is the number of iterations.
    print("Cross-validation score:")
    print(scores)
    print("The mean cross-validation score:",scores.mean())

    y_score = model.predict_proba(X_test)
    if n_classes == 3:
        plotROC(3,Y_test,y_score)
    else:
        plotROC_two(2,Y_test,y_score)