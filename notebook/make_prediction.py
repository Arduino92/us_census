from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import preprocessing_census

def get_model_performances(results, test, empty_cols):
    print("Model parameters:")
    print(results.get_params())
    df_test, target_test, categorical_columns, numeric_columns = preprocessing_census.feature_transform(test, empty_cols)
    y_pred = results.predict(df_test)
    
    print("Output length : " + str(len(y_pred)))
    
    tn, fp, fn, tp = confusion_matrix(target_test.astype(int), y_pred.astype(int)).ravel()
    print(confusion_matrix(target_test.astype(int), y_pred.astype(int)))
    print('accuracy: ' + str(accuracy_score(target_test.to_numpy().astype(int), y_pred)))
    #fpr, tpr, thresholds = roc_curve(y_test.astype(int), y_pred.astype(int), pos_label=2)
    sensitivity = tp/(fp+tp)
    print('sensitivity: ' + str(sensitivity))
    specificity = tn/(fn+tn)
    print('specificity: ' + str(specificity))
    
    print('Best estimator : ' + str(results['gridsearchcv'].best_estimator_))
    
    print('Best score : ' + str(results['gridsearchcv'].best_score_))
    