# stacked generalization with linear meta model on blobs dataset
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from keras.models import load_model
from keras.utils import to_categorical
from numpy import dstack
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, precision_score, f1_score
from keras.utils import plot_model
from sklearn.model_selection import GridSearchCV

grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge
metrics_stacking_file = 'models_DNN/select/metrics_stacking_file.csv'

# load models from file
def load_all_models(n_models):
	all_models = list()
	for i in range(n_models):
		# define filename for this ensemble
		filename = 'models_DNN/select/model_' + str(i + 1) + '.h5'
		# load model from file
		model = load_model(filename)
		# add to list of members
		all_models.append(model)
		print('>loaded %s' % filename)
	return all_models


# create stacked model input dataset as outputs from the ensemble
def stacked_dataset(members, inputX):
	stackX = None
	for model in members:
		# make prediction
		yhat = model.predict(inputX, verbose=0)
		# stack predictions into [rows, members, probabilities]
		if stackX is None:
			stackX = yhat
		else:
			stackX = dstack((stackX, yhat))
	# flatten predictions to [rows, members x probabilities]
	stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
	return stackX

# fit a model based on the outputs from the ensemble members
def fit_stacked_model(members, inputX, inputy):
    # create dataset using ensemble
    stackedX = stacked_dataset(members, inputX)
    # fit standalone model
    
#     model = LogisticRegression()
    model = KNeighborsClassifier()

#     model = LogisticRegression(C= 0.01,penalty="l2")

#     logreg_cv=GridSearchCV(model,grid,cv=10)
#     logreg_cv.fit(stackedX,inputy)

#     print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
#     print("accuracy :",logreg_cv.best_score_)
    
    model.fit(stackedX, inputy)
    
    return model


# make a prediction with the stacked model
def stacked_prediction(members, model, inputX):
	# create dataset using ensemble
	stackedX = stacked_dataset(members, inputX)
	# make a prediction
	yhat = model.predict(stackedX)
	return yhat



trainX, testX, trainy, testy = train_test_split(X, y, test_size = 0.2)

print(trainX.shape, testX.shape)
# load all models
n_members = 3
members = load_all_models(n_members)
print('Loaded %d models' % len(members))
# evaluate standalone models on test dataset
for model in members:
    
    # evaluate the model
    y_pred = model.predict(testX)
    y_pred =(y_pred > 0.5)
    report = classification_report(testy, y_pred, output_dict=True)
    df = pd.DataFrame(report).reset_index().transpose()

#     testy_enc = to_categorical(testy)
    _, acc = model.evaluate(testX, testy, verbose=0)
#     auc = roc_auc_score(testy, y_pred)

#     print("---------------------")
#     print('Model Accuracy: %.3f' % acc)
#     print(f"Precision:{round(precision_score(testy, y_pred),5)}")
#     print(f"Recall   :{round(recall_score(testy, y_pred),5)}")
#     print(f"F1-score :{round(f1_score(testy, y_pred),5)}")
#     print("---------------------")
#     print(model.name)
    
    
    ############## export to csv file ################
    with open(metrics_stacking_file, 'a', encoding='utf-8') as f:
        f.write("Mô hình: DNN %s \n" %model.name)  
        df.to_csv(metrics_stacking_file, mode='a', header=False)
        
        f.write(f"Precision:  {round(precision_score(testy, y_pred),5)}\n")
        f.write(f"Recall   : {round(recall_score(testy, y_pred),5)}\n")
        f.write(f"F1-score : {round(f1_score(testy, y_pred),5)}\n")
        f.write('Model Accuracy: %.3f\n' % acc)
        f.write(f"ROC : {round(roc_auc_score(testy, y_pred),4)}\n")
        f.write("\n\n\n")
    f.close()
    
    ##############################
    
# fit stacked model using the ensemble
model = fit_stacked_model(members, testX, testy)

# filename = 'models_DNN/select/stacking_model.h5'
# model.save(filename)
    
# evaluate model on test set
yhat = stacked_prediction(members, model, testX)
acc = accuracy_score(testy, yhat)
print(classification_report(testy, yhat))

print('Stacked Test Accuracy: %.3f' % acc)
print(f"Precision:{round(precision_score(testy, yhat),5)}")
print(f"Recall   :{round(recall_score(testy, yhat),5)}")
print(f"F1-score :{round(f1_score(testy, yhat),5)}")


############## export to csv file ################
report = classification_report(testy, y_pred, output_dict=True)
df = pd.DataFrame(report).reset_index().transpose()
    
with open(metrics_stacking_file, 'a', encoding='utf-8') as f:
    f.write('Stacking\n')
    df.to_csv(metrics_stacking_file, mode='a', header=False)
    
    f.write(f"Precision: {round(precision_score(testy, yhat),5)}\n")
    f.write(f"Recall   : {round(recall_score(testy, yhat),5)}\n")
    f.write(f"F1-score : {round(f1_score(testy, yhat),5)}\n")
    f.write('Stacked Test Accuracy: %.3f\n' % acc)
    f.write(f"ROC : {round(roc_auc_score(testy, yhat),4)}\n")
    
    f.write("\n\n\n")
f.close()