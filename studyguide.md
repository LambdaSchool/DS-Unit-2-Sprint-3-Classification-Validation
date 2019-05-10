Logistic Regression
-baseline classification model
model = LogisticRegression()
classifiers come with a .predict_proba, which gives the predicted probability of both outcomes
so .predict will give you the models answer, and .predict_proba will give you the reason behind the answer
.score returns accuracy in classifiers. That's correct_predictions/total_predictions
sklearn has cross_val_score(model,x,y,cv=number of crossval groups), which seems to cross validate the training data

category encoder
-OneHotEncoder will take categorical features and give each of the different observations their own column represented as booleans

Feature importance
A really cool graph showing feature importance
log_reg.fit(X_train_imputed, y_train)
coefficients = pd.Series(log_reg.coef_[0], X_train_encoded.columns)
plt.figure(figsize=(10,10))
coefficients.sort_values().plot.barh(color='grey');

Scalers
scalers put the numerical features into the same range
MinMaxScaler()
StandardScaler()

Pipeline
combines all that shit like category encoders, scalers, and even the model into one thing.
make_pipeline()
there's also pipeline() but make_pipeline() is simpler


Baselines and Validation
Baseline is the thing youre measuring against. It's like your made up standard that you gotta beat.
One baseline that you can do in time features is say that the value will be the same as it was the day/hour/minute/etc. before

To get a baseline for classifiers, find your majority class aka whichever there is more of
y_train.value_counts(normalize=True)
majority_class = y_train.mode()[0]
y_pred = [majority_class] * len(y_val)
majority_class = y_train.mode()[0]
y_pred = [majority_class] * len(y_val)

ROC AUC will tell you how much better you are than guessing
roc_auc_score(y_val,y_pred)

DO THINGS FAST
X_train.isnull().sum().sort_values()
look for leakage with shallow trees
DecisionTreeClassifier(x,y,max_depth=1)
Leakage is stuff from the future. I guess features that give too much info.

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=2)
tree.fit(X_train_numeric, y_train)
y_pred_proba = tree.predict_proba(X_val_numeric)[:,1]
#[:,1] <-gives only the positive predictions
roc_auc_score(y_val,y_pred_proba)

import graphviz
from sklearn.tree import export_graphviz
dot_data = export_graphviz(tree, out_file=None, feature_names=X_train_numeric.columns, 
                           class_names=['No', 'Yes'], filled=True, impurity=False, proportion=True)
graphviz.Source(dot_data)

OneHotEncoder is good for low cardinality: few unique features. 10 is low

log regression is a good baseline too.

classification metrics: 
accuracy: (true pos + true neg)/(pred neg + pred post) pred is the true and false combined. 
precision:True pos/(false pos+ true pos)
recall: true pos/ all pos. all pos is true pos and false neg 
f1: 2*precision*recall/(precision+recall). higher is better 
rocauc
confusion matrix
y_pred_proba = cross_val_predict(pipeline, X_train, y_train, cv=3, n_jobs=-1, 
                                 method='predict_proba')[:,1]
from sklearn.metrics import classification_report, confusion_matrix

threshold = 0.5
y_pred = y_pred_proba >= threshold

print(classification_report(y_train, y_pred))

confusion_matrix(y_train, y_pred)

pd.DataFrame(confusion_matrix(y_train, y_pred), 
             columns=['Predicted Negative', 'Predicted Positive'], 
             index=['Actual Negative', 'Actual Positive'])
imbalanced classes: to deal with imbalance, we give things weight. goes quite well visually. the fewer things
weigh more so that there is balance on the scale.

n_samples = 1000 (number of data points)
weights = (0.95, 0.05)
class_sep = 0.8 (don't know what this is)

X, y = make_classification(n_samples=n_samples, n_features=2, n_informative=2, 
                           n_redundant=0, n_repeated=0, n_classes=2, 
                           n_clusters_per_class=1, weights=weights, 
                           class_sep=class_sep, random_state=0)
                           
class_weight = None

model = LogisticRegression(solver='lbfgs', class_weight=class_weight)
