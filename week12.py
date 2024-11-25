from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
iris = datasets.load_iris()
x = iris.data
y = iris.target
print("Feature names: ", iris.feature_name)
print("Target names: ", iris.feature_names)
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
svm_classifier = SVC(kernel='rbf', gamma = 'scale', c=1.0, random_state=42)
svm_classifier.fit(X_train,y_train)
y_pred = svm_classifier.predict(X_test)
accuracy = accuracy_score(y_test,y_prod)
print("Accuracy: {accuracy*100: 2f}%")
print(classification_report(y_test,y_pred,target_names=iris.target_names))
conf_matrix = confusion_matrix(y_test,y_pred)
print("\n Confusion matrix")
print(conf_matrix)
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix,annot=True,funt='d', cmap='YlGnBu',xticklabels=iris.target_names,yticklabels=iris.target_names)
plt.title('Confusion Matrix for Iris Dataset')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
