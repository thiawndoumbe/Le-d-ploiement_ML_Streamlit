import matplotlib.pyplot as plt
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier

def K_choice(X_train, y_train,X_test,y_test):
  error_rate = []
  for i in range(1,40):
   knn = KNeighborsClassifier(n_neighbors=i)
   knn.fit(X_train,y_train)
   pred_i = knn.predict(X_test)
   error_rate.append(np.mean(pred_i != y_test))

  plt.figure(figsize=(10,6))
  plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', 
           marker='o',markerfacecolor='red', markersize=10)
  plt.title('Error Rate vs. K Value') 
  plt.xlabel('K')
  plt.ylabel('Error Rate')
  # print("Minimum error:-",min(error_rate),"at K =",error_rate.index(min(error_rate)))

import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')