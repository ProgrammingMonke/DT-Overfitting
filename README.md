# Decision Trees, Overfitting, and Evaluating Performance
Mini-project to explore decision trees and overfitting, and learn about evaluating the performance of a classifier.

# Goal

To explore the impact of the following on the extent of overfitting:
* The size of the dataset (n in the call to make_dataset)
* The number of irrelevant features (d in the call to make_dataset)
* The probability of class noise (p in the call to make_dataset)
* The minimum number of samples required for a node to be split.  That is the min_samples_split parameter to the [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier) constructor


### Size of the dataset:
In my example, I start with a size of 10 and go up to a size of 5000. We see that as we increase the size of the dataset, the test accuracy improves significantly. Although the training data accuracy decreases slightly, it gets closer to the same value as the test data. As a result, we can say that a larger dataset is generally better for decision tree models. This is because it provides more examples for the model to learn from and reduces the risk of overfitting. With a larger dataset, the decision tree model can learn patterns that generalize well to unseen data. In contrast, a smaller dataset may not contain enough information for the model to accurately capture the underlying patterns and may lead to overfitting. 

![Screenshot of Size Output](https://github.com/ProgrammingMonke/DT-Overfitting/blob/main/assets/size_output.png)

### Number of irrelevant features: 
In my example, I start the number of irrelevant features at 1 and work them up to 5000, keeping everything else fixed. We see that the training data ends up getting 100% accuracy the more irrelevant features included and the test data accuracy drops. As a result, we find that including irrelevant features in the dataset can lead to overfitting. Decision trees may try to split on irrelevant features, which can lead to the model capturing noise in the data rather than the underlying patterns. Therefore, it is essential to identify and remove irrelevant features from the dataset to improve the performance of the decision tree model.

![Screenshot of Num Irrelevant Features Output](https://github.com/ProgrammingMonke/DT-Overfitting/blob/main/assets/numIrrelevant_output.png)

### Probability of class noise:
In my example, I increase the probability of class noise from 0.0-1.0. When the probability of noise is either 0.0 or 1.0, the test and training data are both 100% accuracy. Also, when the probability is 0.5, the accuracy for either drops significantly. The further the probability is from either 0 or 1, the worse the accuracy is. Class noise refers to the mislabeling of data points in the dataset. We find that if the probability of class noise is high or too low, it can lead to the decision tree model fitting the noise in the data, leading to overfitting.

![Screenshot of Class Noise](https://github.com/ProgrammingMonke/DT-Overfitting/blob/main/assets/probClassNoise_output.png)

### Minimum number of samples required for a node to be split:
In my example, we increase the minimum samples split parameter of the tree from 2 to 1000. As we increase the minimum samples split, the accuracy of the training and test data increases briefly, then sharply decreases significantly as both approach the same value. The minimum number of samples required for a node to be split is a hyperparameter that determines how fine-grained the decision tree can be. If this value is too small, the decision tree model may split too much, leading to overfitting. On the other hand, if this value is too large, the decision tree model may not be able to capture the underlying patterns in the data, leading to underfitting. We see this as the test data spikes and increases in accuracy when the minSampleSplit value is 4, but drops afterwards. It is therefore essential to tune this hyperparameter carefully to balance between underfitting and overfitting.

![Screenshot of Min Samples to Split](https://github.com/ProgrammingMonke/DT-Overfitting/blob/main/assets/minSamples_output.png)