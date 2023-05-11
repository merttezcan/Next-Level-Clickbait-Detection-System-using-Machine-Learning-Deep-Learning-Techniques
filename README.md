# Next Level Clickbait Detection System using Machine Learning and Deep Learning Techniques

This project aims to detect clickbait headlines using machine learning and deep learning techniques. The dataset used in this project is taken from [Kaggle](https://www.kaggle.com/datasets/suleymancan/turkishnewstitle20000clickbaitclassified).

# Requirements

To run this project, you need to have the following libraries installed:

- pandas
- numpy
- matplotlib
- seaborn
- re
- string
- nltk
- sklearn
- scipy
- tensorflow
- keras


# Results

We trained and tested several machine learning and deep learning models, including Logistic Regression, Naive Bayes, Support Vector Machine, Random Forest, LSTM and Bidirectional LSTM. The best performing model was Linear Support Vector Classifier, with an accuracy of 88%.

# Future Work

There are several ways we can improve the performance of our model:

- We can try to do better hyperparameter tuning which also can take a lot of time.
- Most importantly, to use LSTM or and other deep learning model, we need a more comprehensive dataset.

# Conclusion

In this project, we started by exploring the given dataset and performing some basic data preprocessing tasks, such as removing duplicates, filling in missing values, etc. We then proceeded with feature engineering and to experiment with several machine learning models, including Logistic Regression, LinearSVC, Naive Bayes, and Random Forests.

After comparing the performance of these models based on their accuracy, precision, recall, and F1 score, we selected LinearSVC as the best model for this task, with an F1 score of 0.88. We then fine-tuned the model using GridSearchCV to optimize its hyperparameters, however the result was close.

Next, we experimented with deep learning models, specifically LSTM and Bidirectional LSTM, to see if they could outperform the LinearSVC model. However, even after trying tuning their hyperparameters (manually), their validation accuracies were not better than the LinearSVC model.

Based on these results, we can conclude that the given dataset may not be complex enough to require the use of deep learning models. LinearSVC, a simple yet powerful machine learning model, was able to achieve high accuracy and F1 score. However, it's important to note that this conclusion may be specific to this dataset, and other datasets may require more complex models to achieve good performance.

Although the bidirectional LSTM had a lower performance than the LinearSVC, it has the advantage of being able to capture the sequential dependencies in the text data. Therefore, it could be worth exploring other ways to improve the performance of the bidirectional LSTM, such as using pre-trained embeddings or more complex architectures.

Furthermore, we observed that the size of the dataset could be a limiting factor for the performance of these models. In future work, collecting more data could improve the performance of the models.
