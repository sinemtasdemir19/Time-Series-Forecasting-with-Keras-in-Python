# Time-Series-Forecasting-with-Keras-in-Python

In this document of my project , I’ll explain three different artificial neural network (ANN) architectures (DNN, RNN and LSTM) on the same univariate dataset — daily spend of an e-commerce company.
First, I will explain some concepts and why we use them.
## 1.	Time Series
Imagine you have a set of data points that were collected at different time intervals. This collection of data, where each data point is associated with a specific time, is called a time series. For example, you might have data on the daily temperature for the past month, where each data point represents the temperature recorded on a specific day.

Time series forecasting is like predicting what will happen in the future based on patterns and trends observed in the past. Using our example of daily temperature data, time series forecasting would involve trying to predict the temperature for future days based on the temperature patterns we have observed in the past.

To do this, we analyze the historical data to identify any regular patterns, such as seasonal changes or trends. We then use statistical techniques or machine learning algorithms to make predictions about future values in the time series. These predictions can be valuable for planning and decision-making in various fields, such as finance, weather forecasting, or sales forecasting.

In essence, time series and time series forecasting are about understanding how data changes over time and using that understanding to make predictions about future values in the series. It's like looking at the past to predict what will happen in the future.

## 2.	Deep Learning
Deep learning is a subset of machine learning, which is a field of artificial intelligence (AI). It involves training artificial neural networks (inspired by the structure of the human brain) to learn and make predictions from data. Just like our brain has interconnected neurons, deep learning models have layers of artificial neurons called "neural networks."

Deep Learning for Forecasting: Deep learning can be applied to time series forecasting, where we want to predict future values based on historical data patterns. By using deep learning techniques, we can train neural networks to analyze complex patterns and relationships in time series data and make accurate predictions.
To apply deep learning for time series forecasting, we first prepare the data and design a specific neural network architecture. This architecture determines the structure of the network, including the number of layers and neurons. We then train the network by feeding it historical time series data and adjusting the weights and biases of the neurons. Once trained, the network can make predictions for future time points by using the patterns it has learned from the historical data.

Deep learning for forecasting has the advantage of being able to capture complex patterns and nonlinear relationships in time series data. It can handle large and high-dimensional datasets, and it automatically extracts relevant features from the data, reducing the need for manual feature engineering. By using deep learning for forecasting, we can improve the accuracy and reliability of our predictions, helping us make informed decisions and plans based on the expected future behavior of the time series data.

In summary, deep learning is a technique in AI that involves training neural networks to learn from data, and deep learning for forecasting applies this technique specifically to make accurate predictions about future values in time series data.


## 3.	Keras
Keras is a deep learning library written in Python and supports different backends like TensorFlow, Theano or CNTK. Keras is a tool specifically used for rapid prototyping of deep learning models. Keras presents a set of neural network models and techniques used to solve time series forecasting problems.

Here are some advantages of using Keras for time series forecasting:

A high-level API: Keras simplifies tasks such as model creation, compilation, and training.

Flexible architecture: It allows combining different neural network layers to create complex model structures.

Automatic hyperparameter tuning: Keras supports Hyperopt, which automatically tunes hyperparameters for improved model performance.

For Time series forecasting we can use Keras by following the steps below:

a.	Data Preparation: The first step is to prepare the time series data you will use. Operations such as data cleaning, scaling, and feature engineering can be performed in this step.

b.	Model Creation: To create a model with Keras, you must first create a model object. You can add layers to this model using either the sequential API or the functional API. For example, you can add layers such as Input layer, Recursive Neural Network (RNN) layer, and Output layer sequentially.

c.	Model Compilation: After creating the model, it is necessary to compile the model. In this step, you must specify the loss function, optimization algorithm and evaluation metrics.

d.	Model Training: You can use your readily available time series data to train the compiled model. By feeding the training data into the model, you can fit the model to the data. Keras performs the training loop through the fit() function.

e.	Making Predictions: You can make predictions using the trained model. For this, you can predict future values by feeding existing data into the model.

f.	Model Evaluation: You can use validation data to evaluate the performance of the model you created. In this step, you can calculate the model's accuracy, error, or other evaluation metrics.

## 4.	Dickey-Fuller Test
The Dickey-Fuller test is a statistical test used to assess the stationarity property of a time series. It examines the presence or absence of a unit root to determine whether the time series is stationary or non-stationary.

## 5. ANN (Artificial Neural Network)
ANN is a computational model inspired by the biological neural networks in the human 
brain. ANN consists of interconnected nodes, also known as neurons, that process and 
transmit information. It is a versatile machine learning algorithm used for tasks such as 
pattern recognition, classification, regression, and data analysis. ANN can have multiple 
layers, including input, hidden, and output layers, and the connections between neurons 
are weighted. 
Now, I will explain three different neural network architectures one by one and show 
these three different neural network architectures on the same univariate dataset.

• DNN (Deep Neural Network)

• RNN (Recurrent Neural Network)

• LSTM (Long Short-Term Memory)

### 5.1. DNN (Deep Neural Network)
DNN is a type of ANN that has multiple hidden layers between the input and output 
layers. The depth of the network refers to the number of hidden layers. DNNs are 
capable of learning hierarchical representations of data and can handle complex tasks 
by learning abstract features at different levels. They are widely used in image 
recognition, speech recognition, natural language processing, and many other 
applications.
### 5.2. RNN (Recurrent Neural Network)
RNN is a type of neural network designed for sequential data or time series analysis. 
Unlike feedforward neural networks like DNNs, RNNs have feedback connections that 
allow information to flow in a loop. This loop enables RNNs to capture temporal 
dependencies and process data with sequential or time-based patterns. RNNs are 
commonly used in tasks such as language modeling, speech recognition, machine 
translation, and sentiment analysis.
### 5.3. LSTM (Long Short-Term Memory)
LSTM is a specialized type of RNN that addresses the problem of capturing long-term 
dependencies. Regular RNNs often struggle to remember information from distant past 
time steps, but LSTMs overcome this limitation. They achieve this by introducing a 
memory cell and various gates that control the flow of information. The gates, such as 
input, output, and forget gates, regulate the information flow and help LSTMs 
selectively retain or forget information over long sequences. LSTMs are particularly 
useful in tasks that involve long-range dependencies, such as language modeling, speech 
recognition, and sentiment analysis.
## 6. Conclusion
In conclusion, time series forecasting with Keras and deep learning techniques offers a 
powerful tool for predicting future values based on historical patterns. By leveraging neural 
network architectures such as DNN, RNN, and LSTM, we can uncover complex relationships 
and make accurate predictions in various domains.
