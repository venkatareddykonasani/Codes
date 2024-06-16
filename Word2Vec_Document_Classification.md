# Streamlining Complaint Management at the CFPB

🔗 [Consumer Financial Protection Bureau (CFPB)](https://www.consumerfinance.gov/)

🤝 The **Consumer Financial Protection Bureau (CFPB)** serves as a mediator between financial institutions and consumers, helping to resolve disputes when complaints are filed. 

🛠️ To enhance both efficiency and accuracy in processing customer complaints, the CFPB aims to implement automatic systems. These systems will classify and route complaints to the relevant teams based on the content of the complaint and the financial product involved.

🔗 [Kaggle Dataset for Bank Customer Complaint Analysis](https://www.kaggle.com/datasets/adhamelkomy/bank-customer-complaint-analysis/data)



```python
## Step 1: Load the Dataset

# Download the dataset from the given URL using wget.
!wget https://github.com/venkatareddykonasani/Datasets/raw/master/Bank_Customer_Complaints/complaints_v2.zip

# Unzip the downloaded dataset file. The -o flag ensures that existing files are overwritten.
!unzip -o complaints_v2.zip

# Read the CSV file into a pandas DataFrame.
complaints_data = pd.read_csv("/content/complaints_v2.csv")

# Display the first few rows of the DataFrame to understand its structure.
complaints_data.head()

## Step 2: Pre-processing

# Take a random sample of 50% of the data to build the model quickly. 
# The random_state parameter ensures reproducibility of the random sampling.
data = complaints_data.sample(frac=0.5, random_state=42)

# Print the shape of the sampled data to understand its size.
print("Shape", data.shape)

# Print the count of each unique value in the 'product' column.
# This helps to understand the distribution of different products in the dataset.
print(data['product'].value_counts())

# Convert all values in the 'text' column to strings and store them in a new column 'processed_text'.
# This ensures that all data is in a consistent format for further text processing.
data['processed_text'] = data['text'].astype(str)
```

### Explanation

#### Step 1: Load the Dataset
- **Downloading the Dataset**: The `wget` command is used to download the dataset file from a specified URL.
- **Unzipping the Dataset**: The `unzip` command extracts the contents of the downloaded ZIP file. The `-o` flag overwrites any existing files with the same name.
- **Reading the CSV File**: The `pd.read_csv` function from the pandas library reads the CSV file into a DataFrame called `complaints_data`.
- **Displaying the Data**: The `head()` method is used to display the first few rows of the DataFrame to get an overview of its structure and contents.

#### Step 2: Pre-processing
- **Sampling the Data**: A random sample of 50% of the data is taken using the `sample` method with `frac=0.5`. The `random_state` parameter ensures that the same sample can be obtained again if needed.
- **Understanding the Data Size**: The `shape` attribute of the DataFrame is printed to show the number of rows and columns in the sampled data.
- **Analyzing the 'Product' Column**: The `value_counts` method is used to print the distribution of different product types in the sampled data.
- **Processing Text Data**: The `astype(str)` method converts all entries in the 'text' column to strings, storing them in a new column 'processed_text'. This step is crucial for text processing tasks that follow.



```python
## Step 3: Prepare the Data for TensorFlow

# Initialize the Tokenizer to convert text into sequences of integers.
tokenizer = Tokenizer()

# Fit the tokenizer on the 'processed_text' column to learn the vocabulary.
tokenizer.fit_on_texts(data['processed_text'])

# Convert the text in 'processed_text' column to sequences of integers.
sequences = tokenizer.texts_to_sequences(data['processed_text'])

# Define the maximum length for the sequences. Any sequences longer than this will be truncated,
# and any sequences shorter will be padded.
max_length = 100  # Maximum length of a complaint narrative

# Pad or truncate the sequences so that they all have the same length of max_length.
X = pad_sequences(sequences, maxlen=max_length)

# The target variable 'product' is stored in y.
y = data['product']

# Initialize the LabelEncoder to convert categorical labels to integers.
label_encoder = LabelEncoder()

# Fit the label encoder on the 'product' column and transform the labels to integer encoded format.
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets. 
# 70% of the data will be used for training, and 30% will be used for testing.
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
```

### Explanation

#### Step 3: Prepare the Data for TensorFlow
- **Tokenizer Initialization**: The `Tokenizer` class from Keras is used to convert text into sequences of integers, where each unique word in the text is assigned a unique integer.
- **Fitting the Tokenizer**: The `fit_on_texts` method is called on the 'processed_text' column to build the vocabulary index based on the text data.
- **Text to Sequences**: The `texts_to_sequences` method converts the text data into sequences of integers.
- **Padding Sequences**: The `pad_sequences` function pads or truncates the sequences to ensure they all have the same length (`max_length`). This step is necessary because neural networks require input sequences to have a consistent length.
- **Label Encoding**: The `LabelEncoder` from scikit-learn is used to convert the categorical labels (product types) into integer values. This is important for training classification models.
- **Splitting the Data**: The `train_test_split` function from scikit-learn splits the data into training and testing sets. `test_size=0.3` means that 30% of the data will be used for testing, and `random_state=42` ensures the split is reproducible.


```python
## Step 4: Configure the model

# Calculate the vocabulary size. The vocabulary size is the total number of unique words in the text data plus one.
vocab_size = len(tokenizer.word_index) + 1  # Vocabulary size

# Initialize a Sequential model.
model = Sequential()

# Add an Embedding layer. This layer will learn word embeddings for the input sequences.
# - input_dim: Size of the vocabulary.
# - output_dim: Dimension of the dense embedding.
# - input_length: Length of input sequences.
model.add(Embedding(input_dim=vocab_size, output_dim=32, input_length=max_length))

# Add a GlobalAveragePooling1D layer. This layer calculates the average of all the embeddings in a sequence.
# This reduces the dimensionality and helps to prevent overfitting.
model.add(GlobalAveragePooling1D())

# Add a Dense layer with 64 units and ReLU activation. This layer acts as a hidden layer in the neural network.
model.add(Dense(64, activation='relu'))

# Add a Dense output layer with a number of units equal to the number of unique labels.
# The softmax activation function is used to output a probability distribution over the classes.
model.add(Dense(len(label_encoder.classes_), activation='softmax'))
```

### Explanation

#### Step 4: Configure the Model
- **Vocabulary Size Calculation**: The size of the vocabulary is calculated as the total number of unique words found by the tokenizer, plus one (to account for padding and out-of-vocabulary tokens).
- **Model Initialization**: A `Sequential` model from Keras is initialized. This type of model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.
- **Embedding Layer**: 
  - The `Embedding` layer is the first layer of the model. It takes the integer-encoded sequences and maps each integer (word index) to a dense vector of fixed size (32 in this case). 
  - `input_dim` is set to the vocabulary size.
  - `output_dim` specifies the size of the word vectors.
  - `input_length` specifies the length of input sequences, ensuring that all sequences are of the same length.
- **GlobalAveragePooling1D Layer**: 
  - This layer computes the average of the embeddings across all words in the sequence, resulting in a single vector for each sequence. This reduces the model's complexity and helps prevent overfitting.
- **Dense Layer (Hidden Layer)**: 
  - A fully connected (Dense) layer with 64 units and ReLU activation. This hidden layer helps the model learn more complex representations.
- **Dense Layer (Output Layer)**: 
  - The output layer has a number of units equal to the number of unique classes (products) in the dataset, and uses the softmax activation function. This outputs a probability distribution over the classes for multi-class classification.

  Here's the continuation of the Python code snippet with detailed comments added to explain each step, and the output formatted in Markdown:

```python
## Step 5: Train the Model

# Compile the model. The compile step specifies the optimizer, loss function, and evaluation metrics.
# - optimizer: 'adam' is a popular optimizer that adjusts the learning rate during training.
# - loss: 'sparse_categorical_crossentropy' is used for multi-class classification when labels are provided as integers.
# - metrics: 'accuracy' will track the accuracy of the model during training and evaluation.
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model using the training data.
# - X_train: Training input data.
# - y_train: Training target data.
# - epochs: Number of times to iterate over the training data.
# - batch_size: Number of samples per gradient update.
# - validation_data: Data on which to evaluate the loss and any model metrics at the end of each epoch.
model.fit(X_train, y_train, epochs=5, batch_size=128, validation_data=(X_test, y_test))

# Save the trained model's weights to a file.
model.save_weights('complaints_model.h5')

# Load the model weights from the saved file. This step can be used to reload the model for further evaluation or inference.
model.load_weights('complaints_model.h5')
```

### Explanation

#### Step 5: Train the Model
- **Model Compilation**: 
  - `optimizer='adam'`: The Adam optimizer is used because it is efficient and performs well in practice.
  - `loss='sparse_categorical_crossentropy'`: This loss function is suitable for multi-class classification problems where the target labels are integers.
  - `metrics=['accuracy']`: Accuracy is used as the metric to evaluate the model's performance during training and testing.
- **Model Training**: 
  - `model.fit()`: This method trains the model on the training data (`X_train` and `y_train`).
  - `epochs=5`: The number of complete passes through the training dataset.
  - `batch_size=32`: The number of samples per batch of computation.
  - `validation_data=(X_test, y_test)`: Validation data is used to evaluate the model's performance on unseen data at the end of each epoch.
- **Saving the Model**: 
  - `model.save_weights('complaints_model.h5')`: The model's weights are saved to a file for future use.
- **Loading the Model**: 
  - `model.load_weights('complaints_model.h5')`: The saved weights are loaded back into the model. This is useful for continuing training or making predictions without retraining the model.



```python
## Step 6: Evaluate the Model

# Predict the classes for the test data.
# The model.predict method returns probabilities for each class, and np.argmax is used to get the class with the highest probability.
y_pred = np.argmax(model.predict(X_test), axis=1)

# Calculate the confusion matrix to evaluate the accuracy of the classification.
# The confusion matrix shows the number of correct and incorrect predictions for each class.
cm = tf.math.confusion_matrix(y_test, y_pred)
print(cm)

# Calculate the accuracy score by comparing the true labels with the predicted labels.
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Print the classification report, which includes precision, recall, f1-score, and support for each class.
print("Classification Report:")
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print(report)
```

### Explanation

#### Step 6: Evaluate the Model
- **Predicting Classes**:
  - `model.predict(X_test)`: The model predicts the probability distribution for each class for the test data.
  - `np.argmax(..., axis=1)`: For each sample, `np.argmax` finds the index of the class with the highest probability, which is the predicted class label.
- **Confusion Matrix**:
  - `tf.math.confusion_matrix(y_test, y_pred)`: This function computes the confusion matrix, which summarizes the performance of the classification model. Each row of the matrix represents the actual class, while each column represents the predicted class.
- **Accuracy Score**:
  - `accuracy_score(y_test, y_pred)`: This function calculates the accuracy of the model, which is the ratio of correctly predicted instances to the total instances.
- **Classification Report**:
  - `classification_report(y_test, y_pred, target_names=label_encoder.classes_)`: This function generates a detailed classification report that includes precision, recall, F1-score, and support for each class. The `target_names` parameter is used to map the encoded labels back to their original class names.


```python
# Making a prediction on new narrations

# Define a list of new complaint texts to predict their categories.
new_complaints = [
    "payment history missing credit report made mistake put account forbearance without authorization"
]

# Convert the new complaint texts into sequences of integers using the previously fitted tokenizer.
new_sequences = tokenizer.texts_to_sequences(new_complaints)

# Pad the sequences so that they all have the same length as the training data (max_length).
new_X = pad_sequences(new_sequences, maxlen=max_length)

# Predict the class probabilities for the new complaint sequences.
new_predictions = model.predict(new_X)

# Determine the predicted class for each new complaint by finding the index of the maximum probability.
pred_class = np.argmax(new_predictions, axis=1)

# Print the predicted class indices.
print(pred_class)
```

### Explanation

#### Making a Prediction on New Narrations
- **New Complaint Texts**:
  - A list `new_complaints` contains new complaint narratives that we want to classify using the trained model.
- **Text to Sequences**:
  - `tokenizer.texts_to_sequences(new_complaints)`: Converts the new complaint texts into sequences of integers based on the vocabulary learned during training.
- **Padding Sequences**:
  - `pad_sequences(new_sequences, maxlen=max_length)`: Ensures that the new sequences are padded or truncated to the same length (`max_length`) as the training data.
- **Predicting Class Probabilities**:
  - `model.predict(new_X)`: The model predicts the class probabilities for each new complaint.
- **Determining Predicted Classes**:
  - `np.argmax(new_predictions, axis=1)`: Finds the index of the highest probability for each prediction, which corresponds to the predicted class.
- **Printing Predicted Classes**:
  - `print(pred_class)`: Outputs the indices of the predicted classes for the new complaints.

