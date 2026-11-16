# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
The Iris dataset consists of 150 samples from three species of iris flowers (Iris setosa, Iris versicolor, and Iris virginica). Each sample has four features: sepal length, sepal width, petal length, and petal width. The goal is to build a neural network model that can classify a given iris flower into one of these three species based on the provided features.

## Neural Network Model
<img width="867" height="877" alt="image" src="https://github.com/user-attachments/assets/112ffd62-906c-496d-a0a4-ffe87c48f21a" />


## DESIGN STEPS
### STEP 1: 
Import the required libraries such as PyTorch, NumPy, Scikit-learn, and Matplotlib, and load the Iris dataset.

### STEP 2: 
Preprocess the dataset by separating the four input features and the target labels. Split the dataset into training and testing sets.


### STEP 3: 
 Convert the training and testing data into PyTorch tensors and create DataLoaders for batch processing.


### STEP 4: 

Define the neural network model with an input layer of 4 neurons, two hidden layers containing 16 and 8 neurons, and an output layer of 3 neurons. Apply the ReLU activation function to the hidden layers.

### STEP 5: 
Initialize the model, define the Cross-Entropy Loss function, and use the Adam optimizer to train the model for the specified number of epochs.


### STEP 6: 
Evaluate the trained model using test data. Generate the confusion matrix and classification report to measure performance, and predict the species of a new iris flower sample.




## PROGRAM

### Name: S.Nithyasri

### Register Number: 212225040289
```
class IrisClassifier(nn.Module):
    def __init__(self, input_size):
        super(IrisClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 3)  # 3 output classes for Iris dataset

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def train_model(model, train_loader, criterion, optimizer, epochs):
       for epoch in range(epochs):
          model.train()
          for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')                                                                             # Initialize model, loss function, and optimizer


model = IrisClassifier(input_size=X.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

```

### Dataset Information
<img width="806" height="637" alt="Screenshot 2026-11-16 174357" src="https://github.com/user-attachments/assets/7a61f44a-e090-46ce-b11f-c107fc6b6afe" />


### OUTPUT

## Confusion Matrix

<img width="916" height="832" alt="Screenshot 2026-11-16 174224" src="https://github.com/user-attachments/assets/b9248e63-9811-4326-887e-d19f2054a7b3" />


## Classification Report
<img width="797" height="522" alt="Screenshot 2026-11-16 174317" src="https://github.com/user-attachments/assets/4ff721ba-9cac-4306-ac28-9d0f9217352a" />


### New Sample Data Prediction
<img width="1188" height="252" alt="Screenshot 2026-11-16 174808" src="https://github.com/user-attachments/assets/043592b5-0aa5-4f4c-a6ea-927043200a5e" />


## RESULT
Thus, a neural network classification model was developed and trained successfully using the Iris dataset. The model classifies iris flowers into Iris setosa, Iris versicolor, and Iris virginica based on four input features. Its performance was evaluated using a confusion matrix and classification report, and predictions were made for new sample data.
