# DL- Developing a Neural Network Classification Model using Transfer Learning

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## Problem Statement and Dataset

### Problem Statement
To develop an image classification model using **Transfer Learning** with the **VGG19 architecture** for classifying images from the given dataset.

### Dataset
The dataset consists of image samples organized into **training** and **testing** folders with different class categories.

<img width="438" height="182" alt="image" src="https://github.com/user-attachments/assets/7d26d902-7ae5-436d-9487-c2c1576e2a3c" />

---

## Neural Network Model
<img width="715" height="517" alt="image" src="https://github.com/user-attachments/assets/99335698-c3c4-4c5d-95ee-e310d2f85ad4" />


---

## DESIGN STEPS

### STEP 1:
Load and preprocess the dataset by resizing images and converting them into tensors.

### STEP 2:
Load the pre-trained **VGG19 model** and modify the final fully connected layer according to the dataset classes.

### STEP 3:
Freeze the feature extraction layers and define the loss function and optimizer.

### STEP 4:
Train the model using the training dataset and validate using the testing dataset.

### STEP 5:
Evaluate the model performance using confusion matrix and classification report.

### STEP 6:
Predict the output for new sample images and compare actual and predicted results.

---

## PROGRAM

### Name: Vikamuhan Reddy

### Register Number: 212223240181

```python
# Load Pretrained Model and Modify for Transfer Learning
model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

from torchsummary import summary
summary(model, input_size=(3, 224, 224))


# Modify the final fully connected layer to match the dataset classes
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)

model = model.to(device)

for param in model.features.parameters():
    param.requires_grad = False

summary(model, input_size=(3, 224, 224))


# Include the Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)


# Train the model
def train_model(model, train_loader, test_loader, num_epochs=10):

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):

        model.train()

        running_loss = 0.0

        for images, labels in train_loader:

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels.unsqueeze(1).float())

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        train_losses.append(running_loss / len(train_loader))

        model.eval()

        val_loss = 0.0

        with torch.no_grad():

            for images, labels in test_loader:

                images, labels = images.to(device), labels.to(device)

                outputs = model(images)

                loss = criterion(outputs, labels.unsqueeze(1).float())

                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')
```

### OUTPUT

## Training Loss, Validation Loss Vs Iteration Plot
<img width="761" height="546" alt="image" src="https://github.com/user-attachments/assets/8533bf94-0f2b-477e-8179-3a4ea763bbe3" />


## Confusion Matrix
<img width="732" height="576" alt="image" src="https://github.com/user-attachments/assets/f2b1c6b2-18db-45f5-a153-66f379c48861" />


## Classification Report
<img width="461" height="208" alt="image" src="https://github.com/user-attachments/assets/85439509-6103-4465-a03a-9827d6938290" />


### New Sample Data Prediction
<img width="466" height="452" alt="image" src="https://github.com/user-attachments/assets/175ca8a7-16c2-44f6-aa66-0dc61eb22ec4" />


## RESULT
Thus, an image classification model using **Transfer Learning with VGG19 architecture** was successfully developed and trained for the given dataset.
