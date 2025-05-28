import numpy as np
import csv
from collections import Counter

# Function to calculate the Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Function to get the K nearest neighbors
def get_neighbors(training_data, test_point, k):
    distances = []
    for data_point in training_data:
        # Calculate the distance from the test point to the current training point
        distance = euclidean_distance(test_point[:-1], data_point[:-1])  # Exclude the label
        distances.append((data_point, distance))
    
    # Sort the distances by the second element (distance)
    distances.sort(key=lambda x: x[1])
    
    # Return the k nearest neighbors
    neighbors = [distance[0] for distance in distances[:k]]
    return neighbors

# Function to predict the class of a test point based on KNN
def predict(training_data, test_point, k):
    neighbors = get_neighbors(training_data, test_point, k)
    
    # Extract the labels of the neighbors
    labels = [neighbor[-1] for neighbor in neighbors]
    
    # Return the most common label (majority vote)
    most_common = Counter(labels).most_common(1)
    return most_common[0][0]

# Function to train and test the KNN model
def knn(training_data, test_data, k):
    predictions = []
    for test_point in test_data:
        # Predict the class for each test point
        prediction = predict(training_data, test_point, k)
        predictions.append(prediction)
    return predictions

# Function to load the dataset from a CSV file
def load_dataset_from_csv(csv_file):
    dataset = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row if there's one
        for row in reader:
            dataset.append([float(value) if i != len(row) - 1 else int(value) for i, value in enumerate(row)])
    return np.array(dataset)

# Split data into equal training and test sets
def split_data_equal(dataset):
    label=(len(dataset[0]))-1 #To find the last column of the data set which is our output
    x=([dataset[i] for i in range(len(dataset)) if dataset[i][label] == 1])
    y=([dataset[i] for i in range(len(dataset)) if dataset[i][label] == 0])
    train_size1 = int(len(x) *0.8)
    train_size2 = int(len(y) *0.8)
    train_x=x[:train_size1]
    test_x=x[train_size1:]
    train_y=y[:train_size2]
    test_y=y[train_size2:]
    test_data=[]
    train_data=[]
    for i in test_x:
        test_data.append(i)
    for i in test_y:
        test_data.append(i)
    for i in train_x:
        train_data.append(i)
    for i in train_y:
        train_data.append(i)
    #print(test_data)
    return train_data, test_data

def calc_accuracy(predictions,test_data):
    n=(len(predictions))
    label=(len(test_data[0]))-1 #To find the last column of the data set which is our output
    c1=c2=t1=t2=0
    for i in range(n):
        print(test_data[i][label]," : ",predictions[i])
        if(test_data[i][label]==0):
            t1+=1
            if(predictions[i]==test_data[i][label]):
                c1+=1
        else:
            t2+=1
            if(predictions[i]==test_data[i][label]):
                c2+=1
    print("c1=>",c1,"T1=>",t1)
    print("c2=>",c2,"T2=>",t2)
    print("Class 0 :",(c1/t1)*100,"% ")
    print("Class 1 :",(c2/t2)*100,"% ")
    print("OverAll :",((c1+c2)/n*100),"%")

# Example usage
if __name__ == "__main__":
    # Specify your CSV file path
    csv_file = 'final_data.csv'  # Replace with your actual CSV file path
    
    # Load the dataset from the CSV file
    dataset = load_dataset_from_csv(csv_file)
    
    # Split the data into equal training and testing sets
    train_data, test_data = split_data_equal(dataset)
    
    k = 4  # Number of neighbors

    # Get predictions for the test data
    predictions = knn(train_data, test_data, k)
    
    # Print the predictions
    print(f"Predictions: {predictions}")

    calc_accuracy(predictions,test_data)
