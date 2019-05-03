import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import pandas as pd
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import torch.utils.data as data_utils
import torch.nn as nn
import torch.nn.functional as F

"""
    # CNN Model
    Net(
      (conv1): Conv2d(1, 10, kernel_size=(5, 5), stride=(1, 1))
      (maxpool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (conv2): Conv2d(10, 20, kernel_size=(5, 5), stride=(1, 1))
      (fc1): Linear(in_features=320, out_features=50, bias=True)
      (fc2): Linear(in_features=50, out_features=10, bias=True)
    )
"""
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) # 10 channels each of 24 * 24 (because 5*5 kernel applied to 28*28 image will give 24*24 features)
        self.maxpool = nn.MaxPool2d(2) # will reduce 24 * 24 features to 12 * 12 (divide by 2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) # will reduce 12 * 12 features to 8 * 8 when kernel of size 5*5 is applied
        self.fc1 = nn.Linear(320, 50) # 320 = 20 * 4 * 4, the 8 * 8 from above was maxpooled to 4 * 4
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # input data shape = [N = number of images, 1, 28, 28]
        # print("Input shape = " + str(np.shape(x)))
        x = self.conv1(x) # becomes [N, 1, 24, 24] after convolution filter with kernel size 5*5
        # print("After convolution 1 shape = " + str(np.shape(x)))
        x = F.relu(self.maxpool(x)) # becomes [N, 1, 12, 12] after maxpooling with kernel size 2
        # print("After maxpool 1 shape = " + str(np.shape(x)))
        x = self.conv2(x) # becomes [N, 1, 8, 8] after convolution filter with kernel size 5*5
        # print("After convolution 2 shape = " + str(np.shape(x)))
        x = F.relu(self.maxpool(x)) # becomes [N, 1, 4, 4] after maxpooling with kernel size 2
        # print("After maxpool 2 shape = " + str(np.shape(x)))
        x = x.view(-1, 320) # reshaped to [N, 1024] for fully connected layer (1024 = 4 * 4 * 64, 64 = output channels of prev convolution)
        # print("After reshaping = " + str(np.shape(x)))
        x = F.relu(self.fc1(x)) # becomes [N, 200] after fully connected layer with output 200
        # print("After FC1 shape = " + str(np.shape(x)))
        x = self.fc2(x) # becomes [N, 10] after fully connected layer with output 10
        # print("After FC2 shape = " + str(np.shape(x)))
        x = F.log_softmax(x) # stays the same shape
        return x

"""
    This function can be used to create random samples for the MNIST dataset. We can take 1000 samples for 
    each label to reduce the dataset size

    # Code for using balanced_sample_maker
    dataset = np.genfromtxt('train.csv', dtype=int, delimiter=',', skip_header=1)
    ytrn = dataset[:, 0] # select prediction column
    Xtrn = dataset[:, 1:] # select all other columns
    Xtrn, ytrn = balanced_sample_maker(Xtrn, ytrn, sample_size=10000)
"""
def balanced_sample_maker(X, y, sample_size, random_seed = 42):
    uniq_levels = np.unique(y)
    uniq_counts = {level: sum(y == level) for level in uniq_levels}

    if not random_seed is None:
        np.random.seed(random_seed)

    # find observation indexes of each class levels
    groupby_levels = {}
    for ii, level in enumerate(uniq_levels):
        obs_idx = [idx for idx, val in enumerate(y) if val == level]
        groupby_levels[level] = obs_idx
    # oversampling on observations of each label
    balanced_copy_idx = []
    for gb_level, gb_idx in groupby_levels.items():
        over_sample_idx = np.random.choice(gb_idx, size=sample_size, replace=True).tolist()
        balanced_copy_idx+=over_sample_idx
    np.random.shuffle(balanced_copy_idx)

    data_train=X[balanced_copy_idx]
    labels_train=y[balanced_copy_idx]
    if  ((len(data_train)) == (sample_size*len(uniq_levels))):
        print('Number of sampled examples: ', sample_size*len(uniq_levels), '\nNumber of sample per class: ', sample_size, ' #classes: ', len(list(set(uniq_levels))))
    else:
        print('Number of samples is wrong.')

    labels, values = zip(*Counter(labels_train).items())
    print('number of classes ', len(list(set(labels_train))))
    check = all(x == values[0] for x in values)
    print(check)
    if check == True:
        print('All classes have the same number of samples')
    else:
        print('Classes are not balanced')
    indexes = np.arange(len(labels))
    width = 0.5

    return data_train,labels_train

	
"""
    # For directly using torch dataset for MNIST
    train_dataset = datasets.MNIST(root='data/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='data/', train=False, transform=transforms.ToTensor(), download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)
"""
if __name__ == "__main__":
    # load data from csv
    print("Loading data from train.csv")
    df = pd.read_csv('train.csv')
    df_small = df
    df_small= df.loc[df['label'] == 0]
	
	# Using dataframes to take only 1000 samples per digit label. We can also use balanced_sample_maker
    for i in range(1, 10):
       df_tmp = df.loc[df['label'] == i]
       df_small = pd.concat([df_small, df_tmp.head(1000)], axis=0)

    y = df_small['label'].values
    X = df_small.drop(['label'],1).values
    Xtrn, Xtst, ytrn, ytst = train_test_split(X, y, test_size=0.3, random_state=42)
    Xtrn_tensor = torch.from_numpy(Xtrn)
    ytrn_tensor = torch.from_numpy(ytrn)
    Xtrn_tensor = Xtrn_tensor.view(-1, 1, 28, 28).type('torch.FloatTensor')
    train_dataset = data_utils.TensorDataset(Xtrn_tensor, ytrn_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16)
    Xtst_tensor = torch.from_numpy(Xtst)
    ytst_tensor = torch.from_numpy(ytst)
    Xtst_tensor = Xtst_tensor.view(-1, 1, 28, 28).type('torch.FloatTensor')
    test_dataset = data_utils.TensorDataset(Xtst_tensor, ytst_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

    model = Net()
    print("Created CNN model is :")
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)

    n_epochs = 5
    model.train()

    for epoch in range(n_epochs):
        train_loss = 0.0
		# input data shape = [N = number of images, 1, 28, 28]
        # becomes [N, 1, 24, 24] after CNN filter with kernel size 5*5
        # becomes [N, 1, 12, 12] after maxpooling with kernel size 2
        # becomes [N, 1, 8, 8] after CNN filter with kernel size 5*5
        # becomes [N, 1, 4, 4] after maxpooling with kernel size 2
        # reshaped to [N, 1024] for fully connected layer (1024 = 4 * 4 * 64, 64 = output channels of prev convolution)
        # becomes [N, 200] after fully connected layer with output 200
        # becomes [N, 10] after fully connected layer with output 10
        # output shape is [N, 10]
        # target labels shape is [N]
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            target = target.long()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
            
        print('Epoch is:', epoch)
        print('Training loss is:', train_loss/len(train_loader.dataset))

    test_err = 0.0
    correct_samples = list(0.0 for i in range(10))
    total_samples = list(0.0 for i in range(10))

    model.eval()

    for data, target in test_loader:
        output = model(data)
        loss = criterion(output, target)
        test_err += loss.item() * data.size(0)
        _, pred = torch.max(output, 1)
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        
        for i in range(len(target)):
            label = target.data[i]
            correct_samples[label] += correct[i].item()
            total_samples[label] += 1

    test_err = test_err/len(test_loader.dataset)
    accuracy = (np.sum(correct_samples)) / np.sum(total_samples)
    print('Testing loss:', test_err)
    print('Overall Testing Accuracy:', accuracy)
