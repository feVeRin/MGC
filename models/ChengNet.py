from smtpd import DebuggingServer
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import torch.nn.functional as F
import torch.optim as optim
import sys
import load_data_single
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

class ChengNet(nn.Module):
    def __init__(self, conv_kernel_size=3):
        super(ChengNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 128, conv_kernel_size, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 64, conv_kernel_size, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 32, conv_kernel_size, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 16, conv_kernel_size, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(16, 8, conv_kernel_size, stride=1, padding=2),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU())
        # mel
        self.fc1 = nn.Linear(416, 23)
        self.act = nn.Softmax(dim=1)

    def forward(self, x):
        dropout = torch.nn.Dropout(p=0.5)
        out = self.layer1(x)
        out = dropout(out)
        out = self.layer2(out)
        out = dropout(out)
        out = self.layer3(out)
        out = dropout(out)
        out = self.layer4(out)
        out = dropout(out)
        out = self.layer5(out)
        out = dropout(out)
        out = out.reshape(out.size(0), -1)
        
        out = torch.flatten(out, 1)
        # out = dropout(out)
        out = self.fc1(out)
        out = self.act(out)
        
        return out
    
    #
    # Device configuration
     
for i in range(10):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    name = 'ChengNet_' + str(i+1)

    if not os.path.isdir("result/"+name):
        os.mkdir("result/"+name)
    file = open("result/"+name +'/' +name+".txt", 'w')
    sys.stout = file


    print("----------------------------------------------------------------", file=file)
    print("Torch Version: ", torch.__version__, file=file)
    print("Device: ", device, file=file)
    print("----------------------------------------------------------------", file=file)
    print("----------------------------------------------------------------")
    print("Torch Version: ", torch.__version__)
    print("Device: ", device)
    print("----------------------------------------------------------------")


    # Hyper parameters
    num_epoch = 60
    # num_epoch = 100
    num_classes = 5
    batch_size = 32
    # learning_rate = 0.0001
    learning_rate = 0.001
    print("BATCH size: ", batch_size, file=file)
    print("EPOCHS: ", num_epoch, file=file)
    print("learning rate: ", learning_rate, file=file)
    print("----------------------------------------------------------------", file=file)
    print("BATCH size: ", batch_size)
    print("EPOCHS: ", num_epoch)
    print("learning rate: ", learning_rate)
    print("----------------------------------------------------------------")
    trans = transforms.Compose([
                transforms.ToTensor()
                ])

    # model = ConvNet(num_classes).to(device)
    model = ChengNet(num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    fulldataset = load_data_single.Dataset()
    ########################################9:1 validation#########################################################
    # trainsize = int(0.9*len(fulldataset))
    trainsize = int(0.8*len(fulldataset))
    testsize = len(fulldataset) - trainsize

    
    print("Train data size: ", trainsize, file=file)
    print("Test data size: ", testsize, file=file)
    print("----------------------------------------------------------------", file=file)
    trainset, testset = torch.utils.data.random_split(fulldataset, [trainsize, testsize])
    # print(len(trainset))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Train the model
    n=0
    epochs = []
    epochs.append(0)
    train_loss_list = []
    train_loss_list.append(0)
    train_acc_list = []
    train_acc_list.append(0)
    test_acc_list = []
    test_acc_list.append(0)
    total_step = len(train_loader)

    test_acc_best = 0
    
    for epoch in range(num_epoch):  
            #training part
            train_loss = 0.0
            train_acc = 0.0
            running_loss = 0.0
            
            for i, data in enumerate(train_loader, 0):
                input, labels = data
                input, labels = input.to(device), labels.to(device)
                outputs = model(input)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 15 == 0:
                    print("epoch: {} {}/{}".format(epoch+1,i,total_step))  
            train_loss = running_loss/len(train_loader.dataset)    
            
            # val part
            correct = 0.0
            total = 0.0
            
            with torch.no_grad():
                for data in train_loader:
                    # input, labels, labels_onehot, labels_unidis = data
                    # input, labels = input.to(device), labels.to(device)
                    input, labels = data
                    input, labels = input.to(device), labels.to(device)
                    outputs = model(input)
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()  
            train_acc = 100 * correct / total
            
            #test part
            correct = 0.0
            total = 0.0
            with torch.no_grad():
                for data in test_loader:
                    # input, labels, labels_onehot, labels_unidis = data
                    # input, labels = input.to(device), labels.to(device)
                    input, labels = data
                    input, labels = input.to(device), labels.to(device)
                    outputs = model(input)
                    
                    #top-1
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == labels).sum().item()  
                    total += labels.size(0)
            
            test_acc = 100 * correct / total
            if(test_acc>test_acc_best):
                # torch.save(model.state_dict(), "result/" + name + '/' +name+"_"+str(epoch+1)+".pt")
                torch.save(model.state_dict(), "result/" + name + '/' +name+".pt")
                test_acc_best=test_acc
                epoch_best = epoch+1

            print("[Epoch: {:>0}]".format(epoch + 1), file=file)
            print("[Epoch: {:>0}]".format(epoch + 1))
            print(" Train loss = {:>.6} Train accuracy = {:>.6}".format(train_loss, train_acc), file=file)
            print(" Train loss = {:>.6} Train accuracy = {:>.6}".format(train_loss, train_acc))
            print(" Test Top-1 accuracy = {:>.6}".format(test_acc), file=file)
            print(" Test Top-1 accuracy = {:>.6}".format(test_acc))


            n += 1
            epochs.append(n)

            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)

    print("----------------------------------------------------------------", file=file)
    print("Finished Training", file=file)
    print("Best Epoch = {:>0}".format(epoch_best), file=file)
    print("Best Accuracy = {:>.6}".format( test_acc_best), file=file)
    print("----------------------------------------------------------------")
    print("Finished Training")
    print(" Best Epoch = {:>0}".format(epoch_best))
    print(" Best Accuracy = {:>.6}".format(test_acc_best))


    # plotting
    plt.title("Loss Curve")
    plt.plot(epochs, train_loss_list, label="Train")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="center right")
    plt.savefig("result/" + name + '/' +name+"loss.png")
    plt.clf()


    plt.title("Accuracy Curve")
    plt.plot(epochs, train_acc_list, label="Train")
    plt.plot(epochs, test_acc_list, label="Test Top-1")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc="center right")
    plt.savefig("result/" + name + '/' +name+"accuracy.png")
    plt.clf()

    file.close()