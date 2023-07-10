import sys
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import matplotlib.pyplot as plt
import densenet_pro
import os
import numpy as np
import load_data_multi
import load_data_single

from torchsummary import summary
import pytorch_model_summary
import torchinfo
# from prettytable import PrettyTable

#모델 선택
# model, name = models.googlenet(), "GoogleNet"
# model, name = models.densenet(), "DenseNet121"
# model, name = models.densenet161(), "DenseNet161"
#model, name = models.densenet169(), "DenseNet169"
#model, name = models.densenet201(), "DenseNet201"
#model, name = models.squeezenet1_0(), "SqueezeNet"

#model, name = models.vgg11(), "VGGNet11"
# model, name = models.vgg16(), "VGGNet16"
#model, name = models.vgg19(), "VGGNet19"
#model, name = models.shufflenet_v2_x1_0(), "shufflenetv2"
#model, name = models.densenet121(), "DenseNet"

#model, name = models.resnet18(), "ResNet18__p2"
# model, name = models.resnet18(), "ResNet18_t1"
#model, name = models.resnet50(), "ResNet50"
for i in range(10):
    model, name = densenet_pro.densenet121(), 'proposed_den_' + str(i+1)
    # model, name = models.resnext101_32x8d(), 'resnext101_' + str(i+1)
    # model, name = models.resnet101(), "ResNet101"
    # model, name = models.resnet152(), "ResNet152"
    #model, name = models.alexnet(), "AlexNet"

    if not os.path.isdir("result/"+name):
        os.mkdir("result/"+name)
    file = open("result/"+name +'/' +name+".txt", 'w')
    sys.stout = file

    #Batch_size, epochs 설정
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("----------------------------------------------------------------", file=file)
    print("Torch Version: ", torch.__version__, file=file)
    print("Device: ", device, file=file)
    print("----------------------------------------------------------------", file=file)
    print("----------------------------------------------------------------")
    print("Torch Version: ", torch.__version__)
    print("Device: ", device)
    print("----------------------------------------------------------------")
    batch_size = 16
    num_epoch = 60
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

    # fulldataset = load_data_multi.Dataset()
    fulldataset = load_data_single.Dataset()

    trainsize = int(0.8*len(fulldataset))
    testsize = len(fulldataset) - trainsize

    print("Train data size: ", trainsize, file=file)
    print("Test data size: ", testsize, file=file)
    print("----------------------------------------------------------------", file=file)
    trainset, testset = torch.utils.data.random_split(fulldataset, [trainsize, testsize])
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    #모델 설정
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)    

    # print(count_parameters(model))
    # print(numel(model))
    print(summary(model, (3, 256, 1296), device=device.type)) #torchsummary
    # print(pytorch_model_summary.summary(model, torch.zeros(3,3,256,1296), show_input=True))
    # print(torchinfo.summary(model, input_size=(1,3,256,1296), device=device.type))
    print(fasdfasdfas)

    # plot part
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
            # stft, mel, mfcc, labels = data
            # stft, mel, mfcc, labels = stft.to(device), mel.to(device), mfcc.to(device), labels.to(device)
            
            # outputs = model(stft, mel, mfcc)

            # input1, input2, labels = data
            # input1, input2, labels = input1.to(device), input2.to(device), labels.to(device)            
            # outputs = model(input1, input2)

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
                # stft, mel, mfcc, labels = data
                # stft, mel, mfcc, labels = stft.to(device), mel.to(device), mfcc.to(device), labels.to(device)
                # outputs = model(stft, mel, mfcc)

                # input1, input2, labels = data
                # input1, input2, labels = input1.to(device), input2.to(device), labels.to(device)            
                # outputs = model(input1, input2)

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
                # stft, mel, mfcc, labels = data
                # stft, mel, mfcc, labels = stft.to(device), mel.to(device), mfcc.to(device), labels.to(device) 
                # outputs = model(stft, mel, mfcc)

                # input1, input2, labels = data
                # input1, input2, labels = input1.to(device), input2.to(device), labels.to(device)            
                # outputs = model(input1, input2)

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
