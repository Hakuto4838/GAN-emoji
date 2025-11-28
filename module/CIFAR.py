import torch
import torchvision
import torchsummary 
import torch.nn as nn
from module.basic import trans_totrain, trans_toeval
import matplotlib.pyplot as plt

def cheak_GPU():
    
    if(torch.cuda.is_available()):
        print(torch.cuda.device_count())
        print(torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print('cuda is not available')

class cifarMod():
    def __init__(self, input_size, load = None):
        cheak_GPU()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.insize = input_size
        self.module = torchvision.models.vgg19_bn(num_classes=10).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.module.parameters(), lr=0.001, momentum=0.9)

        if load is not None:
            self.module.load_state_dict(torch.load(load, weights_only=True))
            self.module.eval()
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        pass


    def summary(self):
        torchsummary.summary(self.module, self.insize)

    def preparedata(self):
        batch_size = 16

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=trans_totrain)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=trans_totrain)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

        self.reportPerEpoch = len(self.trainloader)/200



    def train(self, epochs):
        self.preparedata()
        self.begin_plt()
        for epoch in range(epochs):
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            self.module.train()
            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.module(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                _, predicted = torch.max(outputs, 1)
                total_train += inputs.size(0)
                correct_train += (predicted == labels).sum().item()

                running_loss += loss.item()
                if i%200 == 199:
                    self.train_acc.append(correct_train/total_train)
                    self.train_loss.append(running_loss/200)
                    print(f'[{epoch+1}, {i+1:5d}] loss: {running_loss/200:.3f}, acc : {correct_train/total_train:.3f}')
                    running_loss=0.0
                    correct_train = 0
                    total_train = 0
                    self.update_plt(epoch+1)
            
            #評估
            self.module.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for data in self.testloader:
                    inputs, labels = data[0].to(self.device), data[1].to(self.device)
                    outputs = self.module(inputs)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()
            
            self.val_loss.append(val_loss/ len(self.testloader))
            self.val_acc.append(correct_val/total_val)
            self.update_plt(epoch+1)
        self.end_plt()


        print('finish')
        plt.savefig('./train/tarinlog.png')
        PATH = './train/cifar_net.pth'
        torch.save(self.module.state_dict(), PATH)

    def begin_plt(self):
        # 初始化動態繪圖
        plt.ion()  # 啟用交互模式
        self.fig = plt.figure(figsize=(10, 5))
        self.ax_loss = self.fig.add_subplot(1, 2, 1)
        self.ax_acc = self.fig.add_subplot(1, 2, 2)

    def update_plt(self, epoch):

        # 繪製損失
        self.ax_loss.clear()
        self.ax_loss.plot([i*epoch/len(self.train_loss) for i in range(len(self.train_loss))], self.train_loss, label='Training Loss', color='blue')
        self.ax_loss.plot(range(1,1+len(self.val_loss)), self.val_loss, label='Validation Loss', color='orange')
        self.ax_loss.set_title("Loss")
        self.ax_loss.set_xlabel("Epochs")
        self.ax_loss.set_ylabel("Loss")
        self.ax_loss.legend()

        # 繪製準確率
        self.ax_acc.clear()
        self.ax_acc.plot([i*epoch/len(self.train_acc) for i in range(len(self.train_acc))], self.train_acc, label='Training Accuracy', color='green')
        self.ax_acc.plot(range(1,1+len(self.val_loss)), self.val_acc, label='Validation Accuracy', color='red')
        self.ax_acc.set_title("Accuracy")
        self.ax_acc.set_xlabel("Epochs") 
        self.ax_acc.set_ylabel("Accuracy")
        self.ax_acc.legend()

        # 暫停更新
        plt.pause(0.01)
    
    def end_plt(self):
        plt.ioff()  # 關閉交互模式
        plt.show()

    def __call__(self, images):
        images = [trans_toeval(image).to(self.device) for image in images] 
        outputs = self.module(torch.stack(images))
        return torch.softmax(outputs, dim=1), [self.classes[torch.argmax(outputs)]]