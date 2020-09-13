import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from lenet import Lenet

transform = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]),
}

trainset = torchvision.datasets.MNIST(root='/home/linghu8812/data/mnist', train=True,
                                      download=True, transform=transform['train'])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=8)

testset = torchvision.datasets.MNIST(root='/home/linghu8812/data/mnist', train=False,
                                     download=True, transform=transform['test'])
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=8)

net = Lenet()
print(net)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=3e-4, weight_decay=5e-4)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

net.to(device)
precision = 0.0
preprecision = 0.0
for epoch in range(100):  # loop over the dataset multiple times
    running_loss = 0.0
    avg_loss = 0.0
    k = 0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.4f' %
                  (epoch + 1, i + 1, running_loss / 200))
            avg_loss += running_loss
            k += 200
            running_loss = 0.0

    avg_loss /= k
    print('Average Loss of the network on the %d epoch: %.4f' % (epoch + 1, avg_loss))
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    precision = 100 * correct / total
    print('Accuracy of the network on the %d epoch: %.2f %%' % (epoch + 1, precision))
    if precision > preprecision:
        print('Better Accuracy: %.4f %%, Saving Net to mnist_net.pt.' % precision)
        torch.save(net, 'mnist_net.pt')
        preprecision = precision
    exp_lr_scheduler.step()

print('Finished Training')
