import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import matplotlib
matplotlib.use('TkAgg')

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_train = transforms.Compose([  # 定义训练集和测试集的图像变换（预处理）操作
    transforms.Resize([224, 224]),  # 调整图像大小为224x224像素
    transforms.ToTensor(),  # 将图像转换为PyTorch张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 对图像进行标准化
])
transform_test = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# 加载训练集和测试集，指定图像变换操作
trainset = datasets.ImageFolder(root=os.path.join('new_COVID_19_Radiography_Dataset', 'train'),  # 训练集根目录
                                transform=transform_train)  # 训练集图像变换
testset = datasets.ImageFolder(root=os.path.join('new_COVID_19_Radiography_Dataset', 'val'),  # 测试集根目录
                               transform=transform_test)  # 测试集图像变换

# 创建训练数据加载器
train_loader = DataLoader(trainset, batch_size=32, num_workers=0,  # 每个批次的样本数为 32 # 使用 0 个额外的子进程来加载数据
                          shuffle=True,
                          pin_memory=True)  # 每个 epoch 开始时，对数据进行洗牌,随机打乱数据集顺序  # 将数据存储在 CUDA 固定内存上，以提高 GPU 加载速度

# 创建测试数据加载器
test_loader = DataLoader(testset, batch_size=32, num_workers=0,
                         shuffle=False, pin_memory=True)


# 定义基础的 CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(  # 定义卷积层和池化层
            # 第一个卷积层：输入通道为3（RGB图像），输出通道为16，卷积核大小为3x3，步长为1，填充为1
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # 换为卷积 更好的特征提取  卷积核为3 步长为1 填充为1
            nn.ReLU(inplace=True),  # 使用ReLU激活函数 # inplace=True 原地进行操作
            # 最大池化层：池化核大小为2x2，步长为2
            nn.MaxPool2d(kernel_size=2, stride=2),  # 将图像大小减半，从224x224变为112x112
            # 第二个卷积层：输入通道为16，输出通道为32，卷积核大小为3x3，步长为1，填充为1
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # 最大池化层：池化核大小为2x2，步长为2
            nn.MaxPool2d(kernel_size=2, stride=2)  # 将图像大小减半，从112x112变为56x56
        )
        # 定义全连接层
        self.classifier = nn.Sequential(
            nn.Linear(32 * 56 * 56, 128),  # 输入大小为32x56x56，输出大小为128   32 是第二个卷积层的输出通道数
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)  # 输入大小为128，输出大小为类别数（默认为4）
        )

    def forward(self, x):
        # 前向传播过程
        x = self.features(x)  # 应用卷积层和池化层
        x = x.view(x.size(0), -1)  # 将特征图展平为一维向量
        x = self.classifier(x)  # 应用全连接层
        return x


# 将模型送到GPU上
num_classes = 4
model = SimpleCNN(num_classes).to(device)


# 定义训练函数
def train(model, train_loader, criterion, optimizer, num_epochs=100):
    """
    训练模型函数
    参数：
        - model: 要训练的模型
        - train_loader: 训练集数据加载器
        - criterion: 损失函数
        - optimizer: 优化器
        - num_epochs: 训练的总epoch数，默认为100
    """

    best_accuracy = 0.0  # 保存最佳准确率
    train_losses = []  # 记录每个epoch的训练损失
    test_accuracies = []  # 记录每个epoch的测试准确率

    for epoch in range(num_epochs):  # 遍历每个 epoch
        model.train()  # 将模型设置为训练模式
        running_loss = 0.0  # 初始化一个变量用于累计每个 epoch 的损失
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}',
                  unit='batch') as pbar:  # 使用 tqdm 库创建一个进度条，显示当前 epoch 的进度
            for inputs, labels in train_loader:  # 遍历每个 batch 的数据
                inputs, labels = inputs.to(device), labels.to(device)  # 将数据送到GPU上
                optimizer.zero_grad()  # 梯度清零，以便后续计算新的梯度
                outputs = model(inputs)  # 将数据输入模型，获取模型的预测输出
                loss = criterion(outputs, labels)  # 计算模型的损失
                loss.backward()  # 反向传播，计算梯度
                optimizer.step()  # 更新模型参数
                running_loss += loss.item() * inputs.size(0)  # 更新当前 epoch 的损失
                pbar.update(1)  # 更新进度条，显示当前 batch 的损失
                pbar.set_postfix({'loss': loss.item()})

        epoch_loss = running_loss / len(train_loader.dataset)  # 计算当前 epoch 的平均损失
        train_losses.append(epoch_loss)  # 记录训练损失
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")  # 打印当前 epoch 的损失

        # 在每个 epoch 结束后评估模型并保存最佳权重
        metrics = evaluate(model, test_loader, criterion, device)
        current_accuracy = metrics['accuracy']  # 从字典中提取准确率

        test_accuracies.append(current_accuracy)  # 记录测试准确率

        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            # 保存最佳模型
            save_model(model, 'best_model.pth')
            print("Model saved with best accuracy:", best_accuracy)

    # 绘制损失和准确率曲线
    plot_metrics(train_losses, test_accuracies)


# 评估模型在测试集上的性能
def evaluate(model, test_loader, criterion, device):
    model.eval()  # 将模型设置为评估模式
    test_loss = 0.0  # 初始化测试损失
    correct = 0  # 初始化预测正确的样本数
    total = 0  # 初始化总样本数
    all_preds = []  # 存储所有预测的标签
    all_labels = []  # 存储所有真实标签
    with torch.no_grad():  # 在评估阶段不需要计算梯度
        for inputs, labels in test_loader:  # 遍历测试集中的每个 batch
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据送到GPU上
            outputs = model(inputs)  # 将数据输入模型，获取模型的预测输出
            loss = criterion(outputs, labels)  # 计算模型的损失
            test_loss += loss.item() * inputs.size(0)  # 累计测试损失
            _, predicted = torch.max(outputs, 1)  # 获取模型预测的类别
            total += labels.size(0)  # 更新总样本数
            correct += (predicted == labels).sum().item()  # 更新预测正确的样本数
            # 收集所有的预测和真实标签，用于计算精确度、召回率、F1分数
            all_preds.extend(predicted.cpu().numpy())  # 添加预测标签
            all_labels.extend(labels.cpu().numpy())  # 添加真实标签

    avg_loss = test_loss / len(test_loader.dataset)  # 计算平均测试损失
    accuracy = 100.0 * correct / total  # 计算准确率
    # 计算其他指标
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")  # 输出测试损失和准确率
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}  # 返回准确率值

# 绘制损失和准确率曲线
def plot_metrics(train_losses, test_accuracies):
    epochs = range(1, len(train_losses) + 1)

    # 绘制训练损失曲线
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid()

    # 绘制测试准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_accuracies, label='Test Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy Curve')
    plt.grid()
    plt.tight_layout()
    plt.show()


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)
    # 定义保存模型权重的文件夹路径
    model_dir = 'model_pth'


if __name__ == '__main__':
    num_epochs = 5  # 设置训练相关的参数
    learning_rate = 0.001  # 可调 0.0001 0.01
    num_classes = 4
    data_dir = "new_COVID_19_Radiography_Dataset"
    save_path = r"model_pth\bets.pth"


    model = SimpleCNN(num_classes).to(device)  # 初始化模型，将模型移至GPU上
    criterion = nn.CrossEntropyLoss()  # 初始化损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 初始化优化器
    train(model, train_loader, criterion, optimizer, num_epochs=num_epochs)  # 使用训练数据集训练模型
    evaluate(model, test_loader, criterion, device)  # 使用测试数据集评估模型性能
