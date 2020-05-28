import numpy as np
import torch
from torch import nn,optim
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
import torchvision
from torchvision import datasets,transforms

from art.attacks import DeepFool
from art.classifiers import PyTorchClassifier


# Step 0: Define the neural network model, return logits instead of activation in forward method
from torchvision.models import resnet18

torch.manual_seed(1234)
lr=1e-3


def value_key(value):  # value -> key
    dict = db.class_to_idx
    key = list(dict.keys())[list(dict.values()).index(value)]
    return key

def plot_image(img, label_onehot, name):
    from matplotlib import pyplot as plt
    fig=plt.figure()
    img = img.transpose(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
    label = np.argmax(label_onehot,axis=1)
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(img[i], cmap='gray', interpolation='none')
        plt.title('{}: {}'.format(name,value_key(label[i])))
        plt.xticks([])
        plt.yticks([])
    plt.show()


# Step 1: Load the Pokemon dataset
db=torchvision.datasets.ImageFolder(root='pokemon',
                                    transform=transforms.Compose([
                                        transforms.Resize((64, 64)),
                                        transforms.ToTensor()
                                    ]))
# print(len(db))  # len = 1167
print(db.class_to_idx)
# train: 1000   test: 167
train_db,test_db=torch.utils.data.random_split(db,[1000, 167])
print('train_db:',len(train_db),'test_db:',len(test_db))

# Step 1a: datasets from: torch.datasets.pokemon
train_loader=torch.utils.data.DataLoader(train_db,batch_size=1000,shuffle=True)
test_loader=torch.utils.data.DataLoader(test_db,batch_size=167,shuffle=True)

# Step 1b: Switch to numpy, one_hot
def torch_load_numpy(train_loader,test_loader,N):
    # N=5  # class_nums=5  {'bulbasaur': 0, 'charmander': 1, 'mewtwo': 2, 'pikachu': 3, 'squirtle': 4}
    x_train,y_train=next(iter(train_loader))
    x_train,y_train=x_train.numpy(),y_train.numpy()
    # y_train -> one_hot
    y_train=np.eye(N)[y_train]
    print(x_train.shape,y_train.shape,x_train.min(),x_train.max())

    x_test,y_test=next(iter(test_loader))
    x_test,y_test=x_test.numpy(),y_test.numpy()
    # y_test -> one_hot
    y_test=np.eye(N)[y_test]
    print(x_test.shape,y_test.shape,x_test.min(),x_test.max())

    min_pixel_value = x_train.min()
    max_pixel_value = x_train.max()

    return (x_train,y_train),(x_test,y_test),min_pixel_value, max_pixel_value

(x_train,y_train),(x_test,y_test),min_pixel_value, max_pixel_value = torch_load_numpy(train_loader,test_loader,5)
# plot_image(x_train,y_train,'train')

# Step 2: Create the model
class Flatten(nn.Module):

    def __init__(self):
        super(Flatten,self).__init__()

    def forward(self,x):
        shape=torch.prod(torch.tensor(x.shape[1:])).item()   # torch.prod 返回所有元素的乘积
        return x.view(-1,shape)

trained_model=resnet18(pretrained=True)
model=nn.Sequential(*list(trained_model.children())[:-1],  # [b,512,1,1]
                    Flatten(),  # [b,512,1,1] => [b,512]
                    nn.Linear(512,5)
                    )


# Step 2a: Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


# Step 3: Create the ART classifier
classifier = PyTorchClassifier(
    model=model,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(3, 64, 64),
    nb_classes=5,
)


# Step 4: Train the ART classifier
classifier.fit(x_train, y_train, batch_size=32, nb_epochs=5)


# Step 5: Evaluate the ART classifier on benign test examples
predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))
plot_image(x_test,predictions,'test')


# Step 6: Generate adversarial test examples
attack = DeepFool(classifier=classifier, max_iter=100, batch_size=32)
x_test_adv = attack.generate(x=x_test)
print('x_test_adv.shape=',x_test_adv.shape)


# Step 7: Evaluate the ART classifier on adversarial test examples
predictions = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
plot_image(x_test_adv,predictions,'adv-test')