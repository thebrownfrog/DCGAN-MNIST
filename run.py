import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import time
from torchvision.utils import save_image


epochs = 128
latent_size = (64,10)
batch_size = 250
datasets_path = "./data"

train_transform = transforms.Compose([#didn't do any data augmentation, it propably wouldn't hurt to do it
    #transforms.RandomRotation(10),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root=datasets_path, train=True, download=True, transform=train_transform)
train_data_length = len(train_dataset)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root=datasets_path, train=False, download=True, transform=test_transform)
test_data_length = len(test_dataset)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

def noise(size):
    return torch.randn(size, latent_size[0], device="cuda")*13.5, torch.randint(0,latent_size[1],(size,),device="cuda").long()

class GAN:
    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.init_size = 7  # Initial size before upsampling
            self.label_embedding = nn.Embedding(latent_size[1], latent_size[0])
            
            self.l1 = nn.Sequential(
                nn.Linear(latent_size[0] * 2, 64 * self.init_size * self.init_size),
                nn.BatchNorm1d(64 * self.init_size * self.init_size),
                nn.ReLU(inplace=True)
            )
            
            self.conv_blocks = nn.Sequential(
                nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),
                nn.Tanh()
            )

        def forward(self, z):
            latent_vec, labels = z
            label_embedding = self.label_embedding(labels)
            gen_input = torch.cat((latent_vec, label_embedding), dim=1)
            out = self.l1(gen_input)
            out = out.view(out.size(0), 64, self.init_size, self.init_size)
            img = self.conv_blocks(out)
            return img
    class Discriminator(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)

            LReLU_slope = .01
            dro = .125

            self.conv1 = nn.Sequential(
                nn.Conv2d(1,24,kernel_size=1,stride=1,padding=0),
                nn.Dropout2d(p=dro),
                nn.BatchNorm2d(24),
                nn.LeakyReLU(negative_slope=LReLU_slope))
            self.conv2 = nn.Sequential(
                nn.Conv2d(24,32,kernel_size=3,stride=1,padding=1),
                nn.Dropout2d(p=dro),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(negative_slope=LReLU_slope),
                nn.MaxPool2d(kernel_size=2,stride=2))
            self.conv3 = nn.Sequential(
                nn.Conv2d(32,48,kernel_size=3,stride=1,padding=1),
                nn.Dropout2d(p=dro),
                nn.BatchNorm2d(48),
                nn.LeakyReLU(negative_slope=LReLU_slope),
                nn.MaxPool2d(kernel_size=2,stride=2))
            self.conv4 = nn.Sequential(
                nn.Conv2d(48,128,kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(negative_slope=LReLU_slope),
                nn.AvgPool2d(kernel_size=7),
                nn.Flatten())
            
            
            self.classifier1 = nn.Sequential(
                nn.Linear(128,1),
                nn.Sigmoid())
            self.classifier2 = nn.Sequential(
                nn.Linear(128,latent_size[1]))
            
        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)

            validity = self.classifier1(x)
            label = self.classifier2(x)
            
            return validity, label
    def __init__(self, lr_g, lr_d) -> None:
        self.generator = GAN.Generator().cuda()
        self.discriminator = GAN.Discriminator().cuda()
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr_g)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr_d)
        self.validity_loss = nn.BCELoss()
        self.class_loss = nn.BCEWithLogitsLoss()
        self.ones1 = torch.ones(batch_size,1,device="cuda")
        self.ones2 = torch.ones(batch_size*2,1,device="cuda")
        self.zeros = torch.zeros(batch_size,1,device="cuda")

    def process_training_batch(self, images, labels, train):
        mini_batch_size = images.shape[0]
        labels = F.one_hot(labels, latent_size[1]).float().cuda()
        real_images = images.cuda()

        if train:
            self.discriminator.train()
            self.generator.train()
        else:
            self.discriminator.eval()
            self.generator.eval()

        if train:
            self.discriminator_optimizer.zero_grad()

        nose = noise(mini_batch_size)
        fake_images = self.generator(nose)
        prediction_on_fakes = self.discriminator(fake_images)
        prediction_on_reals = self.discriminator(real_images)

        fake_loss = 0.6*self.validity_loss(prediction_on_fakes[0], self.zeros[:mini_batch_size])+self.class_loss(prediction_on_fakes[1], F.one_hot(nose[1],num_classes=latent_size[1]).float())*.4
        real_loss = 0.5*self.validity_loss(prediction_on_reals[0], self.ones1[:mini_batch_size])+self.class_loss(prediction_on_reals[1], labels)*.5
        
        discriminator_loss = .5*(fake_loss + real_loss)

        if train:
            discriminator_loss.backward()
            self.discriminator_optimizer.step()

        if train:
            self.generator.zero_grad()

        nose = noise(mini_batch_size*2)
        fake_images = self.generator(nose)
        prediction_on_fakes = self.discriminator(fake_images)
        generator_loss = 0.5*self.validity_loss(prediction_on_fakes[0], self.ones2[:mini_batch_size*2]) + self.class_loss(prediction_on_fakes[1], F.one_hot(nose[1],num_classes=latent_size[1]).float())*.5
        
        if train:
            generator_loss.backward()
            self.generator_optimizer.step()

        return discriminator_loss.item(), generator_loss.item()

gan = GAN(lr_d=0.0001,lr_g=0.0002)
showcase_noise = noise(64)[0], torch.floor(torch.arange(0,64,step=1,dtype=float,device="cuda")/64*10).long()#propably not the intended way to do this, basically I want to have digits in increasing order(for the showcase gif)
save_image(gan.generator(showcase_noise),"showcase_epoch-1.png")
total_time = 0.0

print(f"generator parameters: {sum(p.numel() for p in gan.generator.parameters())}")
print(f"discriminator parameters: {sum(p.numel() for p in gan.discriminator.parameters())}")

def process_data(loader, train, datalen):
    start = time.time()
    d_loss = 0.0
    g_loss = 0.0
    for images, labels in loader:
        mini_batch_size = images.shape[0]
        batch_d_loss, batch_g_loss = gan.process_training_batch(images,labels,train)
        d_loss += batch_d_loss * mini_batch_size
        g_loss += batch_g_loss * mini_batch_size
    d_loss /= datalen
    g_loss /= datalen
    return time.time()-start, d_loss, g_loss

print("started training")

for epoch in range(epochs):
    train_epoch_time, train_d_loss, train_g_loss = process_data(train_loader, True, train_data_length)
    test_epoch_time, test_d_loss, test_g_loss = process_data(test_loader, False, test_data_length)
    total_time += train_epoch_time
    total_time += test_epoch_time
    print(f"epoch {epoch} ended(in {train_epoch_time+test_epoch_time:.1f}s, total={total_time:.1f}s). train,test d_loss={train_d_loss:.4f},{test_d_loss:.4f}. train,test g_loss={train_g_loss:.4f},{test_g_loss:.4f}")
    save_image(gan.generator(showcase_noise),f"showcase_epoch{epoch}.png")
