#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


w = np.load('D:\交大電信\深度學習\TibetanMNIST.npz')
img = w['image']
img = np.where(img>128,1,0) #轉成binary image
img = img.reshape([len(img),1,28,28])
tensor_img = torch.tensor(img)
img_dataloader = DataLoader(tensor_img,batch_size=128,shuffle=True)


# In[3]:


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


# In[4]:


class VAE_b(nn.Module):
    def __init__(self):
        super(VAE_b, self).__init__()
        
        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc21 = nn.Linear(500, 20)
        self.fc22 = nn.Linear(500, 20) 
        self.fc3 = nn.Linear(20, 500)
        self.fc4 = nn.Linear(500, 500)
        self.fc5 = nn.Linear(500, 784)
        self.dropout = nn.Dropout(0.1)

    def encode(self, x):
        h1 = self.dropout(F.relu(self.fc1(x)))
        h2 = self.dropout(F.relu(self.fc2(h1)))
        return self.fc21(h2), F.softplus(self.fc22(h2)) 

    def reparameterize(self, mu, std):
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = self.dropout(F.relu(self.fc3(z)))
        h4 = self.dropout(F.relu(self.fc4(h3)))
        return torch.sigmoid( self.fc5(h4) ) 

    def forward(self, x):
        mu, std = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, std)
        return self.decode(z), mu, 2.*torch.log(std) 
    
def loss_vae_b(recon_x, x, mu, logvar,scale=1):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + scale*KLD


# In[6]:


def train_b(model, optimizer, epochs, 
          train_loader, loss_function,scale=1):
    elbo = []
    model.train()
    for epoch in range(1, epochs + 1):
        train_loss = 0
        for batch_idx, data  in enumerate(train_loader):
            data = data.to(device)
            data = data.float()
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar,scale)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        elbo.append(train_loss)
    return elbo


# In[7]:


def plot_recon(img, title):
    npimg = img.numpy()
    plt.figure(figsize = (10, 10))
    plt.title(title)
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation = 'nearest')


# ### 1. Implement VAE and show the learning curve and some reconstructed samples like thegiven examples.

# In[8]:


model_b = VAE_b().to(device)
optimizer_b = torch.optim.Adam(model_b.parameters(), lr=1e-3)
elbo_b = train_b(model_b, optimizer_b, 100,img_dataloader, loss_vae_b)


# In[9]:


plt.plot(list(range(100)),elbo_b)
plt.xlabel('Epoches')
plt.ylabel('Loss')
plt.title('Learning Curve of the Tibetan')
plt.show()


# In[10]:


fixed_x_b = next(iter(img_dataloader))
recon_x_b,_b,_c = model_b(fixed_x_b.float())
recon_x_b = recon_x_b.cpu().view(128, 1, 28, 28).detach()


# In[11]:


plot_recon(torchvision.utils.make_grid(recon_x_b[:128],8 ),'Tibetan')


# ### 2. Sample the prior p(z) and use the latent codes z to synthesize some examples when your model is well-trained

# In[12]:


sample = torch.randn(128, 20).to(device)
sample_vae_b = model_b.decode(sample).cpu().view(128, 1, 28, 28).detach()


# In[13]:


plot_recon(torchvision.utils.make_grid(sample_vae_b[:128],8 ),'Tibetan')


# ### 3. Show the synthesized images based on the interpolation of two latent codes z between two real samples.

# In[15]:


def interpolation_ti(model,z1,z2,divide=7):
    z3 = (z2-z1)/divide
    img1 = []
    for j in range(1,divide+1):
        z4 = z1+j*z3
        out = model.decode(z4).cpu().view(1, 1, 28, 28).detach()
        img1.append(out)
    img1 = torch.tensor([item.detach().numpy() for item in img1]).squeeze(-4)
    return img1


# In[14]:


x1_b = tensor_img[50].view(-1,784)
mu1_b,var1_b = model_b.encode(x1_b.float())
z1_b = model_b.reparameterize(mu1_b,var1_b)
x2_b = tensor_img[70].view(-1,784)
mu2_b,var2_b = model_b.encode(x2_b.float())
z2_b = model_b.reparameterize(mu2_b,var2_b)


# In[16]:


im = interpolation_ti(model_b,z1_b,z2_b,11)


# In[17]:


plot_recon(torchvision.utils.make_grid(im[:],11 ),'Tibetan')


# ### Multiply the Kullback-Leiblier (KL) term with a scale λ and tune λ (e.g. λ = 0 andλ = 100) then show the results based on steps 1, 2, 3 and some analyses.

# In[18]:


model_0_b = VAE_b().to(device)
optimizer_0_b = torch.optim.Adam(model_0_b.parameters(), lr=0.001)
elbo_0_b = train_b(model_0_b, optimizer_0_b, 100,img_dataloader, loss_vae_b,0)


# In[19]:


model_100_b = VAE_b().to(device)
optimizer_100_b = torch.optim.Adam(model_100_b.parameters(), lr=0.001)
elbo_100_b = train_b(model_100_b, optimizer_100_b, 100,img_dataloader, loss_vae_b,100)


# In[20]:


plt.figure(1)
plt.plot(list(range(100)),elbo_0_b)
plt.xlabel('Epoches')
plt.ylabel('Loss')
plt.title('Learning Curve of the Tibetan,scale = 0')
plt.figure(2)
plt.plot(list(range(100)),elbo_100_b)
plt.xlabel('Epoches')
plt.ylabel('Loss')
plt.title('Learning Curve of the Tibetan,scale = 100')
plt.show()


# In[21]:


fixed_x_b_0 = next(iter(img_dataloader))
recon_x_b_0,_b,_c = model_0_b(fixed_x_b_0.float())
recon_x_b_0 = recon_x_b_0.cpu().view(128, 1, 28, 28).detach()
fixed_x_b_100 = next(iter(img_dataloader))
recon_x_b_100,_b,_c = model_100_b(fixed_x_b_100.float())
recon_x_b_100 = recon_x_b_100.cpu().view(128, 1, 28, 28).detach()


# In[22]:


plot_recon(torchvision.utils.make_grid(recon_x_b_0[:128],8 ),' Reconstructed Tibetan,scale = 0')
plot_recon(torchvision.utils.make_grid(recon_x_b_100[:128],8 ),' Reconstructed Tibetan,scale = 100')


# In[23]:


sample_0 = torch.randn(128, 20).to(device)
sample_vae_b_0 = model_0_b.decode(sample_0).cpu().view(128, 1, 28, 28).detach()
sample_100 = torch.randn(128, 20).to(device)
sample_vae_b_100 = model_100_b.decode(sample_100).cpu().view(128, 1, 28, 28).detach()


# In[24]:


plot_recon(torchvision.utils.make_grid(sample_vae_b_0[:128],8 ),' Using the latent codes z to reconstruct , scale = 0')
plot_recon(torchvision.utils.make_grid(sample_vae_b_100[:128],8 ),' Using the latent codes z to reconstruct , scale = 100')


# In[25]:


x1_b_0 = tensor_img[50].view(-1,784)
mu1_b_0,var1_b_0 = model_0_b.encode(x1_b_0.float())
z1_b_0 = model_0_b.reparameterize(mu1_b_0,var1_b_0)
x2_b_0 = tensor_img[70].view(-1,784)
mu2_b_0,var2_b_0 = model_0_b.encode(x2_b_0.float())
z2_b_0 = model_0_b.reparameterize(mu2_b_0,var2_b_0)
x1_b_100 = tensor_img[50].view(-1,784)
mu1_b_100,var1_b_100 = model_100_b.encode(x1_b_100.float())
z1_b_100 = model_100_b.reparameterize(mu1_b_100,var1_b_100)
x2_b_100 = tensor_img[70].view(-1,784)
mu2_b_100,var2_b_100 = model_100_b.encode(x2_b_100.float())
z2_b_100 = model_100_b.reparameterize(mu2_b_100,var2_b_100)


# In[28]:


im_0 = interpolation_ti(model_0_b,z1_b_0,z2_b_0,11)
im_100 = interpolation_ti(model_100_b,z1_b_100,z2_b_100,11)


# In[29]:


plot_recon(torchvision.utils.make_grid(im_0[:],11 ),'Tibetan interpolation,scale = 0')
plot_recon(torchvision.utils.make_grid(im_100[:],11 ),'Tibetan interpolation,scale = 100')


# In[ ]:




