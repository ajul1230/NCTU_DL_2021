#!/usr/bin/env python
# coding: utf-8

# ## Variational Autoencoder for Image Generation

# In[27]:


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


# In[28]:


transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
])
data = datasets.ImageFolder('D:\交大電信\深度學習\data_face',transform = transform)
dataloader = DataLoader(data, batch_size = 128, shuffle = True)


# In[29]:


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


# In[30]:


class VAE(nn.Module):
    def __init__(self,z_dim=128):
        super(VAE,self).__init__()
        self.fc1 = nn.Linear(256,z_dim)
        self.fc2 = nn.Linear(256,z_dim)
        self.fc3 = nn.Linear(z_dim,256)
        self.encoder = nn.Sequential(
            
            nn.Conv2d(3,16,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            
            
            nn.Conv2d(16,32,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            
            nn.Conv2d(32,64,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            
            nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        
            nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
    )
        self.decoder = nn.Sequential(
        
            nn.ConvTranspose2d(256,128,kernel_size=3,stride=2,padding=1,output_padding = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        
            nn.ConvTranspose2d(128,64,kernel_size=3,stride=2,padding=1,output_padding = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            
            nn.ConvTranspose2d(64,32,kernel_size=3,stride=2,padding=1,output_padding = 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            
            nn.ConvTranspose2d(32,16,kernel_size=3,stride=2,padding=1,output_padding = 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            
            nn.ConvTranspose2d(16,16,3,2,1,1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16,3,kernel_size=3,padding=1),
            nn.BatchNorm2d(3),
            nn.Tanh()
    )

    def encode(self,x):
        h = self.encoder(x)
        h = torch.flatten(h,start_dim=1)
        return self.fc1(h),self.fc2(h)
    
    def reparameterize(self,mu,var):
        std = torch.exp(var/2)
        epslion = torch.rand_like(std)
        result = mu+std*epslion
        return result
    
    def decode(self,z):
        r = self.fc3(z)
        r = r.view(-1,256,1,1)
        r = self.decoder(r)
        return r
    
    def forward(self,x):
        mu,var = self.encode(x)
        z = self.reparameterize(mu,var)
        res = self.decode(z)
        return res,mu,var


# In[31]:


def train(model,dataloader,opt,epoch,scale=1):
    ELBO = []
    for e in range(epoch):
        total_loss = 0
        for i,(image,j) in enumerate(dataloader):
            image = image.to(device)
            recon_images, mu, log_var = model(image)
            l = loss(recon_images, image, mu, log_var,scale)
            total_loss += l
            opt.zero_grad()
            l.backward()
            opt.step()
        ELBO.append(total_loss)
    return ELBO


# In[32]:


def loss(recon_x, x, mu, logvar,scale=1):
    reconst_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_div = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconst_loss + scale*kl_div


# In[33]:


def plot_recon(img, title):
    npimg = img.numpy()
    plt.figure(figsize = (10, 10))
    plt.title(title)
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation = 'nearest')


# ### 1. Implement VAE and show the learning curve and some reconstructed samples like thegiven examples.

# In[34]:


model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
elbo = train(model,dataloader,optimizer,100)
torch.save(model.state_dict(), '100epoches.pt')


# In[35]:


plt.figure(1)
plt.plot(list(range(100)),elbo)
plt.xlabel('Epoches')
plt.ylabel('Loss')
plt.title('Average Learning Curve of the animation face')


# In[36]:


fixed_x = next(iter(dataloader))[0]
recon_x, _, _ = model(fixed_x)


# In[37]:


plot_recon(torchvision.utils.make_grid(recon_x.data[:128],8 ),'Animation faces')


# ### 2. Sample the prior p(z) and use the latent codes z to synthesize some examples when your model is well-trained

# In[38]:


z = torch.randn(128, 128)
out1 = model.decode(z)


# In[39]:


plot_recon(torchvision.utils.make_grid(out1.data[:128],8 ),'Animation faces')


# ### 3. Show the synthesized images based on the interpolation of two latent codes z between two real samples.

# In[40]:


model.eval()


# In[41]:


x1 = data[50][0].unsqueeze(0)
mu1,var1 = model.encode(x1)
z1 = model.reparameterize(mu1, var1)
x2 = data[70][0].unsqueeze(0)
mu2,var2 = model.encode(x2)
z2 = model.reparameterize(mu2,var2)


# In[42]:


def interpolation_af(model,z1,z2,divide=7):
    z3 = (z2-z1)/divide
    img1 = []
    for i in range(1,divide+1):
        z4 = z1+i*z3
        out = model.decode(z4)
        img1.append(out.data)
    img1 = torch.tensor([item.detach().numpy() for item in img1]).squeeze(-4)
    return img1


# In[43]:


im = interpolation_af(model,z1,z2,11)


# In[44]:


plot_recon(torchvision.utils.make_grid(im[:],11 ),'Animation faces')


# ### Multiply the Kullback-Leiblier (KL) term with a scale λ and tune λ (e.g. λ = 0 andλ = 100) then show the results based on steps 1, 2, 3 and some analyses.

# In[45]:


model_0 = VAE().to(device)
optimizer_0 = torch.optim.Adam(model_0.parameters(), lr=0.001)
elbo_0 = train(model_0,dataloader,optimizer_0,100,0)


# In[46]:


model_100 = VAE().to(device)
optimizer_100 = torch.optim.Adam(model_100.parameters(), lr=0.001)
elbo_100 = train(model_100,dataloader,optimizer_100,100,100)


# In[47]:


torch.save(model_0.state_dict(), 'scale_0.pt')
torch.save(model_100.state_dict(), 'scale_100.pt')


# In[48]:


model_0.eval()
model_100.eval()


# In[49]:


plt.figure(1)
plt.plot(list(range(100)),elbo_0)
plt.xlabel('Epoches')
plt.ylabel('Loss')
plt.title('Average Learning Curve of the animation face,scale = 0')
plt.figure(2)
plt.plot(list(range(100)),elbo_100)
plt.xlabel('Epoches')
plt.ylabel('Loss')
plt.title('Average Learning Curve of the animation face,scale = 100')
plt.show()


# In[50]:


fixed_x_0 = next(iter(dataloader))[0]
recon_x_0, _, _ = model_0(fixed_x_0)
fixed_x_100 = next(iter(dataloader))[0]
recon_x_100, _, _ = model_100(fixed_x_100)


# In[51]:


plot_recon(torchvision.utils.make_grid(recon_x_0.data[:128],8 ),' Reconstructed Animation faces,scale = 0')
plot_recon(torchvision.utils.make_grid(recon_x_100.data[:128],8 ),' Reconstructed Animation faces,scale = 100')


# In[52]:


z_0 = torch.randn(128, 128)
out1_0 = model_0.decode(z_0)
z_100 = torch.randn(128, 128)
out1_100 = model_100.decode(z_100)


# In[53]:


plot_recon(torchvision.utils.make_grid(out1_0.data[:128],8 ),' Using the latent codes z to reconstruct , scale = 0')
plot_recon(torchvision.utils.make_grid(out1_100.data[:128],8 ),' Using the latent codes z to reconstruct , scale = 100')


# In[54]:


x1_0 = data[50][0].unsqueeze(0)
mu1_0,var1_0 = model_0.encode(x1_0)
z1_0 = model_0.reparameterize(mu1_0, var1_0)
x2_0 = data[70][0].unsqueeze(0)
mu2_0,var2_0 = model_0.encode(x2_0)
z2_0 = model_0.reparameterize(mu2_0,var2_0)
x1_100 = data[50][0].unsqueeze(0)
mu1_100,var1_100 = model_100.encode(x1_100)
z1_100 = model_100.reparameterize(mu1_100, var1_100)
x2_100 = data[70][0].unsqueeze(0)
mu2_100,var2_100 = model_100.encode(x2_100)
z2_100 = model_100.reparameterize(mu2_100,var2_100)


# In[55]:


im_0 = interpolation_af(model_0,z1_0,z2_0,11)
im_100 = interpolation_af(model_100,z1_100,z2_100,11)


# In[56]:


plot_recon(torchvision.utils.make_grid(im_0[:],11 ),'Animation faces interpolation,scale = 0')
plot_recon(torchvision.utils.make_grid(im_100[:],11 ),'Animation faces interpolation,scale = 100')


# In[ ]:




