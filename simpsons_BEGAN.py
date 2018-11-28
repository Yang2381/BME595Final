import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torchvision.utils import make_grid,save_image
import torch.nn.functional as F

# Root directory for dataset
dataroot = "./simpsons_dataset"

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 16

# Number of training epochs
num_epochs = 1000

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

ngpu = 2
#11,121,173,1903,20933
batch_size = 11

dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=True)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print("Device is: ",device)
print(torch.cuda.get_device_name(torch.cuda.current_device()))
print(len(dataloader))
cuda = True if torch.cuda.is_available() else False

def weights_init_normal(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		torch.nn.init.normal_(m.weight.data, 0.0,0.02)
	elif classname.find('BatchNorm2d') != -1:
		torch.nn.init.normal_(m.weight.data, 1.0,0.02)
		torch.nn.init.normal_(m.bias.data, 0.0)


class Decoder(nn.Module):
	"""docstring for Encoder"""
	def __init__(self):
		super(Decoder, self).__init__()
		self.h = 64
		self.n_channel = 64
		self.batch_size = batch_size
		self.l1 = nn.Sequential(nn.Linear(self.h, 8*8*self.n_channel))
		self.decoder = nn.Sequential(
			#In_Channel = 64, Out_Channel = 64, kernel_size =3, Stride =1, Padding =1
			nn.Conv2d(self.n_channel, self.n_channel, 3, 1, 1),
			nn.ELU(inplace = True),
			nn.Conv2d(self.n_channel, self.n_channel, 3, 1, 1),
			nn.ELU(inplace = True),
			nn.Upsample(scale_factor = 2),
			nn.Conv2d(self.n_channel, self.n_channel, 3, 1, 1),
			nn.ELU(inplace = True),
			nn.Conv2d(self.n_channel, self.n_channel, 3, 1, 1),
			nn.ELU(inplace = True),
			nn.Upsample(scale_factor = 2),
			nn.Conv2d(self.n_channel, self.n_channel, 3, 1, 1),
			nn.ELU(inplace = True),
			nn.Conv2d(self.n_channel, self.n_channel, 3, 1, 1),
			nn.ELU(inplace = True),
			nn.Upsample(scale_factor = 2),
			nn.Conv2d(self.n_channel, self.n_channel, 3, 1, 1),
			nn.ELU(inplace = True),
			nn.Conv2d(self.n_channel, self.n_channel, 3, 1, 1),
			nn.ELU(inplace = True),
			nn.Conv2d(self.n_channel, 3,3,1,1),
			#Output Channel is 3
			nn.Tanh()
			)

	def forward(self, z): 
		out = self.l1(z)
		out = out.view(self.batch_size,self.n_channel, 8,8)
		out = self.decoder(out)
		return out

class Encoder(nn.Module):
	"""docstring for Discriminator"""
	def __init__(self):
		super(Encoder, self).__init__()
		self.h = 64
		self.n_channel = 64
		self.batch_size = batch_size
		
		self.encoder = nn.Sequential(
			nn.Conv2d(3, self.n_channel, 3, 1, 1),
			nn.ELU(inplace = True),
			nn.Conv2d(self.n_channel, self.n_channel, 3, 1, 1),
			nn.ELU(inplace = True),
			nn.Conv2d(self.n_channel, self.n_channel, 3, 1, 1),
			nn.ELU(inplace = True),
			nn.Conv2d(self.n_channel, self.n_channel, 1, 1, 0),
			nn.AvgPool2d(2,2),
			nn.Conv2d(self.n_channel, self.n_channel, 3, 1, 1),
			nn.ELU(inplace = True),
			nn.Conv2d(self.n_channel, self.n_channel, 3, 1, 1),
			nn.ELU(inplace = True),
			nn.Conv2d(self.n_channel, 2*self.n_channel, 1, 1, 0),
			nn.AvgPool2d(2,2),
			nn.Conv2d(2*self.n_channel, 2*self.n_channel, 3, 1, 1),
			nn.ELU(inplace = True),
			nn.Conv2d(2*self.n_channel, 2*self.n_channel, 3, 1, 1),
			nn.ELU(inplace = True),
			nn.Conv2d(2*self.n_channel, 3*self.n_channel, 1, 1, 0),
			nn.AvgPool2d(2,2),
			nn.Conv2d(3*self.n_channel, 3*self.n_channel, 3, 1, 1),
			nn.ELU(inplace = True),
			nn.Conv2d(3*self.n_channel, 3*self.n_channel, 3, 1, 1),
			nn.ELU(inplace = True)
			)

		self.linear = nn.Sequential(
			nn.Linear(8*8*3*self.n_channel, self.h)
			)


	def forward(self, z):
		out = self.encoder(z)
		out = out.view(-1, 8*8*3*self.n_channel)
		out = self.linear(out)
		return out

class Discriminator(nn.Module):
	"""docstring for Discriminator"""
	def __init__(self):
		super(Discriminator, self).__init__()
		self.enc = Encoder()
		self.dec = Decoder()

	def forward(self, inputs):
		out = self.enc(inputs)
		return self.dec(out)

class Generator(nn.Module):
	"""docstring for Generator"""
	def __init__(self):
		super(Generator, self).__init__()
		self.dec = Decoder()

	def forward(self, inputs):
		out = self.dec(inputs)
		return out

#generator = Generator()
#discriminator = Discriminator()
generator = Generator().to(device)
discriminator = Discriminator().to(device)


generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

optimizer_G = optim.Adam(generator.parameters(), lr = lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr = lr, betas=(0.5, 0.999))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def save_checkpoint(e,filename='ckpt.tar'):
	modle_states = {
	'G': generator.state_dict(),
	'D': discriminator.state_dict()
	}
	optim_states = {
	'G_optim': optimizer_G.state_dict(),
	'D_optim': optimizer_D.state_dict()
	}
	states = {
	'iter': e,
	'epoch':num_epochs,
	'modle_states': modle_states,
	'optim_states': optim_states
	}
	#file_path = dataroot.joinpath(filename)
	file_path = os.path.join(dataroot, filename)
	torch.save(states, file_path)
	print("Saved Checkpoint '{}' (iter {})".format(file_path, e))

def load_checkpoint(filename='ckpt.tar'):
	file_path = dataroot.joinpath(filename)
	if file_path.is_file():
		checkpiont = torch.load(file_path.open('rb'))
		generator.load_state_dict(checkpiont['modle_states']['G'])
		discriminator.load_state_dict(checkpiont['modle_states']['D'])
		optimizer_G.load_state_dict(checkpoint['optim_states']['G_optim'])
		optimizer_D.load_state_dict(checkpoint['optim_states']['D_optim'])
		e = checkpoint['e']
		print("Load checkpoint '{} at iteration".format(file_path,e))
	else:
		print("Failed to loadfile")

k = 0.0
gamma = 0.75
lambda_k = 0.001
img_list = []
measure_history = []

for epoch in range(num_epochs):
	for i, data in enumerate(dataloader, 0):
		real_img = Variable(data[0]).to(device)

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()

        #Sample Noise Batch * Image Size
        z = Variable(Tensor(batch_size,64)).to(device)

        #Generated batch images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = torch.mean(torch.abs(discriminator(gen_imgs) - gen_imgs))
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        # Measure discriminator's ability to classify real from generated samples
        d_real = discriminator(real_img)
        d_fake = discriminator(gen_imgs.detach())
        d_loss_real = F.l1_loss(d_real, real_img)
        d_loss_fake = F.l1_loss(d_fake, gen_imgs.detach())


        #d_loss_real = torch.mean(torch.abs(d_real - real_img))
        #d_loss_fake = torch.mean(torch.abs(d_fake - gen_imgs.detach()))
        d_loss = d_loss_real - k * d_loss_fake

        d_loss.backward()
        optimizer_D.step()

        #----------------
        # Update weights
        #----------------

        diff = torch.mean(gamma * d_loss_real - d_loss_fake)
       	
        # Update weight term for fake samples
        k = k + lambda_k * diff.item()
        k = min(max(k,0.0),1.0) # Constraint to interval [0, 1]

        M = (d_loss_real + torch.abs(diff)).item()
        measure_history.append(M)
        done = epoch * len(dataloader) + i

        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] -- M: %f, k: %f" % (epoch, num_epochs, i, len(dataloader),
                                                            d_loss.item(), g_loss.item(),
                                                            M, k))
        done = epoch * len(dataloader) + i
        #print(done, done %1000 == 0)
        if i % 500 == 0 or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
        	#with torch.no_grad():
        	#	print("Entered here")
        	#	fake = generator(Variable(Tensor(batch_size,64)).to(device)).detach().cpu()
          	#img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
          	save_checkpoint(i)
  	#print(epoch%1, img_list == [])
        
  	if epoch%10 == 0 and img_list != None:
		
		print("saving images")
		name = "BEGAN_"+str(epoch)+".png"
		print("filename = ",name)
		generator.eval()
		noise = Variable(torch.randn(batch_size,64)).to(device)
		sample = generator(noise).mul(0.5).add(0.5)
		sample = sample.data.cpu()
		grid = make_grid(sample, nrow = 10, padding=2, normalize=False)
		save_image(grid, filename = name)
		generator.train()


		#plt.figure(1)
		#plt.axis("off")
		#plt.title("fake_image")
		#print(img_list)
		#plt.imsave(name,np.transpose(img_list[-1],(1,2,0)))

plt.figure(figsize=(10,5))
plt.title("Measure of Convergence")
plt.plot(measure_history,label="D_x")
plt.xlabel("Iterations")
plt.ylabel("Convergence")
plt.legend()
plt.savefig('convergence.png')
