# DCGAN-MNIST
DCGAN trained on MNIST handwritten digits dataset with pytorch
(!)Model is not being saved after training

to run, a GPU is required. 

# Training info
epochs:128
learning rates:
- discriminator: 0.0001
- generator: 0.0002
batch size:250
# architectures
generator has 510337 parameters(it could propably be lower). discriminator has 78171. Further details about the specific architecture can be seen in the code.

# Results

![showcase](https://github.com/thebrownfrog/DCGAN-MNIST/assets/158177659/085bbec7-15db-4610-8046-cf20261d26cb)

I forgot to create a proper loss graph, here are screenshots while training:

![Screenshot from 2024-05-28 19-17-23](https://github.com/thebrownfrog/DCGAN-MNIST/assets/158177659/90c6ab52-c731-4aa9-8183-00950875a75f)
![Screenshot from 2024-05-28 19-16-50](https://github.com/thebrownfrog/DCGAN-MNIST/assets/158177659/9fcd6d2a-a4ff-4c9e-bf9c-d2f2d1c4784b)
