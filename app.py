
import streamlit as st
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

class Generator(torch.nn.Module):
    def __init__(self, latent_dim=100, num_classes=10):
        super().__init__()
        self.label_emb = torch.nn.Embedding(num_classes, num_classes)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(latent_dim + num_classes, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 784),
            torch.nn.Tanh()
        )

    def forward(self, noise, labels):
        x = torch.cat((noise, self.label_emb(labels)), dim=1)
        img = self.model(x)
        return img.view(img.size(0), 1, 28, 28)

device = torch.device("cpu")
generator = Generator().to(device)
generator.load_state_dict(torch.load("mnist_generator.pth", map_location=device))
generator.eval()

st.title("Handwritten Digit Generator")
digit = st.number_input("Enter a digit (0–9):", min_value=0, max_value=9, step=1)

if st.button("Generate"):
    base_noise = torch.randn(5, 100)
    diversity_noise = torch.randn(5, 100) * 0.25  
    noise = base_noise + diversity_noise  
    labels = torch.tensor([digit] * 5)

    with torch.no_grad():
        images = generator(noise, labels)
        images = torch.clamp(images, -1, 1)  #

    grid = make_grid(images, nrow=5, normalize=True)
    npimg = grid.numpy().transpose((1, 2, 0))
    st.image(npimg, caption=f"Generated images for digit {digit}")
  