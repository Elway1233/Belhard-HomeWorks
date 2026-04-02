import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
from model import Encoder, Generator, Discriminator, LATENT_DIM

os.makedirs("generated", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 200

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

dataset = datasets.ImageFolder(root="dataset", transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

enc = Encoder().to(DEVICE)
gen = Generator().to(DEVICE)
disc = Discriminator().to(DEVICE)

try:
    enc.load_state_dict(torch.load("saved_models/encoder.pth", map_location=DEVICE))
    gen.load_state_dict(torch.load("saved_models/generator.pth", map_location=DEVICE))
    disc.load_state_dict(torch.load("saved_models/discriminator.pth", map_location=DEVICE))
    print("Успешно загружены веса с прошлой тренировки")
except Exception as e:
    print(f"Не удалось загрузить веса: {e}")


LR_G_E = 0.00001
LR_D =   0.00005

opt_E = optim.Adam(enc.parameters(), lr=LR_G_E, betas=(0.5, 0.999))
opt_G = optim.Adam(gen.parameters(), lr=LR_G_E, betas=(0.5, 0.999))
opt_D = optim.Adam(disc.parameters(), lr=LR_D, betas=(0.5, 0.999),weight_decay=1e-5)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt_G, mode='min', factor=0.5, patience=5)

bce_loss = nn.BCELoss()
l1_loss = nn.L1Loss()

print(f"Начинаем обучение на {DEVICE}...")
print(f"Всего картинок: {len(dataset)}")

START_EPOCH = 140
EPOCHS = 200

for epoch in range(START_EPOCH, EPOCHS):
    epoch_loss_GE = 0
    for i, (real_imgs, _) in enumerate(dataloader):
        real_imgs = real_imgs.to(DEVICE)
        curr_batch_size = real_imgs.size(0)

        real_labels = torch.full((curr_batch_size, 1), 0.9, device=DEVICE)
        fake_labels = torch.full((curr_batch_size, 1), 0.1, device=DEVICE)

        opt_D.zero_grad()

        z_random = torch.randn(curr_batch_size, LATENT_DIM).to(DEVICE)
        fake_imgs = gen(z_random)

        loss_D_real = bce_loss(disc(real_imgs), real_labels)
        loss_D_fake = bce_loss(disc(fake_imgs.detach()), fake_labels)
        loss_D = (loss_D_real + loss_D_fake) / 2

        loss_D.backward()
        opt_D.step()

        opt_G.zero_grad()
        opt_E.zero_grad()

        encoded_z = enc(real_imgs)
        reconstructed_imgs = gen(encoded_z)
        loss_pixel = l1_loss(reconstructed_imgs, real_imgs)

        fake_imgs_2 = gen(z_random)
        loss_GAN = bce_loss(disc(fake_imgs_2), real_labels)

        loss_latent = torch.mean(encoded_z ** 2)
        loss_GE = loss_GAN + 10 * loss_pixel + 0.1 * loss_latent
        loss_GE.backward()

        opt_G.step()
        opt_E.step()
        epoch_loss_GE += loss_GE.item()
    scheduler.step(epoch_loss_GE / len(dataloader))

    print(f"Эпоха [{epoch + 1}/{EPOCHS}] | Ошибка D: {loss_D.item():.4f} | Ошибка G+E: {loss_GE.item():.4f}")

    if (epoch + 1) % 10 == 0:
        save_image((reconstructed_imgs.data + 1) / 2, f"generated/epoch_{epoch + 1}.png", nrow=8)

        torch.save(enc.state_dict(), "saved_models/encoder.pth")
        torch.save(gen.state_dict(), "saved_models/generator.pth")
        torch.save(disc.state_dict(), "saved_models/discriminator.pth")

print("Обучение завершено!")
