import torch
import os
from torchvision import transforms
from PIL import Image
from torchvision.utils import save_image
from model import Encoder, Generator

os.makedirs("test_images", exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_image(img_path):
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    return transform(img).unsqueeze(0).to(DEVICE)


enc = Encoder().to(DEVICE)
gen = Generator().to(DEVICE)

try:
    enc.load_state_dict(torch.load("saved_models/encoder.pth", map_location=DEVICE))
    gen.load_state_dict(torch.load("saved_models/generator.pth", map_location=DEVICE))
    enc.eval()
    gen.eval()
except Exception as e:
    print(f"Ошибка загрузки весов: {e}")
    print("Сначала запустите train.py, чтобы обучить нейросеть!")
    exit()

cat_img = process_image("test_images/cat.jpg")
dog_img = process_image("test_images/dog.jpg")

with torch.no_grad():
    z_cat = enc(cat_img)
    z_dog = enc(dog_img)

    steps = 7
    alphas = torch.linspace(0, 1, steps).to(DEVICE)

    generated_images = []

    for alpha in alphas:
        z_hybrid = z_cat * (1 - alpha) + z_dog * alpha
        hybrid_img = gen(z_hybrid)
        generated_images.append(hybrid_img)

    all_imgs = torch.cat(generated_images, dim=0)

all_imgs = (all_imgs + 1) / 2.0

save_image(all_imgs, "test_images/catdog_morphing.png", nrow=steps)
print("Готово! Проверьте папку test_images, там появилась catdog_morphing.png")
