import os
import torch
from torchvision import transforms
from ghostnet import ghostnet
from PIL import Image

with open('./label.txt', 'r') as f:
    text_labels = [''.join(l.split("'")[1]) for l in f]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

model = ghostnet().to(device)
model.load_state_dict(torch.load('./models/state_dict_93.98.pth'))

image_list = os.listdir('./samples')
with torch.no_grad():
    model.eval()
    for image_name in image_list:
        image_path = os.path.join('./samples', image_name)
        print(image_path)
        img = Image.open(image_path).convert('RGB')
        tensor = transform(img).unsqueeze(0)
        output = model(tensor.to(device))
        pred = output.max(1, keepdim=True)[1]
        print(f'result: {text_labels[pred]}')
