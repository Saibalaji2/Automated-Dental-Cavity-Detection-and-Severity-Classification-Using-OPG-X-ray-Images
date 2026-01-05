import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image
from gradcam import GradCAM

LABELS = [
    "No Cavity",
    "Early Enamel Decay",
    "Dentin Decay",
    "Severe / Pulp Involvement"
]

model = models.resnet18()
model.fc = torch.nn.Linear(512, 4)
model.load_state_dict(torch.load("cavity_stage_model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

IMG_PATH = "E:/Dental OPG XRAY Dataset/Augmented_Data/test/images/sample.jpg"

img = Image.open(IMG_PATH).convert("L")
input_tensor = transform(img).unsqueeze(0)

output = model(input_tensor)
pred = torch.argmax(output).item()

cam = GradCAM(model, model.layer4)
heatmap = cam.generate(input_tensor, pred)

img_np = np.array(img.resize((224,224)))
heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(
    cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR),
    0.6,
    heatmap,
    0.4,
    0
)

plt.imshow(overlay)
plt.title(f"Prediction: {LABELS[pred]}")
plt.axis("off")
plt.show()
