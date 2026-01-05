import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from dataset import DentalOPGDataset

ROOT_DIR = "E:/project/Dental OPG XRAY Dataset"

dataset = DentalOPGDataset(
    csv_file="labels.csv",
    root_dir=ROOT_DIR
)

loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = models.resnet18(weights="DEFAULT")
model.fc = nn.Linear(512, 4)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(15):
    running_loss = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss: {running_loss:.4f}")

torch.save(model.state_dict(), "cavity_stage_model.pth")
