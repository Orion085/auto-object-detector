import torch
from torch.utils.data import DataLoader
from src.models.simple_detector import SimpleDetector
from src.data.dataloader import ObjectDetectionDataset

def train():
    dataset = ObjectDetectionDataset(
        images_dir='data/images', 
        annotations_file='data/annotations.json'
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    num_classes = 2  # adjust as needed
    model = SimpleDetector(num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(5):
        model.train()
        for images, targets in dataloader:
            images = torch.stack([torchvision.transforms.ToTensor()(img) for img in images])
            labels = torch.tensor([t['labels'][0] for t in targets])
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} complete. Loss: {loss.item()}")

if __name__ == "__main__":
    train()