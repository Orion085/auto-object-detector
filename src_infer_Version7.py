import torch
from PIL import Image
import torchvision.transforms as T
from src.models.simple_detector import SimpleDetector

def infer(image_path, model_path, class_names):
    model = SimpleDetector(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.Resize((224,224)), T.ToTensor()])
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).item()
    print(f"Prediction: {class_names[pred]}")

if __name__ == "__main__":
    infer('test.jpg', 'model.pth', ['person', 'car'])