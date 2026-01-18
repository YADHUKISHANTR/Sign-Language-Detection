import cv2
import numpy as np
import torch
import torch.nn as nn
from cvzone.HandTrackingModule import HandDetector

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = r"C:\Users\HP\Downloads\filtered dataset\sign_language_cnn.pth" 

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(64 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, 3)  # 3 classes
        self.relu  = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Load model
model = CNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

class_map = {0: 'A', 1: 'B', 2: 'C'}

def resize_with_padding(img, size=28):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    x_offset = (size - new_w) // 2
    y_offset = (size - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

while True:
    success, img = cap.read()
    if not success:
        break

    hands, _ = detector.findHands(img, draw=False)
    predicted_letter = ''  

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        cx = x + w // 2
        cy = y + h // 2
        margin = 20
        crop_size = max(w, h) + margin

        x1 = max(0, cx - crop_size // 2)
        y1 = max(0, cy - crop_size // 2)
        x2 = min(img.shape[1], cx + crop_size // 2)
        y2 = min(img.shape[0], cy + crop_size // 2)

        hand_img = img[y1:y2, x1:x2]
        if hand_img.size != 0:
            hand_img_28 = resize_with_padding(hand_img, 28)
            gray_scale = cv2.cvtColor(hand_img_28, cv2.COLOR_BGR2GRAY)

            normalized = gray_scale / 255.0
            standardized = (normalized - 0.5) / 0.5

            hand_tensor = torch.tensor(standardized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                outputs = model(hand_tensor)
                _, predicted = torch.max(outputs, 1)
                predicted_letter = class_map.get(predicted.item(), '?')
                print("Predicted letter:", predicted_letter)

            # Show hand crop
            cv2.imshow("Hand 28x28", gray_scale)

    if predicted_letter:
        cv2.putText(img, f"Prediction: {predicted_letter}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("Webcam Feed", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
