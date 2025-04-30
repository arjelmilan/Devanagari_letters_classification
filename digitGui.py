import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tkinter as tk
from tkinter import Canvas
from PIL import Image, ImageDraw, ImageOps
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
my_dict = {'क': 0,
 'ञ': 1,
 'ट': 2,
 'ठ': 3,
 'ड': 4,
 'ढ': 5,
 'ण': 6,
 'त': 7,
 'थ': 8,
 'द': 9,
 'ध': 10,
 'ख': 11,
 'न': 12,
 'प': 13,
 'फ': 14,
 'ब': 15,
 'भ': 16,
 'म': 17,
 'य': 18,
 'र': 19,
 'ल': 20,
 'व': 21,
 'ग': 22,
 'श': 23,
 'ष': 24,
 'स': 25,
'ह': 26,
 '34': 27,
 '35': 28,
 '36': 29,
 '37': 30,
 '38': 31,
 '39': 32,
 '4': 33,
 '40': 34,
 '41': 35,
 '42': 36,
 '43': 37,
 '44': 38,
 '45': 39,
 '46': 40,
 '47': 41,
 '48': 42,
 '5': 43,
 '6': 44,
 '7': 45,
 '8': 46,
 '9': 47}

class Net(nn.Module):

    def __init__(self, input_shape: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=32,
                      kernel_size=3,  # how big is the square that's going over the image?
                      stride=1,  # default
                      padding=1),
            # options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=5,  # how big is the square that's going over the image?
                      stride=2,  # default
                      padding=1),
            # options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2))

        self.block_2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from?
            # It's because each layer of our network compresses and changes the shape of our input data.
            nn.Dropout(0.3),
            nn.Linear(in_features=128 * 3 * 3,
                      out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        # print(x.shape)
        x = self.block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x


# Load the trained CNN model
model = Net(input_shape = 3, output_shape = 48)
model.load_state_dict(torch.load("model_weights.pth",map_location = 'cpu'))
model.eval()

# GUI for Drawing and Prediction
class DigitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognizer")

        # Tkinter canvas
        self.canvas = Canvas(root, width=280, height=280, bg="white")
        self.canvas.pack()

        self.button_predict = tk.Button(root, text="Predict", command=self.predict_digit)
        self.button_predict.pack()

        self.button_clear = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.button_clear.pack()

        self.label = tk.Label(root, text="Draw a digit and click 'Predict'")
        self.label.pack()

        self.canvas.bind("<B1-Motion>", self.draw)

        # Use RGB mode for 3-channel input
        self.image = Image.new("RGB", (280, 280), (255, 255, 255))
        self.draw_image = ImageDraw.Draw(self.image)

    def draw(self, event):
        x, y = event.x, event.y
        r = 10
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black", outline="black")
        self.draw_image.ellipse([x - r, y - r, x + r, y + r], fill=(0, 0, 0))

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("RGB", (280, 280), (255, 255, 255))
        self.draw_image = ImageDraw.Draw(self.image)

    def predict_digit(self):
        img = self.image.resize((28, 28))  # Keep RGB, don't convert to L
        # img = ImageOps.invert(img)        # Optional: if your model expects inverted colors
        img = np.array(img, dtype=np.uint8)
        img = img / 255.0  # Normalize

        # Convert to PyTorch tensor: shape [1, 3, 28, 28]
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

        # Get prediction
        with torch.no_grad():
            output = model(img)
            predicted_digit = torch.argmax(output).item()
        devnagri_label = [key for key , value in my_dict.items() if value == predicted_digit]


        self.label.config(text=f"Prediction: {devnagri_label}")

# Run the app
root = tk.Tk()
app = DigitApp(root)
root.mainloop()