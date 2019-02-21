import os

from PIL import Image, ImageOps

for file in os.listdir("images/raw"):
    image = Image.open("images/raw/" + file)
    image = ImageOps.grayscale(image)
    image.save("images/grayscale-raw/" + file, "JPEG")
