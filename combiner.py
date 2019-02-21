import os

from PIL import Image

for file in os.listdir("segmentation/data/train/raw"):
    filename = file.split(".")[0]
    try:
        dirtImage = Image.open("segmentation/data/train/dirt/" + filename + ".png")
    except IOError:
        dirtImage = Image.new("RGBA", (227, 227), color=(0, 0, 0, 0))

    try:
        wallImage = Image.open("segmentation/data/train/wall/" + filename + ".png")
    except IOError:
        wallImage = Image.new("RGBA", (227, 227), color=(0, 0, 0, 0))

    try:
        polypImage = Image.open("segmentation/data/train/polyp/" + filename + ".png")
    except IOError:
        polypImage = Image.new("RGBA", (227, 227), color=(0, 0, 0, 0))

    mask = Image.new("RGB", (227, 227))
    maskData = mask.load()

    pixdata = dirtImage.load()
    width, height = dirtImage.size
    for y in range(height):
        for x in range(width):
            if pixdata[x, y][-1] != 0:
                maskData[x, y] = (255, 0, 0, 255)

    pixdata = wallImage.load()
    width, height = wallImage.size
    for y in range(height):
        for x in range(width):
            if pixdata[x, y][-1] != 0:
                maskData[x, y] = (0, 255, 0, 255)

    pixdata = polypImage.load()
    width, height = polypImage.size
    for y in range(height):
        for x in range(width):
            if pixdata[x, y][-1] != 0:
                maskData[x, y] = (0, 0, 255, 255)

    mask.save("segmentation/data/train/mask/" + filename + ".jpg", "JPEG")
    print("Saved " + filename)
