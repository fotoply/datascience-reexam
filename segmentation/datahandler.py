import numpy
from PIL import Image

imageHeight = 227

imageWidth = 227


def imageToNetworkInput(image: Image):
    return numpy.asarray(image)


def networkToImage(inputList) -> Image:
    image = Image.new("RGBA", (imageWidth, imageHeight), (0, 0, 0))
    imageData = image.load()
    for i in range(inputList.size):
        x = i % imageWidth
        y = numpy.math.floor(i / imageHeight) % imageHeight
        channel = numpy.math.floor(i / (imageWidth * imageHeight))

        value = list(imageData[x, y])
        value[channel] = inputList[0][i]
        imageData[x, y] = tuple(value)
    # image.putdata(imageData)
    return image


def channelNetworkToImage(inputList) -> Image:
    inputList = inputList[0]
    image = Image.new("RGB", (imageWidth, imageHeight), (255, 255, 255))
    imageData = image.load()
    for i in range(0, inputList.size, 3):
        x = i % imageWidth
        y = numpy.math.floor(i / imageHeight) % imageHeight
        r = int(inputList[i] * 255)
        g = int(inputList[i + 1] * 255)
        b = int(inputList[i + 2] * 255)
        imageData[x, y] = (r, g, b)
    # image.putdata(imageData)
    return image


def maskToOutput(image: Image):
    elements = []
    data = image.load()
    width, height = image.size
    for y in range(height):
        for x in range(width):
            elements.append(data[x, y][0])
    return elements


def channelMaskToOutput(image: Image):
    elements = []
    image = image.convert("RGB")
    data = image.load()
    width, height = image.size
    for y in range(height):
        for x in range(width):
            elements.append(data[x, y][0])
            elements.append(data[x, y][1])
            elements.append(data[x, y][2])
    return elements

def flattenImage(image: Image) -> list:
    rArray = []
    gArray = []
    bArray = []

    data = image.load()
    width, height = image.size
    for y in range(height):
        for x in range(width):
            rArray.append(data[x, y][0])
            gArray.append(data[x, y][1])
            bArray.append(data[x, y][2])

    values = []
    values.extend(rArray)
    values.extend(gArray)
    values.extend(bArray)

    return values

def unflattenImage(values) -> Image:
    size = 227*227
    image = Image.new("RGB", (227, 227))

    data = image.load()
    width, height = image.size
    for y in range(height):
        for x in range(width):
            i = y * 227 + x
            data[x, y] = (values[i], values[i+size], values[i+size*2])

    return image