import ntpath
import os
from PIL import Image, ImageFilter, ImageOps


def extractName(filePath):
    return os.path.splitext(ntpath.basename(filePath))[0]


def extractDirectory(filePath):
    return ntpath.dirname(filePath)


def extractExtension(filePath):
    split = extractName(filePath).split(".")
    return split[len(split) - 1]


def augmentDirectory(path):
    imageList = []
    for file in os.listdir(path):
        imageList.extend(augmentImagePath(path + "/" + file))
    return imageList


def augmentList(list):
    imageList = []
    for image in list:
        imageList.extend(augmentImage(image[1], image[0]))
    return imageList


def augmentImagePath(filepath, applyRotation=True, applyMirror=True, applyGrayscale=True, printGeneratedLists=True,
                     isMask=False):
    image = Image.open(filepath)
    name = extractName(filepath)
    return augmentImage(image, name, applyRotation, applyMirror, applyGrayscale, printGeneratedLists, isMask)


def augmentImage(image: Image, name, applyRotation=True, applyMirror=True, applyGrayscale=True,
                 printGeneratedLists=True, isMask=False):
    """
    Augments an image to extend the neural networks understanding of it
    Only works for images that are orientation independent
    :param name: The filepath for the input image
    :return: a list of images and image names (imageName, image) in the PIL image format, including the original image
    """
    imageList = []
    imageList.append((name, image))

    if applyMirror:
        mirroredImage = mirrorImage(image, name)
        imageList.append(mirroredImage)
    if applyGrayscale:
        if isMask:
            imageList.append((name + "-g", image.copy()))
        else:
            grayscaledImage = grayscaleImage(image, name)
            imageList.append(grayscaledImage)
    if applyGrayscale and applyMirror:
        if isMask:
            grayMirrorImage = mirrorImage(image, name + "-g-m")
            imageList.append(grayMirrorImage)
        else:
            grayMirrorImage = mirrorImage(grayscaledImage[1], grayscaledImage[0])
            imageList.append(grayMirrorImage)

    if applyRotation:
        for i in range(1, 4):
            imageList.append(rotateImage(image, name, i * 90))
            if applyMirror:
                imageList.append(rotateImage(mirroredImage[1], mirroredImage[0], i * 90))
            if applyGrayscale:
                if isMask:
                    imageList.append(rotateImage(image, name + "-g", i * 90))
                else:
                    imageList.append(rotateImage(grayscaledImage[1], grayscaledImage[0], i * 90))
            if applyGrayscale and applyMirror:
                if isMask:
                    imageList.append(rotateImage(mirroredImage[1], mirroredImage[0], i * 90))
                else:
                    imageList.append(rotateImage(grayMirrorImage[1], grayMirrorImage[0], i * 90))

    if printGeneratedLists:
        print(imageList)

    return imageList


def rotateImage(file: Image, name, rotationAmount):
    newImage = file.copy()
    newImage = newImage.rotate(rotationAmount)
    return (name + "-r" + str(rotationAmount), newImage)


def mirrorImage(file: Image, name):
    newImage = file.copy()
    newImage = ImageOps.mirror(newImage)
    return (name + "-m", newImage)


def grayscaleImage(file: Image, name):
    newImage = file.copy()
    newImage = ImageOps.grayscale(newImage)
    newImage = newImage.convert("RGB")
    return (name + "-g", newImage)


def createMaskFromTransparencyPath(filePath):
    pillowImage = Image.open(filePath)
    return (extractName(filePath), createMaskFromTransparency(pillowImage))


def createMaskFromTransparency(file: Image):
    """
    Removes transparency from a mask, turning all non-transparent areas white and all transparent areas black
    :param file: A Pillow image file to convert
    :return: A pillow image file which has been converted
    """
    pillowImage = file.copy()
    pillowImage = pillowImage.convert("RGBA")
    pixdata = pillowImage.load()

    width, height = pillowImage.size
    for y in range(height):
        for x in range(width):
            if pixdata[x, y][-1] != 0:
                pixdata[x, y] = (255, 255, 255, 255)
            else:
                pixdata[x, y] = (0, 0, 0, 255)
    # pillowImage.(pixdata)
    return pillowImage

def padImage(file: Image, newSize, paddingColor = (0, 0, 0)) -> Image:
    paddedImage = Image.new("RGB",
                     newSize,
                     paddingColor) #Pads to black by default
    paddedImage.paste(file)  # Not centered, top-left corner
    return paddedImage

# augmentDirectory("../images/raw/")
# augmentImage("../images/raw/001.jpg")
