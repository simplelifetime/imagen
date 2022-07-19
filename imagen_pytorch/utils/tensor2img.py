from torchvision import transforms


def tensor2img(tensor):
    toPIL = transforms.ToPILImage()
    pic = toPIL(tensor)
    pic.save('random.jpg')