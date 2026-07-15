from torchvision import transforms


def get_transform():
    return transforms.Compose([transforms.ToTensor()])
