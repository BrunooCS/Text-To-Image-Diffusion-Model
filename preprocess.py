from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def load_data(data_dir="data/tiny-imagenet-200/", image_size=128, batch_size=256):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.RandomHorizontalFlip(),
    ])

    dataset = ImageFolder(root=f"{data_dir}/train", transform=transform)
    
    trainloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    return trainloader, dataset



def load_labels(dataset, words_file = "data/tiny-imagenet-200/words.txt"):

    wnid_to_text = {}
    with open(words_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                wnid, name = parts
                wnid_to_text[wnid] = name.split(",")[0]

    # Build a mapping from numerical class index to text prompt
    label2text = {}
    for idx, wnid in enumerate(dataset.classes):
        if wnid in wnid_to_text:
            label2text[int(idx)] = wnid_to_text[wnid]
        else:
            label2text[int(idx)] = wnid  # Fallback to wnid if missing

    return label2text
            