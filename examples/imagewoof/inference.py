import torch
import csv

from ast import literal_eval
from torchvision.utils import make_grid, save_image

from train import get_default_config
from LitDiffusionUnet import LitDiffusionEDM
from WoofLoader import WoofLoader

def read_config_from_csv(fpath):
    config = {}
    with open(fpath) as f:
        csv_reader = csv.reader(f, delimiter=',')
        rows = list(csv_reader)

    for key, value in zip(rows[0], rows[1]):
        try:
            config[key] = literal_eval(value)

        except (ValueError, SyntaxError):
            config[key] = value

    return config


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    torch.set_float32_matmul_precision('high')

    batch_size = 8
    testloader, test_samples = WoofLoader(batch_size = batch_size, fold = "val")


    # hardcoded in a pretrained (very slightly) model
    model_path = "/lustre/storeB/project/metkl/DigitalSeaIce/are-phd/DiffusICE/examples/imagewoof/lightning_logs/version_3/checkpoints/epoch=30-step=12555.ckpt"
    model = LitDiffusionEDM.load_from_checkpoint(
        model_path,
        config = read_config_from_csv("/lustre/storeB/project/metkl/DigitalSeaIce/are-phd/DiffusICE/examples/imagewoof/outputs/configs/configsNone.csv")
    ).eval().to(device)

    means = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
    stds = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(0).unsqueeze(2).unsqueeze(3)

    with torch.no_grad():
        for y, _ in testloader:
            pred = model.generate_samples(batch_size, device = device)

            break 


    images = torch.cat([pred.cpu().detach(), y])

    images * stds + means

    grid = make_grid(images, nrow = 8)

    save_image(grid, "/lustre/storeB/project/metkl/DigitalSeaIce/are-phd/DiffusICE/examples/imagewoof/outputs/figures/dog_grid.png", normalize = True)

if __name__ == "__main__":
    main()