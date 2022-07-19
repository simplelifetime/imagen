import torch
from imagen_pytorch import Unet, Imagen, ImagenTrainer
from imagen_pytorch.dataset import CocoCaptionKarpathyDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from imagen_pytorch import t5
import IPython
from torchvision import transforms

def get_parameter_number(net):
    '''
    :param net: model class
    :return: params statistics
    '''
    total_num = sum(p.numel() for p in net.parameters()) / 1000 / 1000
    trainable_num = sum(p.numel() for p in net.parameters()
                        if p.requires_grad) / 1000 / 1000
    return {'Total(M)': total_num, 'Trainable(M)': trainable_num}

def tensor2img(tensor):
    toPIL = transforms.ToPILImage()
    pic = toPIL(tensor)
    pic.save('random.jpg')

# unet for imagen
epochs = 10

load_path = "./logs/epochs_" + str(0) + ".pt"

def get_imagen():
    unet1 = Unet(
        dim = 32,
        cond_dim = 512,
        dim_mults = (1, 2, 4, 8),
        num_resnet_blocks = 3,
        layer_attns = (False, True, True, True),
    )

    unet2 = Unet(
        dim = 32,
        cond_dim = 512,
        dim_mults = (1, 2, 4, 8),
        num_resnet_blocks = (2, 4, 8, 8),
        layer_attns = (False, False, False, True),
        layer_cross_attns = (False, False, False, True)
    )

    # imagen, which contains the unets above (base unet and super resoluting ones)
    return Imagen(
        unets = (unet1, unet2),
        image_sizes = (64, 256),
        timesteps = 1000,
        cond_drop_prob = 0.1
    ).cuda()

imagen = get_imagen()
# wrap imagen with the trainer class
trainer = ImagenTrainer(imagen).cuda()

trainer.load(load_path)

imagen_test = trainer.load(load_path)

IPython.embed()

# images = trainer.imagen.sample(texts = [
#     'rock',
#     'river',
#     'field',
#     'sun',
#     'city',
#     'cockroach'
# ], cond_scale = 3.)

# images = trainer.imagen.sample(texts = [
#     'a whale breaching from afar',
#     'young girl blowing out candles on her birthday cake',
#     'fireworks with blue and green sparkles'
# ], cond_scale = 3.)


