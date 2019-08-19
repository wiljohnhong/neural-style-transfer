import torch
import torch.optim as optim
import torchvision.transforms as transforms

from PIL import Image
from config import Config
from model import TransferNet
from torchvision.utils import save_image

config = Config()
device = torch.device('cuda:{}'.format(config.gpu) if torch.cuda.is_available() else 'cpu')


def image_loader(filepath, size=(config.size, config.size)):
    img = Image.open(filepath)
    preprocess = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
    img = preprocess(img).unsqueeze(dim=0)
    return img


def main():
    content_image, style_image = image_loader(config.content_path), image_loader(config.style_path)

    net = TransferNet(content_image, style_image, device, config)

    # perform better if learning starts from origin content image
    generated_image = torch.clone(content_image).to(device).requires_grad_()
    # generated_image = torch.randn_like(content_image).to(device).requires_grad_()

    # Better performance using LBFGS optimizer
    opt = optim.LBFGS([generated_image])

    i_iter = [0]
    while i_iter[0] < config.n_iter:

        def closure():
            i_iter[0] += 1
            # must clamp the `data` within that tensor, otherwise raise RuntimeError:
            # "a leaf Variable that requires grad has been used in an in-place operation."
            generated_image.data.clamp_(0, 1)

            opt.zero_grad()

            content_loss, style_loss = net(generated_image)
            content_loss = config.content_weight * content_loss
            style_loss = config.style_weight * style_loss
            loss = content_loss + style_loss

            if i_iter[0] % config.print_freq == 0:
                print("Iteration:", i_iter[0])
                print("Content loss: {:.4f}, Style loss: {:.4f}".format(content_loss, style_loss))
                print()
            loss.backward()

            return loss

        opt.step(closure)

    generated_image.data.clamp_(0, 1)  # last clamp
    save_image(generated_image.squeeze(), config.generated_path)


if __name__ == '__main__':
    main()

