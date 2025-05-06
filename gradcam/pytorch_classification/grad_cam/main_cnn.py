import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(num_classes=7)
    model_weight_pth = '/home/q22301197/DG/CIRL/CIRL_main/train_resoult/PACS_ResNet18/cartoon/2023-10-03-19-11-53/best_model.tar'
    
    # model.load_state_dict(torch.load(model_weight_pth,map_location=device),strict=False)
    state_dict = torch.load(model_weight_pth, map_location=lambda storage, loc: storage)
    encoder_state = state_dict["encoder_state_dict"]
    model.load_state_dict(encoder_state)
    target_layers = [model.layer4]

    # model = models.vgg16(pretrained=True)
    # target_layers = [model.features]

    # model = models.resnet34(pretrained=True)
    # target_layers = [model.layer4]

    # model = models.regnet_y_800mf(pretrained=True)
    # target_layers = [model.trunk_output]

    # model = models.efficientnet_b0(pretrained=True)
    # target_layers = [model.features]

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # load image
    img_path = "/home/q22301197/DG/CIRL/CIRL_main/gradcam/pytorch_classification/grad_cam/both.png"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    # img = center_crop_img(img, 224)

    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    target_category = 0  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
        # 保存图像到指定路径
    save_path = "/home/q22301197/DG/CIRL/CIRL_main/gradcam/pytorch_classification/grad_cam/visualization.png"
    plt.imsave(save_path, visualization)
    print(f"Visualization saved to: {save_path}")
    # plt.imshow(visualization)
    # plt.show()


if __name__ == '__main__':
    main()
