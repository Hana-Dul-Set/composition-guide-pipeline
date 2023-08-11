import os

from PIL import Image
import torch
import torch.nn as nn
import torchvision.models
from torchvision.transforms import transforms

class CSNet(nn.Module):
    def __init__(self):
        super(CSNet, self).__init__()

        self.backbone = self.build_backbone(pretrained=True)

        self.spp_pool_size = [5, 2, 1]
        
        self.last_layer = nn.Sequential(
            nn.Linear(38400, 1024),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
        )

        self.output_layer = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        
    def forward(self, image):
        feature_map = self.backbone(image)
        spp = self.spatial_pyramid_pool(feature_map, feature_map.shape[0], self.spp_pool_size)
        feature_vector = self.last_layer(spp)

        output = self.output_layer(feature_vector)
        return output
    
    def build_backbone(self, pretrained):
        model = torchvision.models.mobilenet_v2(pretrained)
        modules = list(model.children())[:-1]
        backbone = nn.Sequential(*modules)
        return backbone
    
    # parameter: tensor, batch_size, tensor width and height, spp pool size
    def spatial_pyramid_pool(self, previous_conv, num_sample, out_pool_size):
        for i in range(len(out_pool_size)):
            maxpool = nn.AdaptiveMaxPool2d((out_pool_size[i], out_pool_size[i]))
            x = maxpool(previous_conv)
            if i == 0:
                spp = x.view([num_sample, -1])
            else:
                spp = torch.cat((spp, x.view([num_sample, -1])), 1)
        return spp
    
def get_pretrained_CSNet(weight_path=None):

    model = CSNet()
    if weight_path != None:
        weight_file = os.path.join(weight_path)
        model.load_state_dict(torch.load(weight_file))

    return model

def inference_from_dir(image_dir):
    device = torch.device('cuda:{}'.format(0))
    device = torch.device('cpu')
    model = get_pretrained_CSNet()
    model.eval()
    model.to(device)
    image_name_list = os.listdir(image_dir)

    image_list = []
    for image_name in image_name_list:
        image = Image.open(os.path.join(image_dir, image_name))
        image_list.append({
            'image_name': image_name,
            'image': image
        })
    
    return inference(model, image_list)


def inference(model, image_list):
    device = torch.device('cuda:{}'.format(0))
    device = torch.device('cpu')
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transformer = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transformed_image_list = []
    
    for data in image_list:
        image = data['image']
        transforemd_image = transformer(image)
        transforemd_image.to(device)
        transformed_image_list.append(transforemd_image)
    
    transformed_image_tensor = torch.stack(transformed_image_list, dim=0)
    score_tesnor = model(transformed_image_tensor)
    score_list = score_tesnor.tolist()

    output = []
    for index, score in enumerate(score_list):
        output.append({
            'image_name': image_list[index]['image_name'],
            'score': score
        })
    return output

if __name__ == '__main__':
    model = CSNet()
    model.eval()
    x = torch.randn((1, 3, 224, 224))
    output = model(x)
    print(output)
