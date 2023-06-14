from samp_net import SAMPNet
from inference_dataset import InferenceDataset
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from config import Config

def tensor_normalize(tensor):
    min_value = torch.min(tensor)
    normalized_tensor = tensor - min_value
    tensor_sum = torch.sum(normalized_tensor)
    normalized_tensor = normalized_tensor / tensor_sum
    return normalized_tensor

def get_list_from_tesnor(tensor):
    list = tensor.tolist()[0]
    for i in range(0, len(list)):
        list[i] = round(list[i], 2)
    return list

def print_pattern_weight(weights):
    print("=====PATTERN WEIGHT=====")
    for i in range(len(weights)):
        print('pattern', i+1, ': ', round(weights[i], 2))
    print("========================")
    return

def print_attribute_weight(weights):
    print("=====ATTRIBUTE WEIGHT=====")
    attribute_types = Config.attribute_types
    for i in range(len(weights)):
        print(attribute_types[i], ': ', round(weights[i], 2))
    print("==========================")
    return

def dist2ave(pred_dist):
    pred_score = torch.sum(pred_dist* torch.Tensor(range(1,6)).to(pred_dist.device), dim=-1, keepdim=True)
    return pred_score

def inference(model, cfg):
    model.eval()
    device = next(model.parameters()).device
    testdataset = InferenceDataset(cfg)
    testloader = DataLoader(testdataset,
                            batch_size=cfg.batch_size,
                            shuffle=False,
                            num_workers=cfg.num_workers,
                            drop_last=False)

    inference_output_list = []
    print()
    print('Inference begining...')
    with torch.no_grad():
        for (image_name, im, saliency) in tqdm(testloader):
            image = im.to(device)
            saliency = saliency.to(device)
            weight, atts, output = model(image, saliency)

            pred_score = round(dist2ave(output).item(), 2)
            pattern_weight_list = get_list_from_tesnor(tensor_normalize(weight))
            attribute_weight_list = get_list_from_tesnor(tensor_normalize(atts))     

            print(image_name[0])
            print("pred_score:", pred_score)
            print_pattern_weight(tensor_normalize(weight).tolist()[0])
            print()
            print_attribute_weight(tensor_normalize(atts).tolist()[0])    
            print()    

            inference_output_list.append({
                'image_name': image_name[0], 
                'score': pred_score, 
                'pattern_weight_list': pattern_weight_list, 
                'attribute_weight_list': attribute_weight_list
            })
            """
            pattern_weight_list = get_list_from_tesnor(tensor_normalize(weight))
            attribute_weight_list = get_list_from_tesnor(tensor_normalize(atts))
            
            with open('image_assessment_data.csv', 'a') as file:
                file.writelines(image_name[0] + '/' + str(pred_score) +  '/' + str(pattern_weight_list) + '/' + str(pattern_weight_list) + '/' + str(attribute_weight_list) + '\n')
            """

    print('Inference result...')
    return inference_output_list

def inference_from_dir(image_dir):
    cfg = Config()
    cfg.inf_image_path = image_dir
    device = torch.device('cuda:{}'.format(cfg.gpu_id))
    model = SAMPNet(cfg,pretrained=False).to(device)
    weight_file = './SAMPNet/pretrained_model/samp_net.pth'
    model.load_state_dict(torch.load(weight_file, map_location='cuda:0'))
    return inference(model, cfg)

if __name__ == '__main__':
    cfg = Config()
    device = torch.device('cuda:{}'.format(cfg.gpu_id))
    model = SAMPNet(cfg,pretrained=False).to(device)
    weight_file = './pretrained_model/samp_net.pth'
    model.load_state_dict(torch.load(weight_file, map_location='cuda:0'))
    inference(model, cfg)