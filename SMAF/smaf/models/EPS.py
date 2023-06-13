import torch
import torch.nn as nn
import torch.nn.functional as F

# import network.resnet38d
from models import resnet38d

# class Net(network.resnet38d.Net):
class Net(resnet38d.Net):
    def __init__(self, num_classes):
        super().__init__()

        self.fc8 = nn.Conv2d(128, num_classes, 1, bias=False)
        self.cov_feat = nn.Conv2d(4096, 128, 1, bias=False)
        self.bn = nn.BatchNorm2d(128)
        torch.nn.init.xavier_uniform_(self.fc8.weight)
        torch.nn.init.xavier_uniform_(self.cov_feat.weight)

        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]
        self.from_scratch_layers = [self.fc8,self.cov_feat]

    def forward(self, x):
        x = super().forward(x)
#         print("x.shape {}".format(x.shape))
        x = self.cov_feat(x)
        x = self.bn(x)
        x = F.relu(x)
        cam = self.fc8(x)

        _, _, h, w = cam.size()
        pred = F.avg_pool2d(cam, kernel_size=(h, w), padding=0)

        pred = pred.view(pred.size(0), -1)
        return pred, cam, x

    def forward_cam(self, x):
        x = super().forward(x)
        x = self.cov_feat(x)
        ###加入bn和relu
        x = self.bn(x)
        x = F.relu(x)
        
        cam = self.fc8(x)

        return cam

    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:

                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups
    
def EPS_net(args):
    
    ## 实例化
    model = Net(args.num_classes+1)
    
    ### 导入模型
    if args.weights[-7:] == '.params':
        #assert args.network in ["network.resnet38_cls", "network.resnet38_eps","network.resnet38_part_AD"]
        import models.resnet38d   # models  network
        #weights_dict = network.resnet38d.convert_mxnet_to_torch(args.weights)
        weights_dict = resnet38d.convert_mxnet_to_torch(args.weights)
        model.load_state_dict(weights_dict, strict=False)
    else:
#         weights_dict = torch.load(args.weights)
        checkpoint = torch.load(args.weights)
#         print(checkpoint)
        model.load_state_dict(checkpoint['Net']["model_state"])
        print("load weight succeed")
        
    
    
    ###返回
    return model
    
    