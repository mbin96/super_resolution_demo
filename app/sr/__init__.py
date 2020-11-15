import argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

from .model import UNETPP
from .utils import convert_rgb_to_y, denormalize, calc_psnr
from torchsummary import summary

import scipy.misc

class srnet():
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model2x = UNETPP(2)
        self.model4x = UNETPP(4)

        self.model2x.to(self.device)
        self.model4x.to(self.device)

        state_dict = self.model2x.state_dict()
        for n, p in torch.load('app/sr/unetppbp-2x.pt', map_location=lambda storage, loc: storage).items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
            else:
                raise KeyError(n)
        
        state_dict = self.model4x.state_dict()
        for n, p in torch.load('app/sr/unetppbp-4x.pt', map_location=lambda storage, loc: storage).items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
            else:
                raise KeyError(n)

        self.model2x.eval()
        self.model4x.eval()
    
    def get_sr_image(self, path, image_file, scale):
        lr = pil_image.open(path+image_file).convert('RGB')
        bicubic = lr.resize((lr.width * scale, lr.height * scale), resample=pil_image.BICUBIC)
        bicubic.save(path + f'bicubic_{scale}x_'+ image_file)
        reqSplit = (lr.width * lr.height * scale * scale // 900000) + 1
        lr = np.expand_dims(np.array(lr).astype(np.float32).transpose([2, 0, 1]), 0)

            



        if scale == 2:
            model = self.model2x
        
        elif scale == 4:
            model = self.model4x

        # try : 
        if reqSplit > 1:
            print('split')
            print(lr.size)
            lr = np.array_split(lr, reqSplit,2)
            # print(lr)
            preds_li = []
            for lr_s in lr:
                print(lr_s.size)
                lr_s = torch.from_numpy(lr_s).to(self.device)
                # pil_image.fromarray(lr_s.data.permute(1, 2, 0).byte().cpu().numpy()).save('hell.png')
                print('save')
                with torch.no_grad():
                    preds = model(lr_s).squeeze(0)
                preds[preds>255]=255
                preds[preds<0]=0
                preds_li.append(preds.data.permute(1, 2, 0).byte().cpu().numpy())
                torch.cuda.empty_cache()
            sr_img = np.concatenate(preds_li,0)

        
        else:

            print('hole')
            print(lr.size)
            lr = torch.from_numpy(lr).to(self.device)
            print(lr)
            
            with torch.no_grad():
                preds = model(lr).squeeze(0)
            preds[preds>255]=255
            preds[preds<0]=0
            sr_img = preds.data.permute(1, 2, 0).byte().cpu().numpy()

        output = pil_image.fromarray(sr_img)

            
        # except RuntimeError:
        #     del lr
        #     del bicubic

        #     torch.cuda.empty_cache()
        #     return [image_file,f'bicubic_{scale}x_' + image_file,'fail']

        output.save(path + f'sr_{scale}x_'+image_file)
        del output
        del lr
        del sr_img
        del bicubic
        del preds
        torch.cuda.empty_cache()
        return [image_file, f'bicubic_{scale}x_' + image_file, f'sr_{scale}x_' + image_file]

    def get_sr_image_jpeg(self, path, image_file, scale=4):
        lr = pil_image.open(path+image_file).convert('RGB')
        bicubic = lr.resize((lr.width * 2, lr.height * 2), resample=pil_image.BICUBIC)
        bicubic.save(path + f'bicubic_{scale}x_'+ image_file)
        lr = lr.resize((lr.width//2, lr.height//2), resample=pil_image.BICUBIC)
        reqSplit = (lr.width * lr.height * scale * scale // 900000) + 1
        lr = np.expand_dims(np.array(lr).astype(np.float32).transpose([2, 0, 1]), 0)

            

        model = self.model4x

        # try : 
        if reqSplit > 1:
            print('split')
            print(lr.size)
            lr = np.array_split(lr, reqSplit,2)
            # print(lr)
            preds_li = []
            for lr_s in lr:
                print(lr_s.size)
                lr_s = torch.from_numpy(lr_s).to(self.device)
                # pil_image.fromarray(lr_s.data.permute(1, 2, 0).byte().cpu().numpy()).save('hell.png')
                print('save')
                with torch.no_grad():
                    preds = model(lr_s).squeeze(0)
                preds[preds>255]=255
                preds[preds<0]=0
                preds_li.append(preds.data.permute(1, 2, 0).byte().cpu().numpy())
                torch.cuda.empty_cache()
            sr_img = np.concatenate(preds_li,0)

        
        else:

            print('hole')
            print(lr.size)
            lr = torch.from_numpy(lr).to(self.device)
            print(lr)
            
            with torch.no_grad():
                preds = model(lr).squeeze(0)
            preds[preds>255]=255
            preds[preds<0]=0
            sr_img = preds.data.permute(1, 2, 0).byte().cpu().numpy()

        output = pil_image.fromarray(sr_img)

            
        # except RuntimeError:
        #     del lr
        #     del bicubic

        #     torch.cuda.empty_cache()
        #     return [image_file,f'bicubic_{scale}x_' + image_file,'fail']

        output.save(path + f'sr_{scale}x_'+image_file)
        del output
        del lr
        del sr_img
        del bicubic
        del preds
        torch.cuda.empty_cache()
        return [image_file, f'bicubic_{scale}x_' + image_file, f'sr_{scale}x_' + image_file]

    def get_bench_image(self, path, image_file, scale):
        image = pil_image.open(path+image_file).convert('RGB')

        image_width = (image.width // scale) * scale
        image_height = (image.height // scale) * scale

        hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr.width // scale, hr.height // scale), resample=pil_image.BICUBIC)
        lr.save(path + f'lr_{scale}x_'+ image_file)
        bicubic = lr.resize((lr.width * scale, lr.height * scale), resample=pil_image.BICUBIC)
        bicubic.save(path + f'bicubic_{scale}x_'+ image_file)

        lr = np.expand_dims(np.array(lr).astype(np.float32).transpose([2, 0, 1]), 0)
        
        lr = torch.from_numpy(lr).to(self.device)
        
        if scale == 2:
            model = self.model2x
        
        elif scale == 4:
            model = self.model4x

        try : 
            with torch.no_grad():
                preds = model(lr).squeeze(0)
            preds[preds>255]=255
            preds[preds<0]=0
            sr_img = preds.data.permute(1, 2, 0).byte().cpu().numpy()
            output = pil_image.fromarray(sr_img)

        except RuntimeError:
            del lr
            del hr
            del bicubic
            del image
            
            torch.cuda.empty_cache()
            return [f'lr_{scale}x_' + image_file,f'bicubic_{scale}x_' + image_file,'fail']


        output.save(path + f'sr_{scale}x_'+image_file)
        del output
        del lr
        del hr
        del sr_img
        del bicubic
        del image
        del preds
        torch.cuda.empty_cache()
        return [f'lr_{scale}x_' + image_file,f'bicubic_{scale}x_' + image_file,f'sr_{scale}x_' + image_file]
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--image-file', type=str, required=True)
    parser.add_argument('--num-features', type=int, default=64)
    parser.add_argument('--growth-rate', type=int, default=64)
    parser.add_argument('--num-blocks', type=int, default=16)
    parser.add_argument('--num-layers', type=int, default=8)
    parser.add_argument('--scale', type=int, default=4)
    args = parser.parse_args()
    
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = UNETPP(args.scale)
    model.to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()

    image = pil_image.open(args.image_file).convert('RGB')

    image_width = (image.width // args.scale) * args.scale
    image_height = (image.height // args.scale) * args.scale

    hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    lr = hr.resize((hr.width // args.scale, hr.height // args.scale), resample=pil_image.BICUBIC)
    lr.save(f'lr_{args.scale}x_'+ args.image_file)
    bicubic = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
    bicubic.save(f'bicubic_{args.scale}x_'+ args.image_file)

    lr = np.expand_dims(np.array(lr).astype(np.float32).transpose([2, 0, 1]), 0) #/ 255.0
    hr = np.expand_dims(np.array(hr).astype(np.float32).transpose([2, 0, 1]), 0) #/ 255.0

    # lr = np.expand_dims(np.array(lr).transpose([2, 0, 1]), 0) #/ 255.0
    # hr = np.expand_dims(np.array(hr).transpose([2, 0, 1]), 0) #/ 255.0
    # lr = np.array(lr)
    # hr = np.array(hr)
    
    lr = torch.from_numpy(lr).to(device)
    hr = torch.from_numpy(hr).to(device)

    with torch.no_grad():                           
        preds = model(lr).squeeze(0)

    # preds_y = convert_rgb_to_y(denormalize(preds), dim_order='chw')
    # hr_y = convert_rgb_to_y(denormalize(hr.squeeze(0)), dim_order='chw')

    # preds_y = preds_y[args.scale:-args.scale, args.scale:-args.scale]
    # hr_y = hr_y[args.scale:-args.scale, args.scale:-args.scale]

    # psnr = calc_psnr(hr_y, preds_y)
    # print('PSNR: {:.2f}'.format(psnr))
    # output = pil_image.fromarray(preds)
    # print(preds>=256)
    preds[preds>255]=255

    preds[preds<0]=0
    output = pil_image.fromarray(preds.permute(1, 2, 0).byte().cpu().numpy())
    # output = pil_image.fromarray(denormalize(preds).permute(1, 2, 0).byte().cpu().numpy())
    output.save(f'sr_{args.scale}x_'+args.image_file)
