import model
import resnet
import video
from torch.utils.data import DataLoader
from argparse import ArgumentParser, Namespace
import torch
import random
from config import get_config
from typing import Any
import time
import datetime
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def now():
    return str(datetime.datetime.fromtimestamp(time.time()).strftime("%d_%m_%y_%H:%M:%S.%f"))

def get_parser()-> ArgumentParser:
    parser: ArgumentParser = ArgumentParser(
             prog="Training AI as a service",
             description="Trains a model meant to evaluate the performance of video",
             )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--model", nargs=1, default="model.json")
    parser.add_argument("--resnet", nargs=1, default="resnet.json")
    parser.add_argument("--video", nargs=1, default="video.json")
    parser.add_argument("--config", nargs=1, default="config.json")
    parser.add_argument("--video_test", nargs=1, default="video_test.json")
    parser.add_argument("--pretrained", nargs=1, default=None)
    return parser





def get_dataset(args: Namespace, res, test: bool=False)-> tuple[DataLoader, DataLoader| None]:
    dataset: video.Video = video.get_video(args.video, res)
    dataloader: DataLoader = DataLoader(dataset, batch_size=3, shuffle=False, num_workers=4)
    testloader: DataLoader = None
    if test:
        testset: video.Video = video.get_video(args.video_test, res)
        testloader: DataLoader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=4)

    return dataloader, testloader





def init_models(args: Namespace)-> model.Model: 
    mod: model.Model = model.get_model(args)
    return mod

def cp(left, right):
    os.popen("cp %s %s" % (left, right))
    
def copy_files(args: Namespace, path: str):
    cp(args.resnet, path+"/resnet.json")
    cp(args.video, path + "/video.json")
    cp(args.video_test, path+"/video_test.json")
    cp(args.model, path + "/model.json")
    cp(args.config, path + "/config.json")

    
def train_model(args: Namespace, model=None)-> None:
    path = "pickle/" + now()
    os.mkdir(path)
    copy_files(args,path)
    config: Any = get_config(args.model)
    res: resnet.Resnet = resnet.get_resnet(args.resnet)
    res.to(device)
    if model is None:
        model = init_models(args).to(device)

    dataloader, testloader= get_dataset(args, res, test=True)



    adam = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    best = 1000000
    for e in range(config["epochs"]):
        x = []
        y = []
        for i, data in enumerate(dataloader):
            
            if random.random() > 0.1:
                continue
            x.append(data[0].squeeze())
            y.append(data[-1][-1])
            if len(x) >= config["batch_size"]:
                #print("START")
                
                x = torch.stack(x,0).to(device)
                y = torch.stack(y,0).to(device)
                a = model(x)[:,-1,:]

            
                a.squeeze_()
            
                loss = criterion(a, y)
                loss.backward()
                adam.step()
                adam.zero_grad()

                x = []
                y = []

        if testloader is not None:
            sum_loss = 0
            count = 0
            for _, testdata in enumerate(testloader):
                x_test = testdata[0].unsqueeze(0).to(device)
                y_test = testdata[-1][-1].to(device)
                a_test = model(x_test)
                a_test = a_test[-1,:]
                a_test.squeeze_()
                loss_test = criterion(a_test, y_test)
                sum_loss += loss_test.item()
                count += 1
            loss = sum_loss/count
            if loss < best:
                save_model(args, path + "/epoch_" + str(e) + "model.pkl", model)
                best = loss
            print(f"Test Loss: {sum_loss/count}")
    
    save_model(args, path + "/final_model.pkl", model)

    return model


def test_model(args: Namespace, model: model.Model):
    res: resnet.Resnet = resnet.get_resnet(args.resnet)

    dataloader, testloader= get_dataset(args, res, test=True)
    criterion = torch.nn.MSELoss()

    sum_loss = 0
    count = 0
    for _, testdata in enumerate(testloader):
        
        x_test = testdata[0].unsqueeze(0).to(device)
        y_test = testdata[-1][-1].to(device)
        a_test = model(x_test)
        a_test = a_test[-1,:]
        a_test.squeeze_()
        loss_test = criterion(a_test, y_test)
        sum_loss += loss_test.item()
        count += 1
    print(f"Test Loss: {sum_loss/count}")


def save_model(args: Namespace,path: str, model: model.Model)-> None:
    if args.verbose:
        print("Saving model")
    torch.save(model.state_dict(), path)
    if args.verbose:
        print("Model saved")


def load_model(args: Namespace, path: str) -> model.Model:
    if args.verbose:

    f = open(path, "r")
    if f.seekable() is False:
        print("Error file is not seekable: " path)
        exit()
    m = model.get_model(args)
    m.load_state_dict(torch.load(path))
    
    m.to(device)
        
    if args.verbose:
        print("Model loaded usign device: ", device)
    return m





if __name__ == "__main__":
    parser: ArgumentParser = get_parser()
    args: Namespace = parser.parse_args()

    m = None
    if args.pretrained is not None:
        m2 = load_model(args, args.pretrained).to(device)
        m = train_model(args, m2)
    else:
        m = train_model(args)
        
    test_model(args, m)


    save_model(args, "pickle/model.pkl", m)

    
    m2 = load_model(args, "pickle/model.pkl")

    test_model(args, m2)

    m2 = load_model(args, "pickle/model.pkl")

    test_model(args, m2)

    m2 = load_model(args, "pickle/model.pkl")

    test_model(args, m2)

    m2 = load_model(args, "pickle/model.pkl")

    test_model(args, m2)


