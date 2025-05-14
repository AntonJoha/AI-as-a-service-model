import model
import resnet
import video
from torch.utils.data import DataLoader
from argparse import ArgumentParser, Namespace
import torch
import random
from config import get_config
from typing import Any


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
    return parser





def get_dataset(args: Namespace, res, test: bool=False)-> tuple[DataLoader, DataLoader| None]:
    dataset: video.Video = video.get_video(args.video, res)
    dataloader: DataLoader = DataLoader(dataset, batch_size=3, shuffle=False, num_workers=4)
    testloader: DataLoader = None
    if test:
        testset: video.Video = video.get_video(args.video_test, res)
        testloader: DataLoader = DataLoader(dataset, batch_size=3, shuffle=False, num_workers=4)

    return dataloader, testloader





def init_models(args: Namespace)-> model.Model: 
    mod: model.Model = model.get_model(args)
    return mod



def train_model(args: Namespace)-> None:

    config: Any = get_config(args.model)
    res: resnet.Resnet = resnet.get_resnet(args.resnet)
    model = init_models(args)

    dataloader, testloader= get_dataset(args, res, test=True)



    adam = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    for _ in range(config["epochs"]):
        x = []
        y = []
        for _, data in enumerate(dataloader):
            
            if random.random() > 0.5:
                continue

            
            x.append(data[0].squeeze())
            y.append(data[-1][-1])
            if len(x) >= config["batch_size"]:
                
                x = torch.stack(x,0)
                y = torch.stack(y,0)
                a = model(x)[:,-1,:]
            
                a.squeeze_()
            
                loss = criterion(a, y)
                loss.backward()
                adam.step()
                adam.zero_grad()
                print(f"Loss: {loss.item()}")

                x = []
                y = []

            if testloader is not None:
                sum_loss = 0
                count = 0
                for _, testdata in enumerate(testloader):
                    x_test = testdata[0].squeeze()
                    y_test = testdata[-1][-1]
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
    model.save(args.model[0])
    if args.verbose:
        print("Model saved")



if __name__ == "__main__":
    parser: ArgumentParser = get_parser()
    args: Namespace = parser.parse_args()

    train_model(args)

