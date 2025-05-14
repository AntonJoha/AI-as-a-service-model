import model
import resnet
import video
from torch.utils.data import DataLoader
from argparse import ArgumentParser, Namespace
import torch



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
    return parser





def get_dataset(args: Namespace, res)-> DataLoader:
    dataset: video.Video = video.get_video(args, res)
    dataloader: DataLoader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4)
    return dataloader





def init_models(args: Namespace): 
    mod: model.Model = model.get_model(args)
    return mod




if __name__ == "__main__":
    parser: ArgumentParser = get_parser()
    args: Namespace = parser.parse_args()

    res: resnet.Resnet = resnet.get_resnet(args)
    model = init_models(args)
    dataloader: DataLoader = get_dataset(args, res)


    adam = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    for i in range(10):
        for i, data in enumerate(dataloader):
            a = model(data[0])
            a.squeeze_()
            print(a, data[-1])
            loss = criterion(a, data[-1][-1])
            loss.backward()
            adam.step()
            adam.zero_grad()
            print(f"Loss: {loss.item()}")
        
    print(args.verbose, args.model)
    print(type(args.model))


