import model
import resnet
from argparse import ArgumentParser, Namespace




def get_parser()-> ArgumentParser:
    parser: ArgumentParser = ArgumentParser(
             prog="Training AI as a service",
             description="Trains a model meant to evaluate the performance of video",
             )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--model", nargs=1, default=None)
    parser.add_argument("--resnet", nargs=1, default=None)
    parser.add_argument("--video", nargs=1, default=None)
    parser.add_argument("--config", nargs=1, default=None)
    return parser






def init_models(args: Namespace): 
    mod: model.Model = model.get_model(args)
    res: resnet.Resnet = resnet.get_resnet(args)
    vid: video.Video = video.get_video(args)





if __name__ == "__main__":
    parser: ArgumentParser = get_parser()
    args: Namespace = parser.parse_args()

    print(args.verbose, args.model)
    print(type(args.model))


