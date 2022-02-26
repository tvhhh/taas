import argparse

def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-bool", type=str2bool, nargs="?", default=True)
    parser.add_argument("-int", type=int, default=1)
    parser.add_argument("-float", type=float, default=1.0)
    parser.add_argument("-str", type=str, default="")
    parser.add_argument("-intnone", type=int, default=None)
    parser.add_argument("-floatnone", type=float, default=None)
    parser.add_argument("-strnone", type=str, default=None)

    args = parser.parse_args()

    for arg in vars(args):
        print(arg, getattr(args, arg))
