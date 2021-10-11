from .egofnet import egof_net


def get_model_instance_egofnet(mixed_precision=True):
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.mixed_precision = mixed_precision
    return egof_net(args)