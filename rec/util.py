
verbose = 1


def Print(*args, **kwargs):
    def _print(*_args, **_kwargs):
        for s in _args:
            print(f"{s}", end="  ")
        print()
        for k, v in _kwargs.items():
            print(f"{k}: {v}", end="  ")
        print()
    if "verbose" in kwargs:
        if kwargs["verbose"]:
            del kwargs["verbose"]
            _print(*args, **kwargs)
    else:
        if verbose:
            _print(*args, **kwargs)
