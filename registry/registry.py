class _Registry:
    def __init__(self):
        self._registry = {}

    def register(self, name=None):

        def decorator(obj):
            key = name or obj.__name__
            if key in self._registry:
                raise ValueError(f"{key} is already registered.")
            self._registry[key] = obj
            return obj

        return decorator

    def get(self, name):
        if name not in self._registry:
            raise KeyError(f"{name} not found in registry.")
        return self._registry[name]

    def list(self):
        return list(self._registry.keys())

    def __getitem__(self, name):
        return self.get(name)


# Global registries
MODELS = _Registry()
LOSSES = _Registry()
ACQUISITION_FUNCTIONS = _Registry()