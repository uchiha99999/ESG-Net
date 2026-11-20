import os

esi_backbones = {}
def register_multimodal_backbone(name):
    def decorator(cls):
        esi_backbones[name] = cls
        return cls
    return decorator

mode_blocks = {}
def register_dependency_block(name):
    def decorator(cls):
        mode_blocks[name] = cls
        return cls
    return decorator

multimodal_meta_archs = {}
def register_multimodal_meta_arch(name):
    def decorator(cls):
        multimodal_meta_archs[name] = cls
        return cls
    return decorator

# builder functions
def make_multimodal_backbone(name, **kwargs):
    esi_backbone = esi_backbones[name](**kwargs)
    return esi_backbone

def make_dependency_block(name, **kwargs):
    mode_block = mode_blocks[name](**kwargs)
    return mode_block

def make_multimodal_meta_arch(name, **kwargs):
    multimodal_meta_arch = multimodal_meta_archs[name](**kwargs)
    return multimodal_meta_arch

