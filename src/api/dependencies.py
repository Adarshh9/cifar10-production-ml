"""Dependency injection for FastAPI routes."""

_predictor = None
_model_loader = None
_cache = None

def set_predictor(predictor):
    global _predictor
    _predictor = predictor

def set_model_loader(model_loader):
    global _model_loader
    _model_loader = model_loader

def set_cache(cache):
    global _cache
    _cache = cache

def get_predictor():
    return _predictor

def get_model_loader():
    return _model_loader

def get_cache():
    return _cache
