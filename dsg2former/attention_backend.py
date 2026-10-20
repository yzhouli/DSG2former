"""Explicit attention switch; dense remains the default for old experiments."""
from .sds_aligned import PaperAlignedDSG2former
from .sparse_aligned import SparseDSG2former


def make_model(attention='dense', **options):
    if attention == 'dense':
        return PaperAlignedDSG2former(**options)
    if attention == 'sparse':
        return SparseDSG2former(**options)
    raise ValueError('attention must be dense or sparse')


def model_from_checkpoint(checkpoint, attention=None):
    """Load parameters strictly; selecting a new topology does not refit weights.

    Save attention_backend='sparse' in new sparse checkpoints so that later loads
    cannot silently reinterpret them as the paper-aligned dense model.
    """
    if attention is None:
        attention = checkpoint.get('attention_backend')
        if attention is None:
            attention = 'sparse' if checkpoint.get('architecture') == 'SparseDSG2former' else 'dense'
    model = make_model(attention, **checkpoint.get('options', {}))
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model
