"""Independent reference calls cannot alter later compiler launch arguments or buffers."""

import pytest

torch = pytest.importorskip("torch")

from merlin.targetgen import _m2m_capture_worker as worker


def test_float_reference_restores_input_and_buffer_mutations():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("state", torch.zeros(2))

        def forward(self, value):
            self.state.add_(1)
            value.add_(3)
            return value + self.state

    model = Model()
    value = torch.tensor([1.0, 2.0])
    reference = worker._float_reference(model, (value,), torch)
    assert reference["outputs"] == [5.0, 6.0]
    assert torch.equal(value, torch.tensor([1.0, 2.0]))
    assert torch.equal(model.state, torch.zeros(2))


def test_float_reference_restores_metadata_mutation_even_on_error():
    class Model(torch.nn.Module):
        def forward(self, value):
            value.resize_(3, 2)
            raise ValueError("reference failed")

    value = torch.arange(6.0).reshape(2, 3)
    with pytest.raises(ValueError, match="reference failed"):
        worker._float_reference(Model(), (value,), torch)
    assert value.shape == (2, 3)
    assert value.stride() == (3, 1)
    assert torch.equal(value, torch.arange(6.0).reshape(2, 3))
