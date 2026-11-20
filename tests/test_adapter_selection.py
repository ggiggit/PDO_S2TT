from streaming_s2tt.native_true_dual_stream import _set_adapter_if_needed


class Layer:
    def __init__(self, adapter="default"):
        self.active_adapters = [adapter]


class Model:
    def __init__(self):
        self.active_adapters = ["default"]
        self.layers = [Layer(), Layer()]
        self.calls = []

    def set_adapter(self, adapter):
        self.calls.append(adapter)
        self.active_adapters = [adapter]
        for layer in self.layers:
            layer.active_adapters = [adapter]


def test_identical_adapter_selection_is_skipped():
    model = Model()
    _set_adapter_if_needed(model, tuple(model.layers), "default")
    assert model.calls == []


def test_top_level_adapter_change_uses_native_selection():
    model = Model()
    _set_adapter_if_needed(model, tuple(model.layers), "source")
    assert model.calls == ["source"]


def test_child_adapter_change_uses_native_selection():
    model = Model()
    model.layers[0].active_adapters = ["source"]
    _set_adapter_if_needed(model, tuple(model.layers), "default")
    assert model.calls == ["default"]
