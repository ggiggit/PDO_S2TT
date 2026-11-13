from types import SimpleNamespace

from pdo_s2tt.delivery_ledger import install_causal_window, select_commit_boundary
from pdo_s2tt.model import PDOS2TT


class FakeFrontend:
    chunk_local = False
    cache_encoder_windows = False
    chunk_samples = 32_000

    def __init__(self):
        self.processed_samples = 0
        self.prefix_pcm = bytearray()
        self.buffer = bytearray()
        self.previous_prefix_frames = 0

    def accept(self, pcm, final=False):
        return []

    def retention_stats(self):
        return {}


class FakeDecoder:
    target_prompt_mode = "stage5_temporary_draft_chat"
    source_decode = False
    max_target_tokens = 128

    def __init__(self):
        self.llm = SimpleNamespace(active_adapter=None)
        self.reset()

    def reset(self):
        target = SimpleNamespace(attention_length=1, past_key_values=object())
        self.state = SimpleNamespace(
            target=target,
            target_ids=[],
            finished=False,
            acoustic=SimpleNamespace(),
        )

    def _physical_cache_length(self, _cache, attention_length):
        return attention_length

    def accept(self, latent, is_final=False, source_latent=None, update_source=True):
        return [], [], is_final

    def decode(self, values):
        return " ".join(str(value) for value in values)


class FakeTranslator:
    def __init__(self):
        self.decoder = FakeDecoder()
        self.frontend = FakeFrontend()

    def accept_pcm(self, pcm, final=False):
        return []

    def reset(self):
        self.decoder.reset()
        self.frontend = FakeFrontend()


def test_commit_boundary_prefers_a_unique_overlap():
    result = select_commit_boundary(
        [10, 11, 12, 13], [12, 13, 14], retired_frames=2, retained_frames=2
    )
    assert result["selected_split"] == 2
    assert result["selection_reason"] == "unique_exact_anchor"


def test_window_install_uninstall_restores_methods():
    translator = FakeTranslator()
    original_decoder_accept = translator.decoder.accept
    original_accept_pcm = translator.accept_pcm
    controller = install_causal_window(
        translator, window_blocks=12, retire_blocks=6, source_frame_cap=336
    )
    assert controller.window_blocks == 12
    assert controller.retire_blocks == 6
    assert translator.decoder.accept != original_decoder_accept
    assert translator.accept_pcm != original_accept_pcm
    controller.uninstall(translator)
    assert translator.decoder.accept == original_decoder_accept
    assert translator.accept_pcm == original_accept_pcm


def test_model_reset_reinstalls_window_for_each_session():
    model = PDOS2TT.__new__(PDOS2TT)
    model.translator = FakeTranslator()
    model._window_config = {
        "event_cap": 12,
        "retire_events": 6,
        "source_frame_cap": 336,
    }
    model._window_controller = None
    model._install_window()
    first = model._window_controller
    model.reset()
    second = model._window_controller
    assert second is not first
    assert second.window_blocks == 12
    assert second.retire_blocks == 6
    model.reset()
    assert model._window_controller is not second
