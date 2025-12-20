from typing import Any, Optional

from ....configuration_utils import RBLNModelConfig
from ....utils.logging import get_logger
from ...models.siglip import RBLNSiglipVisionModelConfig


logger = get_logger(__name__)


class RBLNLlavaOnevisionForConditionalGenerationConfig(RBLNModelConfig):
    submodules = ["vision_tower", "language_model"]

    def __init__(
        self,
        batch_size: Optional[int] = None,
        vision_tower: Optional[RBLNModelConfig] = None,
        language_model: Optional[RBLNModelConfig] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        self.vision_tower = self.init_submodule_config(
            RBLNSiglipVisionModelConfig,
            vision_tower,
        )

        if self.vision_tower.output_hidden_states is False:
            raise ValueError(
                f"LlavaOnevision requires output_hidden_states to be True, but found output_hidden_states={self.vision_tower.output_hidden_states}. "
                f"Please compile again with the correct argument."
            )
        else:
            self.vision_tower.output_hidden_states = True

        self.language_model = language_model
