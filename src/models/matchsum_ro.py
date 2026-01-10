import torch
import torch.nn as nn
from transformers import AutoModel


class RomanianMatchSum(nn.Module):
    """
    Siamese-BERT architecture for Extractive Summarization.
    Adapted for Romanian Language.
    """

    def __init__(
        self, model_name: str = "dumitrescustefan/bert-base-romanian-cased-v1"
    ):
        super(RomanianMatchSum, self).__init__()

        print(f"🏗️ Initializing RomanianMatchSum with {model_name}...")

        # Load the base Romanian BERT
        self.bert = AutoModel.from_pretrained(model_name)

        # Optimization: Freeze the first 6 layers.
        # The lower layers learn basic syntax (which Ro-BERT already knows).
        # We only want to fine-tune the upper layers for "Summarization Logic".
        # Check if 'encoder' attribute exists to avoid errors on some HF models
        if hasattr(self.bert, "encoder"):  # type: ignore
            for param in self.bert.encoder.layer[:6].parameters():
                param.requires_grad = False

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Returns the [CLS] embedding which represents the semantic meaning
        of the input text (Document or Summary).
        """

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # The [CLS] token is always at index 0.
        # Shape: [Batch_Size, Hidden_Size (768)]
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        return cls_embedding
