import torch
import torch.nn as nn


class TransformerIDS(nn.Module):

    def __init__(
        self,
        input_dim=78,
        num_classes=15,
        d_model=64,
        n_heads=4,
        num_layers=2,
        dropout=0.1,
        max_seq_len=32
    ):
        super().__init__()

        # Convert 78 network features into Transformer embedding
        self.embedding = nn.Linear(input_dim, d_model)

        # Learnable positional encoding
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, max_seq_len, d_model)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):

        # x shape:
        # [batch_size, sequence_length, input_dim]
        # Example: [256, 32, 78]

        x = self.embedding(x)

        # Add temporal position information
        x = x + self.positional_encoding[:, :x.size(1), :]

        # Transformer processes all 32 time steps
        x = self.transformer(x)

        # Mean pooling across sequence
        x = x.mean(dim=1)

        # Classification
        output = self.classifier(x)

        return output


if __name__ == "__main__":

    model = TransformerIDS(
        input_dim=78,
        num_classes=15
    )

    sample = torch.randn(5, 32, 78)

    result = model(sample)

    print("Input shape :", sample.shape)
    print("Output shape:", result.shape)