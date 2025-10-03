import torch
from torch import nn

from scenetokens.models.components import common
from scenetokens.models.components.cross_attention import CrossAttentionBlock
from scenetokens.models.components.self_attention import FactorizedSelfAttentionBlock
from scenetokens.schemas.output_schemas import ScenarioEmbedding


class FactorizedEmbedder(nn.Module):
    """FactorizedEmbedder class."""

    def __init__(  # noqa: PLR0913
        self,
        num_pre_self_attention_blocks: int,
        num_pre_cross_attention_blocks: int,
        num_mid_self_attention_blocks: int,
        num_mid_cross_attention_blocks: int,
        num_post_self_attention_blocks: int,
        hidden_size: int,
        num_timesteps: int,
        num_agents: int,
        num_queries: int,
        num_heads: int,
        widening_factor: int,
        dropout: float = 0.0,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()

        def get_self_attention_block(across: str) -> FactorizedSelfAttentionBlock:
            return FactorizedSelfAttentionBlock(
                input_size=hidden_size,
                hidden_size=hidden_size,
                widening_factor=widening_factor,
                num_heads=num_heads,
                dropout=dropout,
                bias=bias,
                across=across,
            )

        def get_cross_attention_block() -> CrossAttentionBlock:
            return CrossAttentionBlock(
                input_size=hidden_size,
                hidden_size=hidden_size,
                widening_factor=widening_factor,
                num_heads=num_heads,
                dropout=dropout,
                bias=bias,
            )

        self.attention_blocks = []

        # Social-Temporal encoding. Largely based on factorized encoding (https://arxiv.org/pdf/2106.08417.pdf) which
        # sequentially adds a transformer block across agents after a block across time.
        for _ in range(num_pre_self_attention_blocks):
            self.attention_blocks.append(get_self_attention_block(across="time"))
            self.attention_blocks.append(get_self_attention_block(across="agents"))

        # Cross-attention block for agents and map
        for _ in range(num_pre_cross_attention_blocks):
            self.attention_blocks.append(get_cross_attention_block())

        # Intermediate self-attention and cross-attention blocks
        for _ in range(num_mid_self_attention_blocks):
            self.attention_blocks.append(get_self_attention_block(across="time"))
            self.attention_blocks.append(get_self_attention_block(across="agents"))

        # Cross-attention block for agents and map
        for _ in range(num_mid_cross_attention_blocks):
            self.attention_blocks.append(get_cross_attention_block())

        # Post self-attention blocks
        for _ in range(num_post_self_attention_blocks):
            self.attention_blocks.append(get_self_attention_block(across="time"))
            self.attention_blocks.append(get_self_attention_block(across="agents"))

        # Combine all blocks
        self.attention_blocks = nn.ModuleList(self.attention_blocks)

        # Final refiner layer that will transform the output to be compatible with the scene decoder
        # The temporal refiner will summarize the time axis (B, 1+N, T, D) -> (B, 1+N, D)
        self.temporal_refiner = nn.Sequential(nn.Linear(num_timesteps * hidden_size, hidden_size), nn.GELU())

        # The social refiner will summarize the agent axis (B, 1+N, D) -> (B, Q, D)
        self.social_refiner = nn.Sequential(nn.Linear((1 + num_agents) * hidden_size, hidden_size), nn.GELU())

        # The scenario decoder will produce Q scenario prototypes
        self.num_queries = num_queries
        self.scenario_decoder = nn.Linear(hidden_size, self.num_queries * hidden_size)

        self.apply(common.initialize_weights_with_xavier)

    def forward(
        self,
        agent_features: torch.Tensor,
        agent_masks: torch.Tensor,
        road_features: torch.Tensor,
        road_masks: torch.Tensor,
    ) -> ScenarioEmbedding:
        """Embeds the scenario features.
           B: batch size
           N: number of agents
           T: number of timesteps
           P: number of map inputs
           H: hidden size
           Q: number of scenario queries

        Args:
            agent_features (torch.tensor(B, 1+N, T, H)): tensor containing agent feature information.
            agent_masks (torch.tensor(B, 1+N, T)): tensor containg agent mask information.
            road_features (torch.tensor(B, P, H)): tensor containing map feature information.
            road_masks (torch.tensor(B, P)): tensor containg map mask information.

        Returns:
            ScenarioEmbedding: pydantic validator for the trajectory decoder with:
                scenario_enc (torch.tensor(B, 1+N, T, H)): encoded scenario using PerceiverIO
                scenario_dec (torch.tensor(B, Q, H)): decoded scenario using PerceiverIO
        """
        # Encode the scenario using factorized attention across time, agents and map
        scenario_enc = agent_features
        for block in self.attention_blocks:
            scenario_enc = block(scenario_enc, road_features, agent_masks, road_masks)

        # First, summarize the time axis
        batch_size, num_agents, _, _ = scenario_enc.shape
        # scenario_enc reshaped: (B, 1+N, T*H)
        # scenario_dec after temporal summary shape: (B, 1+N, H)
        scenario_dec = self.temporal_refiner(scenario_enc.view(batch_size, num_agents, -1))
        # scenario_dec reshaped: (B, (1+N) * H)
        # scenario_dec after social summary shape: (B, H)
        scenario_dec = self.social_refiner(scenario_dec.view(batch_size, -1))
        # scenario_dec after decoder shape: (B, Q, H)
        scenario_dec = self.scenario_decoder(scenario_dec).reshape(batch_size, self.num_queries, -1)
        return ScenarioEmbedding(scenario_enc=scenario_enc, scenario_dec=scenario_dec)
