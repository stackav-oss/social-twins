"""Code for the SceneTokens-Encoder model. A simplified encoder-only model that performs unsupervised
scenario tokenization on full trajectories (history + future) using VQ-VAE, without trajectory prediction.

This model is designed for scenarios where you want to tokenize/cluster based on complete trajectory
information rather than learning to predict futures from history.
"""

import torch
from omegaconf import DictConfig
from torch import nn

from scenetokens.models.base_model import BaseModel
from scenetokens.models.components import common
from scenetokens.schemas.output_schemas import (
    ModelOutput,
    ScenarioEmbedding,
    TokenizationOutput,
)


class SceneTokensEncoder(BaseModel):
    """SceneTokensEncoder class - encoder-only model without trajectory prediction.


    Class structure:
    - __init__: Copied from SceneTokensStudent, removed motion_decoder
    - forward: Copied from SceneTokensStudent, removed motion decoder call, uses gather_full_trajectory_input
    - gather_full_trajectory_input: New; combine history + future trajectories for full scenario encoding
    - embed: Copied from SceneTokensStudent (identical logic, different input length).
    """

    def __init__(self, config: DictConfig) -> None:
        """Initializes the SceneTokensEncoder class.

        Args:
            config (DictConfig): Configuration for the model.

        Copied from SceneTokensStudent.
        Removed: self.motion_decoder (no trajectory prediction needed)
        Modified: temporal_positional_embedding uses total_len (past + future) for full trajectory support
        """
        super().__init__(config=config)
        # Agent and Road Encoders
        self.road_encoder = nn.Sequential(nn.Linear(self.config.map_input_size, self.config.hidden_size))
        self.agent_encoder = nn.Sequential(nn.Linear(self.config.agents_input_size, self.config.hidden_size))

        # Total sequence length = past + future (or just past if future_len == 0)
        self.total_len = self.past_len + self.future_len

        self.agents_positional_embedding = nn.parameter.Parameter(
            torch.zeros((1, 1, (self.config.max_num_agents + 1), self.config.hidden_size)),
            requires_grad=True,
        )
        self.temporal_positional_embedding = nn.parameter.Parameter(
            torch.zeros((1, self.total_len, 1, self.config.hidden_size)),
            requires_grad=True,
        )

        # Scenario Encoder
        self.scenario_embedder = self.config.scenario_embedder

        # Scenario Tokenizer (VQ-VAE)
        self.scenario_tokenizer = self.config.tokenizer

        self.criterion = self.config.criterion
        self.selu = nn.SELU(inplace=True)

        self.apply(common.initialize_weights_with_xavier)  # pyright: ignore[reportArgumentType]
        self.print_and_get_num_params()

    def forward(self, batch: dict) -> ModelOutput:
        """Model's forward pass.
            B: batch size
            N: max number agents in the scene
            T: total trajectory length (history + future)
            P: number of map polylines
            D: number of agent features + types
            R: number of road features + types
            Qe: number of encoded queries
            Qd: number of decoded queries
            E: hidden size

        Args:
            batch (dict): dictionary containing the following input data batch:
                batch_size (int)
                input_dict (dict): dictionary containing the scenario information

        Returns:
            model_output (ModelOutput): model schema encapsulating all model outputs.

        Copied from SceneTokensStudent.
        Removed: motion_decoder call and token conditioning logic
        Modified: uses gather_full_trajectory_input instead of BaseModel.gather_input
        """
        inputs = batch["input_dict"]

        # Get history trajectories (for ground truth reference)
        history_gt_trajs = inputs["obj_trajs"]
        history_gt_trajs_mask = inputs["obj_trajs_mask"].unsqueeze(-1)
        history_ground_truth = torch.cat([history_gt_trajs, history_gt_trajs_mask], dim=-1)

        # Get future trajectories (for ground truth reference - even though we don't predict)
        center_gt_trajs = inputs["center_gt_trajs"][..., :2]
        center_gt_trajs_mask = inputs["center_gt_trajs_mask"].unsqueeze(-1)
        future_ground_truth = torch.cat([center_gt_trajs, center_gt_trajs_mask], dim=-1)

        # Gather full trajectory input (history + future combined)
        ego_agent, other_agents, roads = self.gather_full_trajectory_input(inputs)

        # Encode and embed the full scenario
        scenario_embedding: ScenarioEmbedding = self.embed(ego_agent, other_agents, roads)

        # Get context for tokenization
        context = scenario_embedding.scenario_dec.value

        # Classify/tokenize the scenario using VQ-VAE
        tokenized_scenario: TokenizationOutput = self.scenario_tokenizer(context)

        return ModelOutput(
            scenario_embedding=scenario_embedding,
            trajectory_decoder_output=None,  # No trajectory prediction
            tokenization_output=tokenized_scenario,
            history_ground_truth=history_ground_truth,  # pyright:ignore[reportArgumentType]
            future_ground_truth=future_ground_truth,  # pyright:ignore[reportArgumentType]
            dataset_name=inputs["dataset_name"],
            scenario_id=inputs["scenario_id"],
            agent_ids=inputs["obj_ids"].squeeze(-1).squeeze(-1),
        )

    def gather_full_trajectory_input(self, inputs: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gathers full trajectory (history + future) for all agents.

        If future_len > 0, combines history and future trajectories.
        If future_len == 0, uses only history (assumes data was prepared with full trajectory as history).

        Args:
            inputs (dict): dictionary containing scenario data.

        Returns:
            ego_in (torch.Tensor): ego agent full trajectory (B, T, D+1)
            agents_in (torch.Tensor): other agents full trajectories (B, T, N, D+1)
            roads (torch.Tensor): road information (B, P, M, D+1)

        This is a new method compared to SceneTokensStudent.
        Purpose: Combines history (obj_trajs) and future (obj_trajs_future_state) trajectories
        into a single tensor for full-scenario encoding. This enables clustering on complete
        trajectory information rather than just history.

        Key differences from BaseModel.gather_input:
        - BaseModel.gather_input returns only history trajectories (obj_trajs)
        - This method concatenates history + future along the time dimension
        - Future state (x,y,vx,vy) is zero-padded to match history feature dimensions
        """
        # History trajectories: (B, N, H, D)
        history_trajs = inputs["obj_trajs"]
        history_mask = inputs["obj_trajs_mask"]

        batch_size, num_agents, _, history_dim = history_trajs.shape

        # Check if we have future data to combine
        if self.future_len > 0 and "obj_trajs_future_state" in inputs:
            # Future trajectories: (B, N, F, 4) where 4 = (x, y, vx, vy)
            future_trajs = inputs["obj_trajs_future_state"]
            future_mask = inputs["obj_trajs_future_mask"]
            future_len = future_trajs.shape[2]

            # Pad future to match history dimensions
            # Future only has (x, y, vx, vy), need to pad to match history_dim
            future_trajs_padded = torch.zeros(
                (batch_size, num_agents, future_len, history_dim),
                device=future_trajs.device,
                dtype=future_trajs.dtype,
            )
            # History feature layout (29 features total):
            #   [0:3]   = (x, y, z) position
            #   [3:6]   = (length, width, height) dimensions
            #   [6:11]  = one-hot agent type (5 classes)
            #   [11:23] = time embeddings
            #   [23:25] = heading (cos, sin)
            #   [25:27] = (vx, vy) velocity
            #   [27:29] = (ax, ay) acceleration
            #
            # Future state layout (4 features): [x, y, vx, vy]
            # Map future features to corresponding history slots:
            pos_x, pos_y = 0, 1
            vel_x, vel_y = 25, 26
            future_trajs_padded[..., pos_x] = future_trajs[..., 0]  # x position
            future_trajs_padded[..., pos_y] = future_trajs[..., 1]  # y position
            future_trajs_padded[..., vel_x] = future_trajs[..., 2]  # vx velocity
            future_trajs_padded[..., vel_y] = future_trajs[..., 3]  # vy velocity

            # Concatenate history and future: (B, N, T, D) where T = H + F
            full_trajs = torch.cat([history_trajs, future_trajs_padded], dim=2)
            full_mask = torch.cat([history_mask, future_mask], dim=2)
        else:
            # No future data - use history only (full trajectory already in history)
            full_trajs = history_trajs
            full_mask = history_mask

        # Get ego agent index
        index_to_predict = inputs["track_index_to_predict"].view(-1, 1, 1, 1).repeat(1, 1, *full_trajs.shape[-2:])

        # Extract ego trajectory: (B, T, D)
        ego_in = torch.gather(full_trajs, 1, index_to_predict).squeeze(1)

        # Extract ego mask: (B, T)
        index_to_predict_mask = inputs["track_index_to_predict"].view(-1, 1, 1).repeat(1, 1, full_mask.shape[-1])
        ego_mask = torch.gather(full_mask, 1, index_to_predict_mask).squeeze(1)

        # Combine trajectories with masks: (B, N, T, D+1) -> (B, T, N, D+1)
        agents_in = torch.cat([full_trajs, full_mask.unsqueeze(-1)], dim=-1)
        agents_in = agents_in.transpose(1, 2)

        # Combine ego with mask: (B, T, D+1)
        ego_in = torch.cat([ego_in, ego_mask.unsqueeze(-1)], dim=-1)

        # Road information (unchanged from base)
        roads = inputs["map_polylines"]
        roads_mask = inputs["map_polylines_mask"].unsqueeze(-1)
        roads = torch.cat([roads, roads_mask], dim=-1)

        return ego_in, agents_in, roads

    def embed(self, ego_agent: torch.Tensor, other_agents: torch.Tensor, roads: torch.Tensor) -> ScenarioEmbedding:
        """Encodes full scenario context (history + future trajectories).
            B: batch size
            N: max number agents in the scene
            T: total trajectory length (history + future)
            P: number of map polylines
            D: number of agent features + types
            M: number of map features + types

        Args:
            ego_agent (torch.tensor(B, T, D+1)): tensor containing ego-agent features (D) and mask (1) information.
            other_agents (torch.tensor(B, T, N, D+1)): tensor containing other agents features and mask information.
            roads (torch.tensor(B, P, M, D+1)): tensor containing map information.

        Returns:
            ScenarioEmbedding: a pydantic validator with
                scenario_enc (torch.tensor(B, Q, H)): tensor containing the embedded scene.
                scenario_dec (torch.tensor(B, Q, H)): tensor containing the refined embeddings of the scene.

        Copied from SceneTokensStudent
        Identical logic, just operates on longer sequences (T = history + future)
        """
        # Ego information
        #   ego tensor shape: (B, T, D+1)
        #   ego mask shape: (B, T)
        ego_tensor = ego_agent[:, :, : self.config.agents_input_size]
        ego_masks = ego_agent[:, :, -1]

        # Other agents information
        #   agent tensor shape: (B, T, N, D+1)
        #   agent masks shape: (B, T, N)
        other_agents_tensor = other_agents[:, :, :, : self.config.agents_input_size]
        other_agents_masks = other_agents[:, :, :, -1]

        # Combined agents information
        #   agents tensor shape: (B, T, 1+N, D)
        #   agents mask shape: (B, T, 1+N)
        agent_masks = torch.cat((torch.ones_like(ego_masks.unsqueeze(-1)), other_agents_masks), dim=-1)
        agent_masks_inv = (1.0 - agent_masks).to(torch.bool)
        agents_tensor = torch.cat((ego_tensor.unsqueeze(2), other_agents_tensor), dim=2)

        batch_size, total_len, num_agents, _ = agents_tensor.shape

        # Encode agent information
        #   agents_emb shape: (B, T, 1+N, hidden_size)
        agents_emb = self.agent_encoder(agents_tensor)
        agents_emb = self.selu(agents_emb)

        # Add positional embeddings
        #   agents_posemb shape: (1, 1, 1+N, hidden_size)
        #   temporal_posembd shape: (1, T, 1, hidden_size)
        pos_emb = (
            self.agents_positional_embedding[:, :, :num_agents] + self.temporal_positional_embedding[:, :total_len]
        )
        agents_emb = (agents_emb + pos_emb).view(batch_size, -1, self.config.hidden_size)

        # Encode map information
        #   roads tensor shape: (B, P, M, D)
        #   roads mask shape: (B, P, M)
        #   roads emb shape: (B, P * M, hidden_size)
        roads_tensor = roads[:, : self.max_num_roads, :, : self.config.map_input_size]
        roads_mask = roads[:, : self.max_num_roads, :, -1]
        roads_inv = (1.0 - roads_mask).to(torch.bool)
        road_emb = self.road_encoder(roads_tensor).view(batch_size, -1, self.config.hidden_size)
        road_emb = self.selu(road_emb)

        # Process mixed information using the perceiver encoder
        #   mixed_input_features shape: (B, T * (1+N) + P * M, hidden_size)
        #   mixed_input_masks shape: (B, T * (1+N) + P * M)
        mixed_input_features = torch.concat([agents_emb, road_emb], dim=1)
        mixed_input_masks = torch.concat([agent_masks_inv.view(batch_size, -1), roads_inv.view(batch_size, -1)], dim=1)

        return self.scenario_embedder(mixed_input_features, mixed_input_masks)
