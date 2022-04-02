import json
import os
import torch
import torch.nn as nn


class NeuralTopicModel(nn.Module):
    
    CONFIG_FILE = "config.json"
    MODEL_STATE_FILE = "model_state.pt"

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)

        # Save configuration
        config_file = os.path.join(save_directory, self.CONFIG_FILE)
        def _to_json_string(config):
            return json.dumps(config, indent=2, sort_keys=True) + "\n"
        with open(config_file, "w", encoding="utf-8") as writer:
            writer.write(_to_json_string(self.config))

        # Save model state dict
        model_state_file = os.path.join(save_directory, self.MODEL_STATE_FILE)
        torch.save(self.state_dict(), model_state_file)
    
    @classmethod
    def from_pretrained(cls, pretrained_directory):
        # Load configuration
        config_file = os.path.join(pretrained_directory, cls.CONFIG_FILE)
        with open(config_file, "r") as reader:
            config = json.load(reader)
        
        # Load model from state dict
        ntm = cls(**config)
        model_state_file = os.path.join(pretrained_directory, cls.MODEL_STATE_FILE)
        state_dict = torch.load(model_state_file)
        ntm.load_state_dict(state_dict)

        return ntm
    
    def load_weights_from_pretrained(self, pretrained_directory):
        model_state_file = os.path.join(pretrained_directory, self.MODEL_STATE_FILE)
        state_dict = torch.load(model_state_file)
        self.load_state_dict(state_dict)
    
    @property
    def device(self):
        return next(self.parameters()).device
