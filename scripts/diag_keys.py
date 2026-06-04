import torch
from pathlib import Path
ck = torch.load(str(Path(__file__).resolve().parent.parent / "checkpoints" / "path_a1_pa_code" / "path_a1_pa_code_warmstart.pt"), map_location="cpu", weights_only=False)
state = ck.get("model_state_dict", ck.get("state_dict", ck))
matches = [k for k in state.keys() if "film" in k.lower() or "gamma_head" in k.lower() or "beta_head" in k.lower() or "organ_emb" in k.lower() or k.startswith("ode.") or k.startswith("ode_")]
for k in matches:
    print(f"  {k:60s}  {tuple(state[k].shape)}")
