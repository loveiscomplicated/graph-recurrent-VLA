import torch


def action_metrics(prediction, target):
    error = (prediction-target)**2
    clipped = prediction.clamp(-1, 1)
    def controller_rotation(action):
        r = action[:, 3:6]
        return -.1*r/r.norm(dim=-1, keepdim=True).clamp_min(1)
    return {"action_mse": float(error.mean()),
            "translation_mse": float(error[:, :3].mean()),
            "translation_rmse": float(error[:, :3].mean().sqrt()),
            "rotation_mse": float(error[:, 3:6].mean()),
            "rotation_rmse": float(error[:, 3:6].mean().sqrt()),
            "gripper_mse": float(error[:, 6].mean()),
            "gripper_mae": float((prediction[:, 6]-target[:, 6]).abs().mean()),
            "bounded_action_mse": float(((clipped-target)**2).mean()),
            "translation_command_rmse_m": float(((.1*clipped[:, :3]-.1*target[:, :3])**2).mean().sqrt()),
            "rotation_command_euler_rmse_rad": float(((controller_rotation(clipped)-controller_rotation(target))**2).mean().sqrt())}
