"""Execute the actual fused Mamba and Transformer kernels before parallel training."""

import argparse
import json
from pathlib import Path

import torch
import yaml

from graph.temporal.fixed_id_corner import FixedIdCornerPredictor
from training.fixed_id_corners import input_statistics, load_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    torch.set_num_threads(2)
    device = torch.device(config["training"]["device"])
    splits, manifest = load_splits(
        Path(config["data"]["root"]), config["data"]["expected_dataset_sha256"], device,
    )
    mean, std = input_statistics(splits["train"])
    results = {}
    common_weights = None
    for architecture in ("mamba", "transformer"):
        torch.manual_seed(0)
        model = FixedIdCornerPredictor(**dict(config["model"], architecture=architecture)).to(device)
        common = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                  if name.startswith(("projection.", "output_norm.", "delta_head."))}
        if common_weights is None:
            common_weights = common
        else:
            for name in common:
                torch.testing.assert_close(common[name], common_weights[name], rtol=0, atol=0)
        model.input_mean.copy_(mean)
        model.input_std.copy_(std)
        # Test nonzero outputs; an initialized zero head cannot prove causality.
        torch.nn.init.normal_(model.delta_head.weight, std=0.02)
        inputs = torch.randn(2, 12, 13, device=device)
        context = torch.ones(2, 12, dtype=torch.bool, device=device)
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            expected = model(inputs, context)
            altered = inputs.clone()
            altered[:, 6:] += 10
            actual = model(altered, context)
            torch.testing.assert_close(actual[:, :6], expected[:, :6], atol=2e-5, rtol=0.01)
            short = model(inputs[:, :6], context[:, :6])
            torch.testing.assert_close(short, expected[:, :6], atol=2e-5, rtol=0.01)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, fused=True)
        batch = {name: value[:config["training"]["batch_size"]] for name, value in splits["train"].items()}
        assert int((batch["supervised"] & batch["observed"]).sum()) > 0
        assert int((batch["supervised"] & ~batch["observed"]).sum()) > 0
        torch.cuda.reset_peak_memory_stats(device)
        losses = []
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(batch["inputs"], batch["context"])
                squared = ((prediction - batch["targets"]) / model.displacement_scale_m).square().sum(-1)
                loss = (squared * batch["supervised"]).sum() / (3 * batch["supervised"].sum())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss))
        results[architecture] = dict(
            losses=losses, peak_gpu_mb=torch.cuda.max_memory_allocated(device) / 2 ** 20,
            parameters=sum(parameter.numel() for parameter in model.parameters()),
            causal_prefix_and_padding_passed=True, finite_backward_passed=True,
        )
        print(architecture, json.dumps(results[architecture]), flush=True)
        del model, optimizer, batch
        torch.cuda.empty_cache()
    print(json.dumps(dict(dataset_sha256=manifest["dataset_sha256"], results=results), indent=2))


if __name__ == "__main__":
    main()
