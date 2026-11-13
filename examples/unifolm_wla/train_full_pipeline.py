"""Base post-training with our data contract and recoverable online runs."""
import gc
import argparse
import datetime
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

REINITIALIZED = {"action_model.action_encoder.layer1.weight",
                 "action_model.action_decoder.layer2.weight",
                 "action_model.action_decoder.layer2.bias"}


def load_base(model, path):
    from safetensors.torch import load_file
    weights = load_file(path)
    target = model.state_dict()
    assert set(weights) == set(target), {"missing": sorted(set(target) - set(weights)),
                                       "unexpected": sorted(set(weights) - set(target))}
    mismatches = {k for k in weights if weights[k].shape != target[k].shape}
    assert mismatches == REINITIALIZED, mismatches
    report = {k: {"pretrained": list(weights[k].shape), "new": list(target[k].shape)}
              for k in sorted(REINITIALIZED)}
    for k in REINITIALIZED:
        del weights[k]
    result = model.load_state_dict(weights, strict=False)
    assert set(result.missing_keys) == REINITIALIZED and not result.unexpected_keys
    return {"loaded_tensors": len(weights), "reinitialized": report}


def save_checkpoint(accelerator, model, out, step, epoch, batches_in_epoch):
    from safetensors.torch import save_file
    folder = out / "checkpoints" / f"step_{step:06d}"
    # All ranks participate: this saves ZeRO optimizer partitions and RNG state.
    accelerator.save_state(str(folder), safe_serialization=True)
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        save_file({k: v.detach().cpu().contiguous() for k, v in state_dict.items()}, str(folder / "model.safetensors"))
        (folder / "progress.json").write_text(json.dumps({"step": step, "epoch": epoch,
                                                         "batches_in_epoch": batches_in_epoch}))
        (folder / "COMPLETE").touch()
        # Only remove optimizer partitions after the new complete checkpoint exists.
        for old in sorted((out / "checkpoints").glob("step_*")):
            if old == folder:
                continue
            for optim in old.rglob("*optim_states.pt"):
                optim.unlink()
    accelerator.wait_for_everyone()


def main():
    # Spawned data workers import this module; keep model/DeepSpeed initialization
    # inside the training process so CPU video workers never create CUDA contexts.
    from examples.unifolm_wla.pytree_cycle_fix import install_cycle_free_pytree
    install_cycle_free_pytree()
    import wandb
    from accelerate import Accelerator, DataLoaderConfiguration, DeepSpeedPlugin
    from accelerate.utils import set_seed
    from omegaconf import OmegaConf
    from transformers import get_scheduler
    from unifolm_wla.dataloader.full_pipeline import build_full_pipeline_dataloader, resolve_training_data
    from unifolm_wla.model.framework.base_framework import build_framework

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-size", type=int, help="Override the configured per-device batch size")
    parser.add_argument("--output", help="Override the configured output directory")
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--random-init", action="store_true", help="Explicit random-weight capacity test; requires --smoke-steps")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--disable-checkpointing", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--validate-smoke", action="store_true", help="Run full validation at the final smoke step")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    manifest, _ = resolve_training_data(cfg.datasets.vla_data)
    if not manifest.is_file():
        raise FileNotFoundError(f"Training manifest does not exist: {manifest}")
    args.batch_size = (cfg.datasets.vla_data.per_device_batch_size
                       if args.batch_size is None else args.batch_size)
    args.output = args.output or cfg.output_dir
    cfg.datasets.vla_data.per_device_batch_size = args.batch_size
    cfg.output_dir = args.output
    cfg.trainer.is_resume = bool(args.resume)
    assert not args.random_init
    cfg.trainer.initialization = "pretrained"
    cfg.trainer.gradient_checkpointing = not args.disable_checkpointing
    assert not args.profile or args.smoke_steps >= 8
    assert not args.random_init or (args.smoke_steps > 0 and not args.resume)
    assert cfg.trainer.gradient_accumulation_steps == 1
    assert cfg.trainer.train_robot_state_projector
    assert cfg.trainer.save_format == "safetensors"
    assert 0 < cfg.trainer.max_train_steps <= 50000
    plugin = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=1,
                            gradient_clipping=cfg.trainer.gradient_clipping)
    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=1,
                              deepspeed_plugin=plugin,
                              dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True, data_seed=cfg.seed, non_blocking=True))
    cfg.trainer.world_size = accelerator.num_processes
    cfg.trainer.global_batch = args.batch_size * accelerator.num_processes
    set_seed(cfg.seed)
    out = Path(args.output)
    if accelerator.is_main_process:
        out.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, out / "config.yaml")
    accelerator.wait_for_everyone()
    model = build_framework(cfg)
    loaded = load_base(model, cfg.trainer.pretrained_checkpoint)
    model.qwen_vl_interface.requires_grad_(not cfg.trainer.freeze_vlm)
    model.qwen_vl_interface.model.robot_state_projector.requires_grad_(True)
    if cfg.trainer.gradient_checkpointing:
        model.qwen_vl_interface.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.qwen_vl_interface.model.config.use_cache = False
    if cfg.trainer.gradient_checkpointing:
        model.action_model.model.enable_gradient_checkpointing()
    vlm = model.qwen_vl_interface.model.model
    loaded["gradient_checkpointing"] = {
        "vision_blocks": sum(layer.gradient_checkpointing for layer in vlm.visual.blocks),
        "language_layers": sum(layer.gradient_checkpointing for layer in vlm.language_model.layers),
        "action_expert": bool(model.action_model.model.gradient_checkpointing)}
    assert loaded["gradient_checkpointing"] == {"vision_blocks": 24, "language_layers": 36, "action_expert": True}
    from examples.unifolm_wla.training_fusion import apply_candidate
    from examples.unifolm_wla.training_worker_inputs import use_worker_inputs, CPUInputsCollator
    apply_candidate(model, "blocks")
    use_worker_inputs(model)
    if args.profile and accelerator.is_main_process:
        def instrument(module, name):
            def enter(m, inputs):
                m._profile_scope = torch.profiler.record_function(name)
                m._profile_scope.__enter__()
            def leave(m, inputs, output):
                m._profile_scope.__exit__(None, None, None)
            module.register_forward_pre_hook(enter)
            module.register_forward_hook(leave)
        instrument(model.qwen_vl_interface, "VLM_forward")
        instrument(model.action_model, "MMDiT_forward")
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if cfg.trainer.freeze_vlm:
        assert all(n.startswith("action_model.") or "robot_state_projector" in n for n, p in trainable)
    else:
        assert all(p.requires_grad for p in model.parameters())
    if accelerator.is_main_process:
        (out / "trainable-parameters.json").write_text(json.dumps({
            "freeze_vlm": cfg.trainer.freeze_vlm,
            "total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for _, p in trainable),
            "vlm_trainable": sum(p.numel() for n, p in trainable if n.startswith("qwen_vl_interface.")),
        }, indent=2))
    optimizer = torch.optim.AdamW([p for n, p in trainable], lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas), eps=cfg.trainer.optimizer.eps,
        weight_decay=cfg.trainer.optimizer.weight_decay, fused=True)
    scheduler = get_scheduler(cfg.trainer.lr_scheduler_type, optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps, num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=OmegaConf.to_container(cfg.trainer.scheduler_specific_kwargs))
    loader = build_full_pipeline_dataloader(cfg)
    loader.collate_fn = CPUInputsCollator(cfg)
    loader.pin_memory = True
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    accelerator.register_for_checkpointing(scheduler)
    set_seed(cfg.seed + accelerator.process_index)
    if accelerator.is_main_process:
        (out / "weight-loading.json").write_text(json.dumps(loaded, indent=2))
        (out / "dataset_statistics.json").write_text(Path(cfg.datasets.vla_data.statistics_path).read_text())
    step = epoch = batches_in_epoch = 0
    if args.resume:
        resume = Path(args.resume)
        assert (resume / "COMPLETE").exists()
        progress = json.loads((resume / "progress.json").read_text())
        accelerator.load_state(str(resume))
        step, epoch, batches_in_epoch = progress["step"], progress["epoch"], progress["batches_in_epoch"]
    run = None
    if accelerator.is_main_process and not args.smoke_steps:
        assert os.environ.get("WANDB_MODE", "online") == "online"
        run = wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project,
                         name=out.name, mode="online", dir=str(out),
                         config=OmegaConf.to_container(cfg, resolve=True))
        (out / "wandb-run.json").write_text(json.dumps({"url": run.url, "id": run.id}))
    if run:
        run.define_metric("*", step_metric="step")
        run.define_metric("validation/*", step_metric="validation/step")
    validation = None
    if cfg.validation.enabled and (not args.smoke_steps or args.validate_smoke):
        from examples.unifolm_wla.training_validation import TrainingValidation
        validation = TrainingValidation(cfg, accelerator, out)
        validation.initialize_runtime(model)
        accelerator.wait_for_everyone()

    from examples.unifolm_wla.checkpoint_retention import prune_checkpoints
    def log_validation(results):
        for result in results:
            print(json.dumps(result), flush=True)
            with (validation.out / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(result) + "\n")
            if run:
                run.log(result)
            prune_checkpoints(out)

    max_steps = args.smoke_steps or cfg.trainer.max_train_steps
    records = []
    prompt_counts = np.zeros(4, dtype=np.int64)
    model.train()
    if validation and args.resume and cfg.validation.validate_on_resume:
        validation.infer(model, step)
    torch.cuda.reset_peak_memory_stats()
    prof = None
    if args.profile and accelerator.is_main_process:
        prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=2, warmup=1, active=3, repeat=1), record_shapes=True)
        prof.start()
    while step < max_steps:
        loader.set_epoch(epoch)
        current_loader = accelerator.skip_first_batches(loader, batches_in_epoch) if batches_in_epoch else loader
        last_end = time.perf_counter()
        for batch in current_loader:
            if validation:
                log_validation(validation.poll())
            started = time.perf_counter()
            data_seconds = started - last_end
            optimizer.zero_grad(set_to_none=True)
            with torch.profiler.record_function("training_forward"), accelerator.autocast():
                loss = model(batch)["action_loss"]
            assert torch.isfinite(loss), "nonfinite loss"
            with torch.profiler.record_function("backward_and_zero_optimizer"):
                accelerator.backward(loss)
            grad_norm = model.get_global_grad_norm()
            optimizer.step()
            scheduler.step()
            torch.cuda.synchronize()
            step += 1
            batches_in_epoch += 1
            loss_mean = accelerator.reduce(loss.detach().float(), reduction="mean").item()
            model_seconds = time.perf_counter() - started
            local_timing = torch.tensor([data_seconds, model_seconds, data_seconds + model_seconds], device=accelerator.device)
            all_timing = accelerator.gather(local_timing[None])
            maximum_timing = all_timing.max(dim=0).values.cpu().tolist()
            memory_stats = torch.cuda.memory_stats_as_nested_dict()
            record = {"step": step, "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      "loss": loss_mean, "grad_norm": float(grad_norm),
                      "learning_rate": scheduler.get_last_lr()[0], "step_seconds": maximum_timing[2],
                      "data_seconds": maximum_timing[0], "model_seconds": maximum_timing[1],
                      "samples_per_second": args.batch_size * accelerator.num_processes / maximum_timing[2],
                      "batch_per_device": args.batch_size, "world_size": accelerator.num_processes,
                      "global_batch": args.batch_size * accelerator.num_processes,
                      "peak_allocated_bytes": memory_stats["allocated_bytes"]["all"]["peak"],
                      "peak_reserved_bytes": memory_stats["reserved_bytes"]["all"]["peak"]}
            assert np.isfinite(record["grad_norm"]), record
            if cfg.datasets.vla_data.get('history_frames', 0):
                names = ['full', 'scene', 'caption', 'caption_missing_scene']
                for example in batch['examples']:
                    prompt_counts[names.index(example['prompt_source'])] += 1
                if step == 1 or step % cfg.trainer.logging_frequency == 0:
                    counts = accelerator.reduce(torch.tensor(prompt_counts, device=accelerator.device), reduction='sum').cpu().tolist()
                    record.update({f'prompt/{name}_fraction':count/sum(counts) for name,count in zip(names,counts)})
            records.append(record)
            if accelerator.is_main_process:
                print(json.dumps(record), flush=True)
                with (out / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps(record) + "\n")
                if run and (step == 1 or step % cfg.trainer.logging_frequency == 0):
                    run.log(record)
            if not args.smoke_steps and (step % cfg.trainer.save_interval == 0 or step == max_steps):
                save_checkpoint(accelerator, model, out, step, epoch, batches_in_epoch)
            if validation and (step % cfg.validation.interval == 0 or step == max_steps):
                validation.infer(model, step)
            if step == 3 or step % cfg.trainer.save_interval == 0 or step == max_steps:
                gc.unfreeze()
                gc.collect()
                gc.freeze()
            last_end = time.perf_counter()
            if prof:
                prof.step()
            if step >= max_steps:
                break
        else:
            epoch += 1
            batches_in_epoch = 0
    if prof:
        prof.stop()
        prof.export_chrome_trace(str(out / "trace.json"))
        events = prof.key_averages()
        (out / "profile-table.txt").write_text(events.table(sort_by="self_cuda_time_total", row_limit=50))
        names = {"VLM_forward", "MMDiT_forward", "training_forward", "backward_and_zero_optimizer"}
        summary = [{"name": e.key, "count": e.count, "cpu_total_us": e.cpu_time_total,
                    "device_total_us": e.device_time_total} for e in events if e.key in names]
        (out / "profile-scopes.json").write_text(json.dumps(summary, indent=2))
    if args.smoke_steps:
        bare = accelerator.unwrap_model(model)
        bare.eval()
        predicted = bare.predict_action(batch)["normalized_actions"]
        assert predicted.shape == (args.batch_size, 50, 114) and np.isfinite(predicted).all()
        report = {"status": "PASS", "rank": accelerator.process_index, "steps": step,
                  "prediction_shape": list(predicted.shape), "records": records}
        (out / f"smoke-rank-{accelerator.process_index:02d}.json").write_text(json.dumps(report, indent=2))
    if validation:
        log_validation(validation.close())
    if run:
        run.finish()
    accelerator.wait_for_everyone()
    gc.unfreeze()
    gc.collect()


if __name__ == "__main__":
    main()
