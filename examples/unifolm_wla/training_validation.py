"""Batched GPU validation followed by CPU-only asynchronous artifact processing."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
import random
import time

import numpy as np
import torch

from examples.unifolm_wla.open_loop_eval import (canonical, metrics, plot_episode, prepare_episode,
                            read_json, rotation_errors, write_json)


@contextmanager
def validation_state(model, device):
    modes = [(module, module.training) for module in model.modules()]
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[device]):
            model.eval()
            with torch.inference_mode():
                yield
    finally:
        for module, training in modes:
            module.training = training
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)


def native_actions(canonical_actions):
    body = canonical_actions[..., 3:102].reshape(*canonical_actions.shape[:-1], 11, 9)
    return np.concatenate((canonical_actions[..., :3], body[..., :3].reshape(*body.shape[:-2], 33),
                           body[..., 3:].reshape(*body.shape[:-2], 66), canonical_actions[..., 102:]), axis=-1)


class TrainingValidation:
    def __init__(self, cfg, accelerator, out):
        from unifolm_wla.dataloader.full_pipeline import CAMERAS, pack_state, pad_resize, read_frame
        self.accelerator, self.cfg = accelerator, cfg
        assert int(cfg.datasets.vla_data.history_frames) == 50
        self.out = Path(out) / cfg.validation.output_subdir
        self.out.mkdir(exist_ok=True)
        self.batch_size = int(cfg.validation.batch_size)
        assert self.batch_size > 0
        self.dataset, self.reference = Path(cfg.validation.dataset), Path(cfg.validation.reference)
        self.meta = read_json(self.reference / "manifest.json")
        stats = read_json(cfg.datasets.vla_data.statistics_path)["full_pipeline"]
        so, ss = (np.asarray(stats["state"][k], dtype=np.float32) for k in ("offset", "scale"))
        self.ao, self.asc = (np.asarray(stats["action"][k], dtype=np.float32) for k in ("offset", "scale"))
        self.prompt = read_json(cfg.datasets.vla_data.manifest_path)["prompt"]
        self.episodes = list(cfg.validation.episodes)
        from examples.unifolm_wla.training_worker_inputs import CPUInputsCollator
        collator = CPUInputsCollator(cfg)
        self.prepared, self.examples, self.batches = {}, {}, {}
        self.inference_batches = {}
        self.runtime = None
        # Cache this rank's inputs before training, so validation does no video IO.
        ranks_per_node = int(cfg.trainer.gpus_per_node)
        node_count = accelerator.num_processes // ranks_per_node
        assert node_count > 0 and accelerator.num_processes % ranks_per_node == 0
        for index, eid in enumerate(self.episodes):
            owner = index % accelerator.num_processes
            if owner != accelerator.process_index:
                continue
            entry = prepare_episode(self.dataset, self.reference, eid)
            self.prepared[eid] = entry
            targets = native_actions(entry["targets"]).reshape(-1, 50, 114)
            examples = []
            for k, anchor in enumerate(entry["anchors"]):
                state, mask = pack_state((entry["states"][anchor] - so) / ss)
                images = [pad_resize(read_frame(str(self.dataset / f"videos/chunk-000/{camera}/episode_{eid:06d}.mp4"), int(anchor)))
                          for camera, _ in CAMERAS]
                examples.append(dict(image=images, image_roles=[role for _, role in CAMERAS], lang=self.prompt,
                    state=state, state_mask=mask, action=(targets[k] - self.ao) / self.asc,
                    action_mask=np.ones(114, dtype=bool), arm_type="dual_with_legs", robot_type="unitree"))
            from unifolm_wla.dataloader.state_history import history_rows, pack_history
            raw_history, history_valid = [], []
            for k, anchor in enumerate(entry['anchors']):
                segment = next(s for s in entry['segments'] if s['frame_start'] <= anchor < s['frame_stop_exclusive'])
                rows, valid = history_rows(int(anchor), segment['frame_start'])
                raw = entry['states'][rows]
                examples[k]['state_history'], examples[k]['state_history_mask'] = pack_history(raw, valid, so, ss)
                raw_history.append(np.concatenate((raw, entry['states'][anchor][None]), axis=0))
                history_valid.append(np.concatenate((valid, [True])))
            raw_history, history_valid = np.asarray(raw_history), np.asarray(history_valid)
            self.examples[eid] = examples
            self.batches[eid] = [collator(examples[start:start+self.batch_size])
                                 for start in range(0, len(examples), self.batch_size)]
            self.inference_batches[eid] = []
            for start in range(0, len(examples), self.batch_size):
                selected = examples[start:start+self.batch_size]
                images = np.stack([np.stack([np.asarray(im) for im in item['image']]) for item in selected])
                states = raw_history[start:start+self.batch_size]
                valid = history_valid[start:start+self.batch_size]
                padding = self.batch_size - len(selected)
                self.inference_batches[eid].append((
                    np.concatenate((images, np.repeat(images[-1:], padding, axis=0))),
                    np.concatenate((states, np.repeat(states[-1:], padding, axis=0))),
                    np.concatenate((valid, np.repeat(valid[-1:], padding, axis=0)))))
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="validation-cpu")
        self.pending = []
        self.cameras = CAMERAS

    def initialize_runtime(self, model):
        if not self.examples:
            return
        from examples.unifolm_wla.optimized_inference.runtime import Runtime
        bare = self.accelerator.unwrap_model(model)
        started = time.perf_counter()
        with validation_state(bare, self.accelerator.device):
            self.runtime = Runtime(bare, self.cfg, batch_size=self.batch_size)
            self.runtime.prepare(self.prompt)
            images, states, valid = next(iter(self.inference_batches.values()))[0]
            self.runtime.images.copy_(torch.as_tensor(images.reshape(-1,320,448,3)))
            self.runtime.state.copy_(torch.as_tensor(states))
            self.runtime.history_valid.copy_(torch.as_tensor(valid))
            self.runtime.capture()
            before = self.runtime.infer(images, states, valid, seed=int(self.cfg.validation.seed))
            # Verify that captured graphs read refreshed weights and derived caches.
            core = self.runtime.core
            core.head.action_decoder.layer2.bias.add_(1)
            core.backbone.language_model.layers[0].self_attn.qkv.weight.zero_()
            core.vision_positions.zero_()
            core.text_embeddings.zero_()
            core.head.model.output_mod_cache.zero_()
            for block in core.head.model.transformer_blocks:
                block.action_mod_cache.zero_()
                block.txt_mod_cache.zero_()
            changed = self.runtime.infer(images, states, valid, seed=int(self.cfg.validation.seed))
            assert not np.array_equal(before, changed), "Captured inference ignored changed weights/caches"
            self.runtime.refresh(bare)
            after = self.runtime.infer(images, states, valid, seed=int(self.cfg.validation.seed))
            np.testing.assert_array_equal(before, after)
        write_json(self.out / f"runtime-rank-{self.accelerator.process_index}.json",
                   dict(setup_seconds=time.perf_counter()-started, batch=self.batch_size,
                        backend="optimized CUDA Graph", refresh_graph_check="PASS", initialized_from="live training model"))

    def infer(self, model, step):
        accelerator = self.accelerator
        bare = accelerator.unwrap_model(model)
        folder = self.out / str(step)
        if accelerator.is_main_process:
            folder.mkdir(exist_ok=False)
        accelerator.wait_for_everyone()
        started = time.perf_counter()
        raw = {}
        with validation_state(bare, accelerator.device):
            if self.runtime is not None:
                self.runtime.refresh(bare)
            for eid, examples in self.examples.items():
                predictions, losses = [], []
                for start in range(0, len(examples), self.batch_size):
                    from accelerate.utils import send_to_device
                    batch = send_to_device(self.batches[eid][start // self.batch_size], accelerator.device)
                    batch_count = len(batch["examples"])
                    seed = int(self.cfg.validation.seed) + eid * 100000 + start
                    torch.random.default_generator.manual_seed(seed)
                    torch.cuda.manual_seed(seed)
                    with torch.compiler.set_stance("force_eager"), accelerator.autocast():
                        loss = bare(batch)["action_loss"]
                    # Separate loss draws from open-loop noise; both repeat across checkpoints.
                    torch.cuda.manual_seed(seed + 1)
                    predicted = self.runtime.infer(*self.inference_batches[eid][start // self.batch_size])[:batch_count]
                    losses.append((loss.detach().float().cpu().item(), batch_count))
                    predictions.append(predicted)
                raw[eid] = (predictions, losses)
        torch.cuda.synchronize(accelerator.device)
        accelerator.wait_for_everyone()
        gpu_seconds = time.perf_counter() - started
        # Everything below runs on CPU in the worker; no distributed calls there.
        self.pending.append(self.executor.submit(self._process_checked, raw, step, gpu_seconds))

    def _process_checked(self, raw, step, gpu_seconds):
        try:
            return self._process(raw, step, gpu_seconds)
        except Exception as error:
            write_json(self.out / str(step) / f"error-rank-{self.accelerator.process_index}.json",
                       {"error": repr(error)})
            raise

    def _process(self, raw, step, gpu_seconds):
        folder = self.out / str(step)
        base = f"/results/{self.cfg.validation.experiment_id}/{step}"
        (folder / "data").mkdir(exist_ok=True)
        for eid, (batches, losses) in raw.items():
            entry = self.prepared[eid]
            normalized = np.concatenate(batches)
            assert normalized.shape == (len(entry["anchors"]), 50, 114)
            assert np.isfinite(normalized).all() and all(np.isfinite(v) for v, n in losses)
            # Retain raw output before derived metrics or plots.
            np.savez(folder / f"raw-{eid}.npz", normalized_actions=normalized, losses=np.asarray(losses))
            p = canonical(normalized * self.asc + self.ao).reshape(-1, 114)
            t = entry["targets"]
            arrays = dict(predictions=p, targets=t, anchors=entry["anchors"], body_centers=entry["centers"],
                segment_ids=entry["segment_ids"], time_s=entry["time_s"], valid_mask=np.ones(len(p), dtype=bool),
                chunk_ids=np.repeat(np.arange(len(entry["anchors"])), 50), video_time_s=np.asarray(entry["group"]["times"]),
                checkpoint=np.asarray(str(self.out.parent / "checkpoints" / f"step_{step:06d}")), step=np.asarray(step))
            np.savez_compressed(folder / f"episode_{eid:03d}.npz", **arrays)
            plot_episode(folder / f"episode_{eid:03d}/open_loop_errors_new.png", arrays, self.meta["channels"],
                         f"UnifoLM-WLA step {step}, episode {eid}", history_frames=50)
            write_json(folder / f"data/group-{eid}.json", entry["group"])
            t.astype("<f4").tofile(folder / f"data/gt-{eid}.f32")
            np.concatenate((p, rotation_errors(p, t)), axis=-1).astype("<f4").tofile(folder / f"data/pred-{step}-{eid}.f32")
            episode = dict(episode_index=eid, samples=len(p), chunks=len(entry["anchors"]), metrics=metrics(p, t),
                loss_sum=sum(v*n for v, n in losses), loss_count=sum(n for v, n in losses),
                segments=[dict(scene=s["scene"], samples=int((entry["segment_ids"] == s["segment_index"]).sum()),
                               metrics=metrics(p[entry["segment_ids"] == s["segment_index"]], t[entry["segment_ids"] == s["segment_index"]]))
                          for s in entry["segments"]],
                assets=dict(group_url=f"{base}/data/group-{eid}.json", prediction_url=f"{base}/data/pred-{step}-{eid}.f32",
                    ground_truth_url=f"{base}/data/gt-{eid}.f32", plot_url=f"{base}/episode_{eid:03d}/open_loop_errors_new.png",
                    arrays_url=f"{base}/episode_{eid:03d}.npz",
                    videos={camera: f"{base}/media/{camera}/episode_{eid:06d}.mp4" for camera, _ in self.cameras}))
            temporary = folder / f"episode-{eid}.json.tmp"
            write_json(temporary, episode)
            temporary.replace(folder / f"episode-{eid}.json")
        if not self.accelerator.is_main_process:
            return None
        paths = [folder / f"episode-{eid}.json" for eid in self.episodes]
        while not all(path.exists() for path in paths):
            errors = list(folder.glob("error-rank-*.json"))
            if errors:
                raise RuntimeError([read_json(path) for path in errors])
            time.sleep(.5)
        episodes = [read_json(path) for path in paths]
        predictions, targets = [], []
        for eid in self.episodes:
            with np.load(folder / f"episode_{eid:03d}.npz") as arrays:
                predictions.append(arrays["predictions"])
                targets.append(arrays["targets"])
        results = metrics(np.concatenate(predictions), np.concatenate(targets))
        results["loss"] = sum(e["loss_sum"] for e in episodes) / sum(e["loss_count"] for e in episodes)
        results["gpu_phase_seconds"] = gpu_seconds
        write_json(folder / "checkpoint.json", dict(step=step, metrics=results, episodes=episodes,
            checkpoint_path=str(self.out.parent / "checkpoints" / f"step_{step:06d}"), validation={"status": "PASS"}))
        (folder / "media").mkdir()
        for camera, _ in self.cameras:
            (folder / "media" / camera).symlink_to(self.dataset / "videos/chunk-000" / camera, target_is_directory=True)
        write_json(folder / "experiment.json", dict(id=self.cfg.validation.experiment_id,
            name=self.cfg.validation.experiment_id, comparison_group="1013-rotation-dt", kind="unifolm-wla",
            dataset=str(self.dataset), source_base=base + "/", status="evaluating", expected_steps=sorted(set(range(int(self.cfg.validation.interval), int(self.cfg.trainer.max_train_steps) + 1, int(self.cfg.validation.interval))) | {int(self.cfg.trainer.max_train_steps)}),
            groups=[g for g in self.meta["groups"] if g["id"] in self.episodes], channels=self.meta["channels"],
            prediction_stride=125, ground_truth_stride=114, chunk_length=50, prompt=self.prompt,
            config=dict(inference_batch=self.batch_size, seed=int(self.cfg.validation.seed), history_frames=50, history_seconds=5, rtc=0,
                        effective_batch_size=int(self.cfg.trainer.global_batch),
                        denoising_steps=int(self.cfg.framework.action_model.num_inference_timesteps))))
        write_json(folder / "protocol.json", dict(episodes=self.episodes, inference_batch=self.batch_size,
            prediction_backend="optimized CUDA Graph, live weights refreshed", loss_backend="training eval forward",
            history_frames=50, history_seconds=5, history_time_base="20Hz observation timeline, offsets -5s..-0.1s",
            padded_tail=True, seed_formula="seed + episode_id * 100000 + batch_start; inference CUDA seed adds 1",
            seed=int(self.cfg.validation.seed), horizon=50, denoising_steps=int(self.cfg.framework.action_model.num_inference_timesteps),
            reference=str(self.reference), semantics="Root-relative body XYZ/rotation; incoming root displacement in previous root yaw"))
        (folder / "READY").touch()
        return {"validation/step": step, **{f"validation/{key}": value for key, value in results.items()}}

    def poll(self):
        completed = []
        remaining = []
        for future in self.pending:
            if future.done():
                result = future.result()
                if result is not None:
                    completed.append(result)
            else:
                remaining.append(future)
        self.pending = remaining
        return completed

    def close(self):
        self.executor.shutdown(wait=True)
        return self.poll()
