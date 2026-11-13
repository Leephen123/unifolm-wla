"""Standalone UnifoLM-WLA evaluation of full-pipeline LeRobot action windows.

Run from the UnifoLM-WLA repository with its Python environment. `evaluate`
requires a CUDA device; `publish` runs on the display service's login host.
The reference viewer supplies the validated video timeline, after exact GT and
anchor checks. Action windows are 50 body samples at 50 Hz; images are 20 Hz.
"""
import argparse
import json
from pathlib import Path
import shutil
import time
import urllib.request


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def canonical(x):
    import numpy as np
    shape = x.shape[:-1]
    body = np.concatenate((x[..., 3:36].reshape(*shape, 11, 3),
                           x[..., 36:102].reshape(*shape, 11, 6)), axis=-1)
    return np.concatenate((x[..., :3], body.reshape(*shape, 99), x[..., 102:]), axis=-1)


def rotation_errors(pred, target):
    import numpy as np
    def matrices(x):
        v = x[:, 3:102].reshape(-1, 11, 9)[..., 3:].astype(np.float64)
        a, b = v[..., :3], v[..., 3:]
        an = np.linalg.norm(a, axis=-1, keepdims=True)
        assert (an > 0).all(), "Undefined rotation: zero first column"
        a = a / an
        b = b - np.sum(a * b, axis=-1, keepdims=True) * a
        bn = np.linalg.norm(b, axis=-1, keepdims=True)
        assert (bn > 0).all(), "Undefined rotation: collinear columns"
        b = b / bn
        return np.stack((a, b, np.cross(a, b)), axis=-1)
    p, t = matrices(pred), matrices(target)
    return np.rad2deg(np.arccos(np.clip((np.sum(p * t, axis=(-1, -2)) - 1) / 2, -1, 1)))


def metrics(pred, target):
    import numpy as np
    diff = pred.astype(np.float64) - target.astype(np.float64)
    xyz = [3 + 9 * j + k for j in range(11) for k in range(3)]
    angles = rotation_errors(pred, target)
    return dict(root_delta_rmse_m=float(np.sqrt(np.mean(diff[:, :3] ** 2))),
                body_xyz_rmse_cm=float(np.sqrt(np.mean(diff[:, xyz] ** 2)) * 100),
                hand_rmse=float(np.sqrt(np.mean(diff[:, 102:] ** 2))),
                rotation_mean_deg=float(np.mean(angles)),
                rotation_rmse_deg=float(np.sqrt(np.mean(angles ** 2))))


def plot_episode(path, arrays, channels, title, history_frames=0):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p, t = arrays["predictions"], arrays["targets"]
    angles = rotation_errors(p, t)
    selected = list(range(3)) + [3 + 9*j + k for j in range(11) for k in range(3)] + list(range(102, 114))
    labels = {c["index"]: c for c in channels if c["key"].startswith("v")}
    fig, axes = plt.subplots(10, 6, figsize=(26, 32))
    for panel, ax in enumerate(axes.flat):
        if panel == 59:
            ax.axis("off")
            ax.text(0, 1, "\n".join(f"{k}: {v:.5g}" for k, v in metrics(p, t).items()), va="top")
            continue
        for sid in np.unique(arrays["segment_ids"]):
            mask = arrays["segment_ids"] == sid
            x = arrays["time_s"][mask]
            if panel < 48:
                c = selected[panel]
                ax.plot(x, t[mask, c], lw=.65, label="GT", color="black")
                ax.plot(x, p[mask, c], lw=.65, label="Prediction", alpha=.8, color="tab:blue")
            else:
                ax.plot(x, angles[mask, panel-48], lw=.65, color="tab:red")
        if panel < 48:
            c = labels[selected[panel]]
            label = c["label"].replace("身体点", "Body").replace("左手", "Left hand").replace("右手", "Right hand").replace("通道", "channel")
            ax.set_title(f"{label} ({c['unit']})", fontsize=9)
        else:
            ax.set_title(f"Body {panel-48}: rotation geodesic error (deg)", fontsize=9)
        for sid in np.unique(arrays["segment_ids"])[1:]:
            boundary = arrays["time_s"][arrays["segment_ids"] == sid][0]
            ax.axvline(boundary, color="gray", ls="--", lw=.6)
            if panel == 0:
                ax.text(boundary, .98, f"Segment {sid}", transform=ax.get_xaxis_transform(),
                        rotation=90, va="top", fontsize=6)
        ax.grid(alpha=.2)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("Body time (s), segments concatenated", fontsize=7)
    axes.flat[0].legend([axes.flat[0].lines[0], axes.flat[0].lines[1]], ["GT", "Prediction"], fontsize=8)
    fig.suptitle(title + "\nRoot-relative body XYZ/rotation; incoming root displacement in previous root yaw.\n"
                 f"50-step open-loop windows, 50 Hz body / 20 Hz observation; RTC0, {history_frames} state-history frames; lines break at source segments.", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def prepare_episode(dataset, reference, eid):
    import numpy as np
    meta = read_json(reference / f"episode-{eid}.json")
    with np.load(reference / f"episode-{eid}.npz") as data:
        result = {key: data[key] for key in data.files}
    result.update(meta)
    return result


def evaluate(args):
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from safetensors.torch import load_file
    from unifolm_wla.model.framework.base_framework import build_framework
    from unifolm_wla.dataloader.full_pipeline import CAMERAS, pad_resize, pack_state, read_frame
    out, checkpoint = args.output.resolve(), args.checkpoint.resolve()
    assert (checkpoint / "COMPLETE").is_file()
    step = read_json(checkpoint / "progress.json")["step"]
    cfg = OmegaConf.load(args.config)
    assert cfg.framework.action_model.action_dim == 114
    assert cfg.framework.action_model.action_horizon == 50
    assert list(cfg.datasets.vla_data.image_size_wh) == [448, 320]
    assert read_json(args.dataset / "meta/info.json")["fps"] == 20
    stats = read_json(cfg.datasets.vla_data.statistics_path)["full_pipeline"]
    so, ss = [np.asarray(stats["state"][k], dtype=np.float32) for k in ("offset", "scale")]
    ao, asc = [np.asarray(stats["action"][k], dtype=np.float32) for k in ("offset", "scale")]
    assert so.shape == ss.shape == (47,) and ao.shape == asc.shape == (114,)
    assert (ss != 0).all() and (asc != 0).all()
    prompt = read_json(cfg.datasets.vla_data.manifest_path)["prompt"]
    reference_meta = read_json(args.reference / "manifest.json")
    prepared = {eid: prepare_episode(args.dataset, args.reference, eid) for eid in args.episodes}
    out.mkdir(parents=True, exist_ok=True)
    result_dir = out / str(step)
    result_dir.mkdir(exist_ok=False)
    (out / "data").mkdir(exist_ok=True)
    (out / "media").mkdir(exist_ok=True)
    for camera, _ in CAMERAS:
        (out / "media" / camera).symlink_to((args.dataset / "videos/chunk-000" / camera).resolve(), target_is_directory=True)
    write_json(out / "preflight.json", dict(status="PASS", reference=str(args.reference),
               checks=["exact ground truth", "anchors", "source segments", "finite inputs", "timeline length"],
               chunks={eid: len(e["anchors"]) for eid, e in prepared.items()}))
    OmegaConf.save(cfg, out / "config.yaml")
    shutil.copyfile(cfg.datasets.vla_data.statistics_path, out / "dataset_statistics.json")
    shutil.copyfile(args.dataset / "segments.json", out / "source_segments.json")
    shutil.copyfile(__file__, out / "evaluator.py")
    write_json(out / "protocol.json", dict(checkpoint=str(checkpoint), step=step, dataset=str(args.dataset),
               prompt=prompt, history=0, rtc=0, horizon=50, stride_body_frames=50, seed=args.seed,
               seed_formula="seed + episode_index * 100000 + anchor", inference_batch=1,
               denoising_steps=int(cfg.framework.action_model.num_inference_timesteps),
               semantics="root-relative XYZ/rotation; incoming root displacement in previous root yaw",
               input_layout="state47 -> masked state60", native_layout="root3 + xyz33 + rot6d66 + hands12",
               saved_layout="root3 + 11*(xyz3+rot6d6) + hands12"))
    print("PREFLIGHT PASS", {i: len(e["anchors"]) for i, e in prepared.items()}, flush=True)
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    print("BUILDING MODEL", flush=True)
    model = build_framework(cfg).to("cuda")
    print("LOADING CHECKPOINT TENSORS", flush=True)
    weights = load_file(str(checkpoint / "model.safetensors"))
    model.load_state_dict(weights, strict=True)
    del weights
    model.eval().requires_grad_(False)
    model.qwen_vl_interface.model.config.use_cache = False
    print("CHECKPOINT LOADED", checkpoint, flush=True)
    started = time.monotonic()
    episodes = []
    all_predictions, all_targets = [], []
    base = f"/results/{args.experiment_id}"
    for eid, entry in prepared.items():
        predictions = []
        for k, anchor in enumerate(entry["anchors"]):
            state, mask = pack_state((entry["states"][anchor] - so) / ss)
            images = [pad_resize(read_frame(str(args.dataset / f"videos/chunk-000/{camera}/episode_{eid:06d}.mp4"), int(anchor))) for camera, _ in CAMERAS]
            example = dict(image=images, image_roles=[role for _, role in CAMERAS], lang=prompt,
                           state=state, state_mask=mask, action_mask=np.ones(114, dtype=bool),
                           arm_type="dual_with_legs", robot_type="unitree")
            torch.manual_seed(args.seed + eid * 100000 + int(anchor))
            with torch.inference_mode():
                normalized = model.predict_action([example])["normalized_actions"]
            assert normalized.shape == (1, 50, 114) and np.isfinite(normalized).all()
            predictions.append(canonical(normalized[0] * asc + ao))
            if k % 20 == 0 or k + 1 == len(entry["anchors"]):
                print(f"episode={eid} chunks={k+1}/{len(entry['anchors'])} elapsed={time.monotonic()-started:.1f}s peak_cuda_GiB={torch.cuda.max_memory_allocated()/2**30:.2f}", flush=True)
        p, t = np.concatenate(predictions).astype(np.float32), entry["targets"]
        arrays = dict(predictions=p, targets=t, anchors=entry["anchors"], body_centers=entry["centers"],
                      segment_ids=entry["segment_ids"], time_s=entry["time_s"],
                      valid_mask=np.ones(len(p), dtype=bool), chunk_ids=np.repeat(np.arange(len(predictions)), 50),
                      video_time_s=np.asarray(entry["group"]["times"]),
                      checkpoint=np.asarray(str(checkpoint)), step=np.asarray(step))
        np.savez_compressed(result_dir / f"episode_{eid:03d}.npz", **arrays)
        saved = np.load(result_dir / f"episode_{eid:03d}.npz", allow_pickle=False)
        np.testing.assert_array_equal(saved["predictions"], p)
        np.testing.assert_array_equal(saved["targets"], t)
        plot_episode(result_dir / f"episode_{eid:03d}/open_loop_errors_new.png", saved, reference_meta["channels"], f"UnifoLM-WLA step {step}, episode {eid}")
        write_json(out / f"data/group-{eid}.json", entry["group"])
        t.astype("<f4").tofile(out / f"data/gt-{eid}.f32")
        np.concatenate((p, rotation_errors(p, t)), axis=-1).astype("<f4").tofile(out / f"data/pred-{step}-{eid}.f32")
        segments = []
        for s in entry["segments"]:
            mask = entry["segment_ids"] == s["segment_index"]
            segments.append(dict(scene=s["scene"], samples=int(mask.sum()), metrics=metrics(p[mask], t[mask])))
        episodes.append(dict(episode_index=eid, samples=len(p), chunks=len(predictions), metrics=metrics(saved["predictions"], saved["targets"]), segments=segments,
            assets=dict(group_url=f"{base}/data/group-{eid}.json", prediction_url=f"{base}/data/pred-{step}-{eid}.f32",
                        ground_truth_url=f"{base}/data/gt-{eid}.f32", plot_url=f"{base}/{step}/episode_{eid:03d}/open_loop_errors_new.png",
                        arrays_url=f"{base}/{step}/episode_{eid:03d}.npz",
                        videos={camera: f"{base}/media/{camera}/episode_{eid:06d}.mp4" for camera, _ in CAMERAS})))
        all_predictions.append(p)
        all_targets.append(t)
    validation = dict(status="PASS", episodes=len(episodes), chunks=sum(e["chunks"] for e in episodes),
                      samples=sum(e["samples"] for e in episodes), full_plots=len(episodes),
                      peak_cuda_GiB=torch.cuda.max_memory_allocated()/2**30,
                      elapsed_seconds=time.monotonic()-started,
                      checks=["reference GT/anchor/segment equality", "finite predictions", "strict checkpoint loading", "saved array roundtrip and metrics"])
    write_json(out / "validation.json", validation)
    payload = dict(step=step, metrics=metrics(np.concatenate(all_predictions), np.concatenate(all_targets)),
                   validation=dict(status="PASS", report=str(out / "validation.json")), checkpoint_path=str(checkpoint), episodes=episodes)
    write_json(out / "checkpoint.json", payload)
    metadata = dict(id=args.experiment_id, name="UnifoLM-WLA full VLM / 1015 test", comparison_group="1013-rotation-dt",
                    dataset=str(args.dataset), kind="unifolm-wla", source_base=base+"/", status="evaluating", expected_steps=[step],
                    groups=[g for g in reference_meta["groups"] if g["id"] in args.episodes], channels=reference_meta["channels"],
                    prediction_stride=125, ground_truth_stride=114, chunk_length=50, prompt=prompt,
                    description="Current three cameras and state47; no history/RTC. Root-relative body pose; incoming root displacement.",
                    config=dict(history_frames=0, rtc=0, denoising_steps=int(cfg.framework.action_model.num_inference_timesteps),
                                effective_batch_size=int(cfg.trainer.global_batch), inference_batch=1, freeze_vlm=False,
                                image_size_wh=[448, 320], seed=args.seed), watcher=None)
    write_json(out / "experiment.json", metadata)
    print("COMPLETE", json.dumps(payload["metrics"]), flush=True)


def publish(args):
    def request(method, endpoint, body=None):
        data = None if body is None else json.dumps(body, allow_nan=False).encode()
        req = urllib.request.Request(args.service_url + "/api/v1" + endpoint, data=data, method=method,
              headers={"Content-Type": "application/json", "X-OpenLoop-Client": "unifolm-open-loop-eval"})
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)
    meta, payload = read_json(args.output / "experiment.json"), read_json(args.output / "checkpoint.json")
    request("GET", "/health")
    catalog = request("GET", "/catalog")
    write_json(args.output / "catalog-before-publish.json", catalog)
    asset_root = (args.asset_root or args.output).resolve()
    link = args.site_root / "results" / meta["id"]
    link.parent.mkdir(exist_ok=True)
    if link.is_symlink():
        assert link.resolve() == asset_root
    else:
        link.symlink_to(asset_root, target_is_directory=True)
    print(request("POST", "/experiments", meta))
    endpoint = f"/experiments/{meta['id']}/checkpoints/{payload['step']}"
    print(request("PUT", endpoint, payload))
    write_json(args.output / "published-checkpoint.json", request("GET", endpoint))
    for ep in payload["episodes"]:
        for name, url in ep["assets"].items():
            urls = list(url.values()) if name == "videos" else [url]
            for asset_url in urls:
                req = urllib.request.Request(args.service_url + asset_url, headers={"Range": "bytes=0-1023"})
                with urllib.request.urlopen(req, timeout=30) as response:
                    assert response.status in (200, 206)
                    assert response.read(16)
                    if name == "videos":
                        assert response.status == 206
    if not args.keep_evaluating:
        print(request("PATCH", f"/experiments/{meta['id']}", {"status": "completed"}))
    write_json(args.output / "http-validation.json", dict(status="PASS", scope="API, arrays, plots, timeline, video Range; browser playback not tested"))


def render(args):
    """Regenerate complete plots from saved arrays without model inference."""
    import numpy as np
    meta = read_json(args.output / "experiment.json")
    payload = read_json(args.output / "checkpoint.json")
    step = payload["step"]
    for episode in payload["episodes"]:
        eid = episode["episode_index"]
        with np.load(args.output / str(step) / f"episode_{eid:03d}.npz", allow_pickle=False) as arrays:
            plot_episode(args.output / str(step) / f"episode_{eid:03d}/open_loop_errors_new.png",
                         arrays, meta["channels"], f"UnifoLM-WLA step {step}, episode {eid}")
        print(f"Rendered episode {eid}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = sub.add_parser("evaluate", help="Evaluate one complete checkpoint on CUDA")
    for name in ("checkpoint", "config", "dataset", "reference", "output"):
        evaluate_parser.add_argument("--" + name, type=Path, required=True)
    evaluate_parser.add_argument("--experiment-id", required=True)
    evaluate_parser.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    evaluate_parser.add_argument("--seed", type=int, default=42)
    evaluate_parser.add_argument("--cpu-threads", type=int, default=8)
    publish_parser = sub.add_parser("publish", help="Publish saved outputs through the existing service API")
    publish_parser.add_argument("--output", type=Path, required=True)
    publish_parser.add_argument("--site-root", type=Path, required=True)
    publish_parser.add_argument("--service-url", default="http://127.0.0.1:18714")
    publish_parser.add_argument("--asset-root", type=Path, help="Shared asset root for training validation steps")
    publish_parser.add_argument("--keep-evaluating", action="store_true")
    render_parser = sub.add_parser("render", help="Regenerate full plots from saved arrays without inference")
    render_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {"evaluate": evaluate, "publish": publish, "render": render}[args.command](args)


if __name__ == "__main__":
    main()
