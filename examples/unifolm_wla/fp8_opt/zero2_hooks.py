"""DeepSpeed ZeRO-2 NVLS SUM + Adaptive Vision-Tail IPG Flush + Two-Stage Async Weight AG + Fused AdamW."""
from __future__ import annotations

import math
import os
import types
from functools import wraps
from inspect import getattr_static
import torch
import torch.distributed as dist

from .cuda_ops import _cuda_device, _ptr, _stream, get_cuda_lib, get_pool, register_direct_ipg_hook, wait_fp8_weight_quant
from .delayed_scaling import clear_all_fp8_weights


def flush_pending_weight_gather(model_or_engine) -> None:
    """Finish pending ZeRO weight broadcasts collectively before exporting weights."""
    bare = model_or_engine
    while hasattr(bare, "module"):
        # Transparent wrappers can proxy these attributes while assignments stay on the wrapper.
        if any(getattr_static(bare, name, None) is not None for name in
               ("_flush_pending_non_vis_ag", "_pending_non_vis_ag", "_deferred_fp8_quant_fn")):
            break
        bare = bare.module
    flush = getattr(bare, "_flush_pending_non_vis_ag", None)
    if callable(flush):
        flush()
        return
    dev = _cuda_device(bare)
    if dev is None:
        return
    stream = torch.cuda.current_stream(dev)
    deferred = getattr(bare, "_deferred_fp8_quant_fn", None)
    if deferred is not None:
        bare._deferred_fp8_quant_fn = None
        deferred(stream.record_event())
    pending = getattr(bare, "_pending_non_vis_ag", None)
    if pending is not None:
        bare._pending_non_vis_ag = None
        pending(stream)
    wait_fp8_weight_quant(dev)


def _install_checkpoint_flush(engine):
    def wrap_save(save):
        @wraps(save)
        def wrapped(eng_self, *args, **kwargs):
            flush_pending_weight_gather(eng_self)
            return save(*args, **kwargs)
        wrapped._wla_flush = True
        return types.MethodType(wrapped, engine)

    for name in ("save_checkpoint", "save_16bit_model"):
        save = getattr(engine, name, None)
        if callable(save) and not getattr(save, "_wla_flush", False):
            setattr(engine, name, wrap_save(save))

    try:
        from accelerate import Accelerator
    except ImportError:
        return
    original = Accelerator.get_state_dict
    if not getattr(original, "_wla_flush", False):
        @wraps(original)
        def get_state_dict(self, model, *args, **kwargs):
            flush_pending_weight_gather(model)
            return original(self, model, *args, **kwargs)
        get_state_dict._wla_flush = True
        Accelerator.get_state_dict = get_state_dict

def fused_adamw_bf16_io_cuda(
    fp32_p: torch.Tensor,
    fp32_m: torch.Tensor,
    fp32_v: torch.Tensor,
    bf16_g: torch.Tensor,
    bf16_p_out: torch.Tensor,
    inv_scale_f32: torch.Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    step: int,
    eps: float,
    weight_decay: float,
):
    import math
    bc1 = 1.0 - (beta1 ** step)
    bc2 = 1.0 - (beta2 ** step)
    step_size = lr / bc1
    inv_bc2_sqrt = 1.0 / math.sqrt(bc2)
    decay_factor = 1.0 - lr * weight_decay
    get_cuda_lib().launch_fused_adamw_bf16_io(
        _ptr(fp32_p), _ptr(fp32_m), _ptr(fp32_v),
        _ptr(bf16_g), _ptr(bf16_p_out), _ptr(inv_scale_f32),
        float(lr), float(beta1), float(beta2),
        float(step_size), float(inv_bc2_sqrt),
        float(eps), float(decay_factor),
        int(fp32_p.numel()), _stream(fp32_p.device),
    )


def install_fast_zero2_hooks(model_or_engine) -> dict:
    ds_opt = getattr(model_or_engine, "optimizer", None)
    if ds_opt is None or type(ds_opt).__name__ != "DeepSpeedZeroOptimizer":
        return {"installed": False}
    import math
    import torch.distributed as dist
    from deepspeed.runtime.utils import is_model_parallel_parameter
    from deepspeed.moe.utils import is_moe_param

    orig_grad_red = ds_opt.gradient_reduction_w_predivide
    orig_grad_norm = ds_opt.get_grad_norm_direct
    orig_unscale = ds_opt.unscale_and_clip_grads
    orig_step = ds_opt.step
    orig_load_state_dict = ds_opt.load_state_dict
    orig_reduce_ipg = ds_opt.reduce_ipg_grads
    orig_ipg_bucket_fn = ds_opt.reduce_independent_p_g_buckets_and_remove_grads
    orig_ipg_epilogue = ds_opt.independent_gradient_partition_epilogue
    orig_copy_part = ds_opt.copy_grads_in_partition
    dp_ws = float(dist.get_world_size(group=ds_opt.dp_process_group))

    use_deferred_div = (
        bool(ds_opt.postscale_gradients)
        and ds_opt.communication_data_type == ds_opt.dtype
        and float(ds_opt.gradient_predivide_factor) == 1.0
        and float(ds_opt.sequence_parallel_size) == 1.0
        and dp_ws > 1.0
    )

    dev = torch.device(f"cuda:{torch.cuda.current_device()}")
    enable_adaptive_vision_flush = (os.environ.get("WLA_VISION_TAIL_FLUSH", "1") == "1")
    vision_bucket_elems = int(os.environ.get("WLA_VISION_BUCKET_ELEMS", "155000000"))
    in_vision_tail = [False]
    vision_tail_flushed = [False]
    ipg_buf_ready_ev = [torch.cuda.Event(), torch.cuda.Event()]
    ipg_buf_in_flight = [False, False]
    direct_committed_pids: set[int] = set()

    # Direct Ordered Partition Gradient Buffer (`flat_part_grad`):
    # Eliminates `1.46 GB` `grads_in_partition` AND `1.74 ms` exposed `get_flat_partition` (`CatArrayBatchedCopy`)!
    can_direct_flat_part = (
        os.environ.get("WLA_DIRECT_FLAT_PART", "1") == "1"
        and len(ds_opt.bit16_groups) == 1
        and (not bool(ds_opt.cpu_offload))
        and (not bool(ds_opt.use_grad_accum_attribute))
        and ds_opt.dtype == torch.bfloat16
        and ds_opt.gradient_accumulation_dtype == torch.bfloat16
    )
    part_dst_map: dict[int, tuple[int, int, int]] = {}
    flat_part_grad: torch.Tensor | None = None
    touched_part_ids: set[int] = set()
    grad_accum_has_data = [False]

    if can_direct_flat_part:
        first_off0 = int(ds_opt.first_offset[0])
        part_sz0 = int(ds_opt.partition_size[0])
        cur_dst = 0
        for idx_p, p in enumerate(ds_opt.params_in_partition[0]):
            src_off = first_off0 if (idx_p == 0 and first_off0 > 0) else 0
            n_el = int(p.numel()) - src_off
            if n_el > (part_sz0 - cur_dst):
                n_el = part_sz0 - cur_dst
            if n_el > 0:
                part_dst_map[id(p)] = (int(src_off), int(cur_dst), int(n_el))
                cur_dst += int(n_el)
        flat_part_grad = torch.zeros(part_sz0, dtype=torch.bfloat16, device=dev)

    def _fast_copy_grads_in_partition(self, param):
        if not can_direct_flat_part or self.cpu_offload:
            return orig_copy_part(param)
        info = part_dst_map.get(id(param))
        if info is not None:
            if grad_accum_has_data[0] and self.averaged_gradients.get(0) is None:
                grad_accum_has_data[0] = False
            src_off, dst_off, n_el = info
            grad_reduc = self.get_gradient_for_reduction(param)
            src_slice = grad_reduc.view(-1).narrow(0, src_off, n_el)
            dst_slice = flat_part_grad.narrow(0, dst_off, n_el)
            if grad_accum_has_data[0]:
                dst_slice.add_(src_slice)
            else:
                dst_slice.copy_(src_slice)
            touched_part_ids.add(id(param))
        self.clear_grad_attribute(param)

    def _fast_reduce_ipg_grads(self):
        cur_idx = self.ipg_index if (self.contiguous_gradients and self.overlap_comm) else -1
        has_work = (self.elements_in_ipg_bucket > 0) or (self.extra_large_param_to_reduce is not None)
        if has_work and self.overlap_comm and self.reduction_stream is not None:
            self.reduction_stream.wait_stream(torch.cuda.current_stream(dev))
        orig_reduce_ipg()
        if has_work and cur_idx in (0, 1) and self.reduction_stream is not None:
            ipg_buf_ready_ev[cur_idx].record(self.reduction_stream)
            ipg_buf_in_flight[cur_idx] = True

    def _flush_and_swap_ipg(self):
        self.reduce_ipg_grads()
        if self.contiguous_gradients and self.overlap_comm:
            self.ipg_index = 1 - self.ipg_index
            if ipg_buf_in_flight[self.ipg_index]:
                torch.cuda.current_stream(dev).wait_event(ipg_buf_ready_ev[self.ipg_index])
                ipg_buf_in_flight[self.ipg_index] = False

    def _fast_reduce_independent_p_g_buckets_and_remove_grads(self, param, i):
        pid_chk = self.param_id.get(id(param))
        if pid_chk is not None and pid_chk in direct_committed_pids:
            direct_committed_pids.remove(pid_chk)
            return

        if self.elements_in_ipg_bucket + param.numel() > self.reduce_bucket_size:
            _flush_and_swap_ipg(self)
        elif self.contiguous_gradients and self.overlap_comm and ipg_buf_in_flight[self.ipg_index]:
            torch.cuda.current_stream(dev).wait_event(ipg_buf_ready_ev[self.ipg_index])
            ipg_buf_in_flight[self.ipg_index] = False

        if not enable_adaptive_vision_flush:
            return orig_ipg_bucket_fn(param, i)

        orig_ipg_bucket_fn(param, i)

        if param.numel() >= 100_000_000:
            in_vision_tail[0] = True
            vision_tail_flushed[0] = False
        elif (
            in_vision_tail[0]
            and (not vision_tail_flushed[0])
            and self.elements_in_ipg_bucket < 350_000_000
            and self.elements_in_ipg_bucket >= vision_bucket_elems
        ):
            vision_tail_flushed[0] = True
            _flush_and_swap_ipg(self)

    # Zero-Copy Direct IPG `dW` Output Hook (`lt_fp8_mm` & `lt_bf16_dw_dbias` write straight into `ipg_buffer`!)
    enable_direct_ipg = (
        os.environ.get("WLA_DIRECT_IPG", "1") == "1"
        and bool(ds_opt.contiguous_gradients)
        and (not bool(ds_opt.reduce_scatter))
        and (not bool(ds_opt.cpu_offload))
        and (not bool(ds_opt.use_grad_accum_attribute))
        and (not bool(ds_opt.has_moe_layers))
        and len(ds_opt.bit16_groups) == 1
    )
    direct_ipg_stats = [0, 0]  # [hits, fallbacks]

    def _try_direct_ipg_block(all_params: tuple, dw_params: tuple):
        if not enable_direct_ipg:
            return None
        if not ds_opt.partition_gradients and not ds_opt.is_gradient_accumulation_boundary:
            return None
        if not ds_opt.ready_for_gradients:
            ds_opt.backward_prologue()
        if ds_opt.ipg_buffer is None:
            return None

        total_numel = 0
        for p in all_params:
            if p is not None and p.requires_grad:
                total_numel += int(p.numel())

        cur_elems = int(ds_opt.elements_in_ipg_bucket)
        if in_vision_tail[0] and (not vision_tail_flushed[0]) and cur_elems < vision_bucket_elems:
            limit = min(int(ds_opt.reduce_bucket_size), vision_bucket_elems)
        else:
            limit = int(ds_opt.reduce_bucket_size)

        if cur_elems + total_numel > limit:
            direct_ipg_stats[1] += 1
            return None

        for p in dw_params:
            pid = ds_opt.param_id.get(id(p))
            if (
                pid is None
                or (not p.requires_grad)
                or (p.grad is not None)
                or ds_opt.params_already_reduced[pid]
                or int(p.numel()) > int(ds_opt.reduce_bucket_size)
            ):
                direct_ipg_stats[1] += 1
                return None

        if ds_opt.overlap_comm and ipg_buf_in_flight[ds_opt.ipg_index]:
            torch.cuda.current_stream(dev).wait_event(ipg_buf_ready_ev[ds_opt.ipg_index])
            ipg_buf_in_flight[ds_opt.ipg_index] = False

        direct_ipg_stats[0] += 1

        def _alloc_and_commit(p: torch.Tensor) -> torch.Tensor:
            n_el = int(p.numel())
            sub_view = ds_opt.ipg_buffer[ds_opt.ipg_index].narrow(0, int(ds_opt.elements_in_ipg_bucket), n_el).view_as(p)
            ds_opt.elements_in_ipg_bucket += n_el
            p.grad = sub_view
            pid = ds_opt.param_id[id(p)]
            ds_opt.grads_in_ipg_bucket.append(sub_view)
            ds_opt.params_in_ipg_bucket.append((0, p.param_idx_in_group, pid))
            direct_committed_pids.add(pid)
            return sub_view

        return _alloc_and_commit

    register_direct_ipg_hook(dev, _try_direct_ipg_block if enable_direct_ipg else None)
    ds_opt._direct_ipg_stats = direct_ipg_stats

    def _fast_independent_gradient_partition_epilogue(self):
        in_vision_tail[0] = False
        vision_tail_flushed[0] = False
        direct_committed_pids.clear()
        if self.cpu_offload:
            return orig_ipg_epilogue()

        self.reduce_ipg_grads()
        for idx in range(len(self.params_already_reduced)):
            self.params_already_reduced[idx] = False

        if self.overlap_comm:
            # GPU stream event wait instead of blocking CPU `torch.cuda.synchronize()` (-1.03 ms CPU stall!)
            torch.cuda.current_stream(dev).wait_stream(self.reduction_stream)
            self._clear_previous_reduced_grads()
        ipg_buf_in_flight[0] = False
        ipg_buf_in_flight[1] = False

        if can_direct_flat_part:
            if grad_accum_has_data[0] and self.averaged_gradients.get(0) is None:
                grad_accum_has_data[0] = False
            if (not grad_accum_has_data[0]) and len(touched_part_ids) < len(part_dst_map):
                for pid_k, (_, dst_off, n_el) in part_dst_map.items():
                    if pid_k not in touched_part_ids:
                        flat_part_grad.narrow(0, dst_off, n_el).zero_()
            touched_part_ids.clear()
            grad_accum_has_data[0] = True
            self.averaged_gradients[0] = [flat_part_grad]
        else:
            for grp_i, _ in enumerate(self.bit16_groups):
                if grp_i not in self.averaged_gradients or self.averaged_gradients[grp_i] is None:
                    self.averaged_gradients[grp_i] = self.get_flat_partition(
                        self.params_in_partition[grp_i],
                        self.first_offset[grp_i],
                        self.partition_size[grp_i],
                        dtype=self.gradient_accumulation_dtype,
                        device=dev,
                        return_tensor_list=True,
                    )
                else:
                    avg_new = self.get_flat_partition(
                        self.params_in_partition[grp_i],
                        self.first_offset[grp_i],
                        self.partition_size[grp_i],
                        dtype=self.gradient_accumulation_dtype,
                        device=dev,
                        return_tensor_list=True,
                    )
                    for acc_g, new_g in zip(self.averaged_gradients[grp_i], avg_new):
                        acc_g.add_(new_g)

        self._release_ipg_buffers()
        self.zero_grad(set_to_none=True)

    def _fast_gradient_reduction_w_predivide(self, tensor):
        if tensor.numel() == 0:
            return tensor
        if use_deferred_div:
            if tensor.dtype == self.communication_data_type:
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.dp_process_group)
            else:
                tmp = tensor.to(self.communication_data_type)
                dist.all_reduce(tmp, op=dist.ReduceOp.SUM, group=self.dp_process_group)
                tensor.copy_(tmp)
            return tensor
        return orig_grad_red(tensor)

    def _fast_get_grad_norm_direct(self, gradients, params, norm_type=2):
        norm_type_f = float(norm_type)
        if norm_type_f != 2.0:
            n = orig_grad_norm(gradients, params, norm_type=norm_type)
            return (n / dp_ws) if use_deferred_div else n
        valid_grads = [
            g.detach()
            for g, p in zip(gradients, params)
            if not (hasattr(p, "ds_pipe_replicated") and p.ds_pipe_replicated)
            and (is_model_parallel_parameter(p) or self.model_parallel_rank == 0)
        ]
        if valid_grads:
            norms = torch._foreach_norm(valid_grads, 2.0)
            total_norm = torch.stack(norms).double().square().sum().float()
        else:
            total_norm = torch.zeros((), dtype=torch.float32, device=self.device)
        dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=self.dp_process_group)
        self._model_parallel_all_reduce(tensor=total_norm, op=dist.ReduceOp.SUM)
        total_norm = total_norm.sqrt()
        if use_deferred_div:
            total_norm = total_norm * (1.0 / dp_ws)
        return torch.nan_to_num(total_norm, nan=float("inf"), posinf=float("inf"), neginf=float("inf"))

    def _fast_unscale_and_clip_grads(self, grad_groups_flat, total_norm):
        combined_scale = float(self.loss_scale)
        if self.clip_grad > 0.0:
            clip = ((total_norm / self.loss_scale) + 1e-6) / self.clip_grad
            clip = torch.clamp(clip, min=1.0)
            combined_scale = clip * self.loss_scale
        if use_deferred_div:
            combined_scale = combined_scale * dp_ws
        inv_scale = 1.0 / combined_scale
        for grad in grad_groups_flat:
            if isinstance(grad, list):
                for g in grad:
                    g.data.mul_(inv_scale)
            else:
                grad.data.mul_(inv_scale)

    norm_sq_f64 = torch.zeros(1, dtype=torch.float64, device=dev)
    global_norm_f32 = torch.zeros(1, dtype=torch.float32, device=dev)
    inv_scale_f32 = torch.zeros(1, dtype=torch.float32, device=dev)
    step_int_cache = {}
    enable_fused_step = (os.environ.get("WLA_FUSED_ZERO2_ADAMW", "1") == "1")

    def _fast_zero2_step(self, closure=None):
        try:
            if (
                (not enable_fused_step)
                or closure is not None
                or self.cpu_offload
                or self.overflow
                or self.dtype != torch.bfloat16
                or self.has_moe_layers
                or self.model_parallel_world_size > 1
                or any(fp32_p not in self.optimizer.state for fp32_p in self.single_partition_of_fp32_groups)
            ):
                step_int_cache.clear()
                return orig_step(closure=closure)

            lib = get_cuda_lib()
            st_cuda = _stream(dev)
            self.micro_step_id = -1
            prev_scale = float(self.loss_scale)
            self._update_scale(False)

            n_groups = len(self.bit16_groups)
            flat_bf16_grads = []

            for i in range(n_groups):
                self.free_grad_in_param_list(self.params_not_in_partition[i])
                part_id = dist.get_rank(group=self.real_dp_process_group[i])
                part_ws = dist.get_world_size(group=self.real_dp_process_group[i])
                part_sz = int(self.partition_size[i])
                if can_direct_flat_part and i == 0 and len(self.averaged_gradients[0]) == 1 and self.averaged_gradients[0][0] is flat_part_grad:
                    flat_g = flat_part_grad
                elif part_id == part_ws - 1:
                    flat_g = self.flatten_dense_tensors_aligned(self.averaged_gradients[i], part_sz)
                else:
                    flat_g = self.flatten(self.averaged_gradients[i])
                self.free_grad_in_param_list(self.params_in_partition[i])
                self.averaged_gradients[i] = None
                flat_bf16_grads.append((flat_g, part_id))
                lib.launch_bf16_sum_sq_f64(_ptr(flat_g), _ptr(norm_sq_f64), int(flat_g.numel()), 1 if i == 0 else 0, st_cuda)

            dist.all_reduce(norm_sq_f64, op=dist.ReduceOp.SUM, group=self.dp_process_group)
            lib.launch_compute_adamw_inv_scale(
                _ptr(norm_sq_f64), _ptr(global_norm_f32), _ptr(inv_scale_f32),
                float(prev_scale), float(self.clip_grad), float(dp_ws),
                1 if use_deferred_div else 0, st_cuda,
            )
            self._global_grad_norm = global_norm_f32

            for i, (flat_g, part_id) in enumerate(flat_bf16_grads):
                fp32_p = self.single_partition_of_fp32_groups[i]
                opt_st = self.optimizer.state[fp32_p]
                pg = self.optimizer.param_groups[i]
                if i not in step_int_cache:
                    step_int_cache[i] = int(opt_st["step"].item())
                step_int_cache[i] += 1
                opt_st["step"].add_(1)
                step_num = step_int_cache[i]

                lr = float(pg["lr"])
                beta1, beta2 = float(pg["betas"][0]), float(pg["betas"][1])
                eps = float(pg["eps"])
                wd = float(pg.get("weight_decay", 0.0))
                bc1 = 1.0 - (beta1 ** step_num)
                bc2 = 1.0 - (beta2 ** step_num)
                step_size = lr / bc1
                inv_bc2_sqrt = 1.0 / math.sqrt(bc2)
                decay_factor = 1.0 - lr * wd

                bf16_out = self.parallel_partitioned_bit16_groups[i][part_id]
                lib.launch_fused_adamw_bf16_io(
                    _ptr(fp32_p), _ptr(opt_st["exp_avg"]), _ptr(opt_st["exp_avg_sq"]),
                    _ptr(flat_g), _ptr(bf16_out), _ptr(inv_scale_f32),
                    lr, beta1, beta2, step_size, inv_bc2_sqrt, eps, decay_factor,
                    int(fp32_p.numel()), st_cuda,
                )
            del flat_bf16_grads

            sc_now = get_pool(dev).step_count
            if can_async_ag and len(self.bit16_groups_flat) == 1 and sc_now >= 2 and (sc_now != 3 or os.environ.get("WLA_WARM_RETUNE", "1") != "1"):
                for i in range(n_groups):
                    self._update_model_bit16_weights(i)
                part_params0 = self.parallel_partitioned_bit16_groups[0]
                dp_pg0 = self.real_dp_process_group[0]
                # Phase 1: Broadcast Vision weights (830.7 MB, ~2.1 ms) immediately on main stream
                dist.broadcast(part_params0[0][:vis_end_elems], src=ag_pg_ranks[0], group=dp_pg0)
                ev_adam_ready.record(torch.cuda.current_stream(dev))

                def _launch_phase2(target_stream):
                    target_stream.wait_event(ev_adam_ready)
                    with torch.cuda.stream(target_stream):
                        with dist._coalescing_manager(group=ag_pg, device=dev, async_ops=False):
                            if vis_end_elems < int(part_params0[0].numel()):
                                dist.broadcast(part_params0[0][vis_end_elems:], src=ag_pg_ranks[0], group=ag_pg)
                            for r in range(1, len(ag_pg_ranks)):
                                dist.broadcast(part_params0[r], src=ag_pg_ranks[r], group=ag_pg)

                bare_model._pending_non_vis_ag = _launch_phase2
            else:
                for i, (group_flat, part_params) in enumerate(
                    zip(self.bit16_groups_flat, self.parallel_partitioned_bit16_groups)
                ):
                    dp_pg = self.real_dp_process_group[i]
                    if dist.get_world_size(group=dp_pg) > 1:
                        part_id = dist.get_rank(group=dp_pg)
                        dist.all_gather_into_tensor(group_flat, part_params[part_id], group=dp_pg)
                for i in range(n_groups):
                    self._update_model_bit16_weights(i)
        finally:
            grad_accum_has_data[0] = False
            touched_part_ids.clear()

    bare_model = model_or_engine.module if hasattr(model_or_engine, "module") else model_or_engine
    enable_async_weight_ag = (os.environ.get("WLA_ASYNC_WEIGHT_AG", "1") == "1")
    can_async_ag = False
    vis_end_elems = 0
    ag_pg = None
    ag_pg_ranks: list[int] = []
    ev_adam_ready = torch.cuda.Event()

    if (
        enable_async_weight_ag
        and len(ds_opt.bit16_groups_flat) == 1
        and hasattr(bare_model, "qwen_vl_interface")
        and hasattr(bare_model.qwen_vl_interface.model.model, "visual")
    ):
        dp_pg0_init = ds_opt.real_dp_process_group[0]
        if dist.get_world_size(group=dp_pg0_init) > 1:
            ds_opt._update_model_bit16_weights(0)
            flat0 = ds_opt.bit16_groups_flat[0]
            part0_numel = int(ds_opt.parallel_partitioned_bit16_groups[0][0].numel())
            base_ptr = flat0.data_ptr()
            flat_numel = int(flat0.numel())
            vis_mod = bare_model.qwen_vl_interface.model.model.visual
            vis_ids = {id(p) for p in vis_mod.parameters() if p.requires_grad}
            max_vis = 0
            min_non_vis = flat_numel
            ok_layout = bool(vis_ids)
            for p in ds_opt.bit16_groups[0]:
                off = (p.data_ptr() - base_ptr) // 2
                end = off + int(p.numel())
                if off < 0 or end > flat_numel:
                    ok_layout = False
                    break
                if id(p) in vis_ids:
                    if end > max_vis:
                        max_vis = end
                else:
                    if off < min_non_vis:
                        min_non_vis = off
            if ok_layout and 0 < max_vis <= min_non_vis and max_vis <= part0_numel:
                vis_end_elems = max_vis
                ag_pg_ranks = dist.get_process_group_ranks(dp_pg0_init)
                ag_ctas = int(os.environ.get("WLA_AG_CTAS", "0"))
                ag_opts = dist.ProcessGroupNCCL.Options()
                ag_opts.is_high_priority_stream = False
                if ag_ctas > 0:
                    ag_opts.config.min_ctas = ag_ctas
                    ag_opts.config.max_ctas = ag_ctas
                ag_pg = dist.new_group(ranks=ag_pg_ranks, backend="nccl", pg_options=ag_opts)
                can_async_ag = True

                def _flush_pending_non_vis_ag():
                    deferred_q = getattr(bare_model, "_deferred_fp8_quant_fn", None)
                    if deferred_q is not None:
                        bare_model._deferred_fp8_quant_fn = None
                        ev_now = torch.cuda.Event()
                        ev_now.record(torch.cuda.current_stream(dev))
                        deferred_q(ev_now)
                    pending = getattr(bare_model, "_pending_non_vis_ag", None)
                    if pending is not None:
                        bare_model._pending_non_vis_ag = None
                        pending(torch.cuda.current_stream(dev))
                    wait_fp8_weight_quant(dev)

                bare_model._flush_pending_non_vis_ag = _flush_pending_non_vis_ag

    ds_opt.copy_grads_in_partition = types.MethodType(_fast_copy_grads_in_partition, ds_opt)
    ds_opt.reduce_ipg_grads = types.MethodType(_fast_reduce_ipg_grads, ds_opt)
    ds_opt.reduce_independent_p_g_buckets_and_remove_grads = types.MethodType(
        _fast_reduce_independent_p_g_buckets_and_remove_grads, ds_opt
    )
    ds_opt.independent_gradient_partition_epilogue = types.MethodType(
        _fast_independent_gradient_partition_epilogue, ds_opt
    )
    ds_opt.gradient_reduction_w_predivide = types.MethodType(_fast_gradient_reduction_w_predivide, ds_opt)
    ds_opt.get_grad_norm_direct = types.MethodType(_fast_get_grad_norm_direct, ds_opt)
    ds_opt.unscale_and_clip_grads = types.MethodType(_fast_unscale_and_clip_grads, ds_opt)
    ds_opt.step = types.MethodType(_fast_zero2_step, ds_opt)
    def load_state_dict(self, *args, **kwargs):
        result = orig_load_state_dict(*args, **kwargs)
        step_int_cache.clear()
        grad_accum_has_data[0] = False
        touched_part_ids.clear()
        direct_committed_pids.clear()
        in_vision_tail[0] = vision_tail_flushed[0] = False
        ipg_buf_in_flight[0] = ipg_buf_in_flight[1] = False
        for name in ("_pending_non_vis_ag", "_deferred_fp8_quant_fn"):
            if hasattr(bare_model, name):
                setattr(bare_model, name, None)
        clear_all_fp8_weights(bare_model, release_memory=True)
        return result

    ds_opt.load_state_dict = types.MethodType(load_state_dict, ds_opt)
    ds_opt._grad_accum_has_data = grad_accum_has_data
    ds_opt._step_int_cache = step_int_cache
    ds_opt._touched_part_ids = touched_part_ids
    _install_checkpoint_flush(model_or_engine)
    return {
        "installed": True,
        "nvls_deferred_div": use_deferred_div,
        "adaptive_vision_tail_flush": enable_adaptive_vision_flush,
        "fused_zero2_adamw": enable_fused_step,
        "direct_flat_part": can_direct_flat_part,
        "direct_ipg_dw": enable_direct_ipg,
        "async_two_stage_weight_ag": can_async_ag,
        "vis_end_elems": vis_end_elems,
    }
