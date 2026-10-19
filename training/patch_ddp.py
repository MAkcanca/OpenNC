"""Make df/train_ddp.py from df/train.py: torchrun data-parallel training (NCCL), rank-0 validation/checkpoints/logs,
per-rank dataloader seeds, epoch split across ranks via global_sampling_factor / WORLD. Usage: patch_ddp.py <site-packages/df>"""
import sys, os
d = sys.argv[1]; src = open(os.path.join(d, 'train.py')).read()
def rep(old, new, count=1):
    global src
    assert src.count(old) == count, (src.count(old), old[:60]); src = src.replace(old, new)
rep('should_stop = False\n', 'should_stop = False\nRANK = 0\nWORLD = 1\n')
rep('    init_logger(file=os.path.join(args.base_dir, "train.log"), level=log_level, model=args.base_dir)\n',
    '    global RANK, WORLD\n    if "RANK" in os.environ:\n        import torch.distributed as dist\n        dist.init_process_group("nccl"); RANK, WORLD = dist.get_rank(), dist.get_world_size(); torch.cuda.set_device(int(os.environ["LOCAL_RANK"])); os.environ["DF_DEVICE"] = f"cuda:{os.environ[\'LOCAL_RANK\']}"\n'
    '    init_logger(file=os.path.join(args.base_dir, "train.log" if RANK == 0 else f"train_rank{RANK}.log"), level=log_level if RANK == 0 else "ERROR", model=args.base_dir)\n')
rep('        num_workers=config("NUM_WORKERS", 4, int, section="train"),\n', '        num_workers=max(1, config("NUM_WORKERS", 4, int, section="train") // WORLD),\n')
rep('        seed=seed,\n', '        seed=seed + RANK,\n')
rep('        global_sampling_factor=config("GLOBAL_DS_SAMPLING_F", 1.0, float, section="train"),\n', '        global_sampling_factor=config("GLOBAL_DS_SAMPLING_F", 1.0, float, section="train") / WORLD,\n')
rep('    if jit:\n        # Load as jit after log_model_summary\n        model = torch.jit.script(model)\n',
    '    if jit:\n        # Load as jit after log_model_summary\n        model = torch.jit.script(model)\n    raw_model = model\n    if WORLD > 1:\n        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()], broadcast_buffers=False, find_unused_parameters=True)\n')
rep('    config.save(os.path.join(args.base_dir, "config.ini"))\n', '    if RANK == 0:\n        config.save(os.path.join(args.base_dir, "config.ini"))\n')
rep('        log_metrics(f"[{epoch}] [train]", metrics)\n        write_cp(model, "model", checkpoint_dir, epoch + 1)\n        write_cp(opt, "opt", checkpoint_dir, epoch + 1)\n',
    '        log_metrics(f"[{epoch}] [train]", metrics)\n        if RANK == 0:\n            write_cp(raw_model, "model", checkpoint_dir, epoch + 1)\n            write_cp(opt, "opt", checkpoint_dir, epoch + 1)\n')
old_val = '''        val_loss = run_epoch(
            model=model,
            epoch=epoch,
            loader=dataloader,
            split="valid",
            opt=opt,
            losses=losses,
            summary_dir=summary_dir,
        )
        metrics = {"loss": val_loss}
        metrics.update(
            {n: torch.mean(torch.stack(vals)).item() for n, vals in losses.get_summaries()}
        )
        val_criteria = metrics[val_criteria_type]
        write_cp(
            model, "model", checkpoint_dir, epoch + 1, metric=val_criteria, cmp=val_criteria_rule
        )
        log_metrics(f"[{epoch}] [valid]", metrics)
        if not check_patience(
            checkpoint_dir,
            max_patience=patience,
            new_metric=val_criteria,
            cmp=val_criteria_rule,
            raise_=False,
        ):
            break
        if should_stop:
            logger.info("Stopping training due to timeout")
            exit(0)
        losses.reset_summaries()
'''
new_val = '''        stop = torch.zeros(1, device=get_device())
        if RANK == 0:
            val_loss = run_epoch(model=raw_model, epoch=epoch, loader=dataloader, split="valid", opt=opt, losses=losses, summary_dir=summary_dir)
            metrics = {"loss": val_loss}
            metrics.update({n: torch.mean(torch.stack(vals)).item() for n, vals in losses.get_summaries()})
            val_criteria = metrics[val_criteria_type]
            write_cp(raw_model, "model", checkpoint_dir, epoch + 1, metric=val_criteria, cmp=val_criteria_rule)
            log_metrics(f"[{epoch}] [valid]", metrics)
            if not check_patience(checkpoint_dir, max_patience=patience, new_metric=val_criteria, cmp=val_criteria_rule, raise_=False) or should_stop:
                stop += 1
        if WORLD > 1:
            torch.distributed.broadcast(stop, 0)
        if stop.item() > 0:
            if should_stop:
                logger.info("Stopping training due to timeout")
            break
        losses.reset_summaries()
'''
rep(old_val, new_val)
rep('    model, epoch = load_model(\n        checkpoint_dir,\n        state,\n        jit=jit,\n        epoch="best",\n',
    '    if RANK != 0:\n        return\n    model, epoch = load_model(\n        checkpoint_dir,\n        state,\n        jit=jit,\n        epoch="best",\n')
rep('    seed = epoch if is_train else 42\n', '    seed = (epoch * WORLD + RANK) if is_train else 42\n')
rep('            log_metrics(f"[{epoch}] [{step}/{max_steps}]", l_dict)\n            summary_write(\n', '            log_metrics(f"[{epoch}] [{step}/{max_steps}]", l_dict)\n            if RANK == 0: summary_write(\n')
open(os.path.join(d, 'train_ddp.py'), 'w').write(src); print('wrote', os.path.join(d, 'train_ddp.py'))
