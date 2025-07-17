from __future__ import annotations


def robust_save_pretrained(model, out_dir, state_dict, save_fn):
    """Attempt safe and standard serialization when saving HuggingFace models."""
    for safe in (True, False):
        try:
            model.save_pretrained(
                out_dir,
                safe_serialization=safe,
                state_dict=state_dict,
                save_function=save_fn,
            )
            return
        except Exception:
            if not safe:
                raise
