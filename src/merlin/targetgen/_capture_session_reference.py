"""Independent pre-quantization trajectory for an audited FP32 stage."""


def freeze_fp32_session_reference(module, inputs, session):
    """Freeze the staged program's trajectory, not its later integerized output.

    The caller must first complete the frontend FP32 precision audit. This is
    a stage-local numerical reference over the declared inputs and recurrence;
    it does not establish a complete application's FP32 accuracy.
    """
    if session is None:
        return None
    import torch
    from m2m.capture.bundle import capture_session_trajectory

    streams = []
    for stream in session.get("streams") or ():
        value = stream.get("values")
        value = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        index = int(stream["input_index"])
        if not 0 <= index < len(inputs):
            raise ValueError("session stream references an unknown staged input")
        # Match the executable input ABI, preserving nonfloating leaves exactly.
        value = value.detach().clone()
        if value.is_floating_point():
            value = value.to(dtype=inputs[index].dtype, device=inputs[index].device)
        elif value.dtype != inputs[index].dtype:
            raise ValueError("session stream changes a nonfloating input dtype")
        streams.append({**stream, "values": value})
    session = {**session, "streams": streams}
    quality = dict(session.get("quality") or {})
    quality.update(
        reference="eager_fp32",
        reference_values=capture_session_trajectory(module, tuple(inputs), session).copy(),
    )
    return {**session, "quality": quality}
