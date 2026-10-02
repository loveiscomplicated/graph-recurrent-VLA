"""Small helpers shared by the ManiSkill benchmark and GT extractor."""


def peg_task(env):
    """Return the raw PegInsertionSide task from a Gymnasium wrapper."""
    task = env.unwrapped
    if task.__class__.__name__ != "PegInsertionSideEnv":
        raise TypeError("Expected a PegInsertionSide-v1 environment")
    return task
