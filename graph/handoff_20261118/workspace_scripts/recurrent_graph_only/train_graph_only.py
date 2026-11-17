"""Train the same BC policy with Graph-only observations."""
import train_mamba


if __name__=='__main__':
    # Keep five sampling levels so the episode sampling RNG matches the mixed run.
    # Every level now supplies the Graph observation.
    train_mamba.ALPHAS=(1.,)*len(train_mamba.ALPHAS)
    train_mamba.main()
