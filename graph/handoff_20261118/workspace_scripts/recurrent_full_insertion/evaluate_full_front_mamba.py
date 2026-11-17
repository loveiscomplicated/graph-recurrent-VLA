"""Reuse the paired Mamba evaluator with the full-insertion frontend."""
import evaluate_mamba
from frontend import FullInsertionFrontend

if __name__=='__main__':
    evaluate_mamba.fixed.HybridFrontend=FullInsertionFrontend
    evaluate_mamba.main()
