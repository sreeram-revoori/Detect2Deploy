"""
nurosim.deploy
Model deployment pipeline: synthetic dataset → train → ONNX export →
FP16 / INT8 variants → parity check → latency benchmark → budget gate.

Run `python -m nurosim.deploy --help` for the CLI.
"""
