# FP32 Gemmini target

Target-owned metadata and execution selection for chipyard.FPGemminiRocketConfig.
Shared mechanisms stay in Merlin. Select the FP32 facts and isolated Spike extension
explicitly; never borrow the integer configuration's facts or installed extension.
Nothing in these resources launches or seals an experiment.
