# distzo2paper

## Nsight-based Runtime Trace Analysis

We conduct four sets of Nsight-based runtime trace analyses to evaluate the effectiveness of our communication strategy and parallelization design.

---

### Experiment 1: With vs. Without Section 6.1 Strategy

![PP+DP With and Without Communication Strategy](figs/comm.png)

**Figure 1**: Trace comparison of DistZO2 with and without the communication strategy proposed in Section 6.1.  
We run DistZO2 using PP + DP parallelism on 4 GPUs while fine-tuning OPT-1.3B. Panel (a) applies our Section 6.1 strategy—parameter slicing with PCIe+NVLink hybrid upload and distributed offload. Panel (b) uses a naive baseline where each GPU redundantly pulls and pushes the full parameter block via PCIe.  
Each row shows four key CUDA streams per GPU: forward compute (blue), CPU→GPU upload (red), inter-GPU slice exchange (green), and GPU→CPU offload (cyan). **Each repeated block in the timeline corresponds to the processing of one transformer block.**

#### Observations:
- With our strategy (panel a), memory streams are shorter and tightly overlapped with compute.
- Without the strategy (panel b), memory transfers (especially upload and offload) are prolonged and serialized.
- Overlap between communication and compute is significantly reduced in the naive version, resulting in under-utilized GPUs.

#### Conclusion:
The Section 6.1 communication strategy reduces PCIe pressure and restores pipeline overlap via parameter slicing and NVLink peer exchange, enabling high-throughput fine-tuning with minimal stalls.

---

### Experiment 2: NVLink vs. PCIe Interconnect

![PP vs DP with and without NVLink](figs/nccl.png)

**Figure 2**: Nsight trace visualization of DistZO2’s communication and computation pipeline under different interconnect settings.  
We compare Perturbation Parallelism (PP) and Distributed Data Parallelism (DP) under NVLink (left column) versus PCIe-only (right column) connections, on a 2-GPU system fine-tuning OPT-1.3B.  
Each GPU displays CUDA streams for forward compute (blue), CPU-to-GPU upload (Stream 14, red), GPU-to-GPU exchange (Stream 26, green), and GPU-to-CPU offload (Stream 22, cyan). **Each repeated segment corresponds to one transformer block.**

#### Observations:
- NVLink-based setups (left) show significantly shorter green segments (Stream 26), indicating faster peer-to-peer transfer and improved overlap.
- PCIe-only setups (right) suffer from longer upload and offload phases, with limited overlap and more visible stalls.
- Full overlap between compute and memory is only achieved when NVLink is enabled and parameter slicing is used.

#### Conclusion:
Using NVLink with parameter slicing drastically reduces inter-GPU communication time, allowing DistZO2 to maintain high GPU utilization and throughput. This validates the hardware-aware communication strategy introduced in Section 6.1.

---
