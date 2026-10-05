# References for the SWE-verifier / self-learning paper

Every entry was checked on its arXiv abstract page on 2026-10-05 (JST): title, authors, ID, date, and the exact
numbers we may quote. A short quote of the key sentence is kept so a reviewer can check our use of it. Anything not
on this list must be verified the same way before it goes into a draft.

| # | Reference | Checked | What we cite it for |
|---|---|---|---|
| 1 | Jiayi Pan, Xingyao Wang, Graham Neubig, Navdeep Jaitly, Heng Ji, Alane Suhr, Yizhe Zhang. **Training Software Engineering Agents and Verifiers with SWE-Gym.** arXiv:2412.21139 (v1 30 Dec 2024, v2 6 Jun 2025). https://arxiv.org/abs/2412.21139 | abstract | Verifiers trained on agent trajectories for inference-time scaling (selecting among trajectories). Quote: "verifiers trained on agent trajectories sampled from SWE-Gym". Numbers: 32.0% SWE-Bench Verified, 26.0% Lite (with their fine-tuned agents). |
| 2 | Mohit Raghavendra, Anisha Gunjal, Bing Liu, Yunzhong He. **Agentic Rubrics as Contextual Verifiers for SWE Agents.** arXiv:2601.04171 (7 Jan 2026). https://arxiv.org/abs/2601.04171 | abstract | Rubric-based verification signal for SWE agents. Numbers: 54.2% (Qwen3-Coder-30B-A3B) and 40.6% (Qwen3-32B) on SWE-Bench Verified, "at least a +3.5 percentage-point gain over the strongest baseline". |
| 3 | Chenyu Wang, Yunbo Lyu, Junda He, Zhou Yang, Chenxing Zhong, Yaniv Harel, David Lo. **Fail-Fast, Restart-Smart: Early Failure Prediction and Restart for SWE Agentic Tasks.** arXiv:2608.03222 (4 Aug 2026). https://arxiv.org/abs/2608.03222 | abstract | Early failure prediction from trajectory prefixes plus restarts: the closest prior work to our anytime study. "a lightweight 0.6B monitor"; "saves 14.6%-20.4% of execution tokens at a target 5% false-positive rate"; "raises Qwen3.6-27B resolution from 66.6% to 71.8%, whereas cold restart reaches only 66.8%". (The abstract evaluates at 5% and 25% false-positive targets; check the full text for which target the 71.8% uses before quoting it.) |
| 4 | Yuling Shi, Zhensu Sun, Junsen Dong, Chengcheng Wan, David Lo, Xiaodong Gu. **EarlyEval: Cheaper Agent Evaluation via Early Outcome Prediction.** arXiv:2609.02783 (2 Sep 2026). https://arxiv.org/abs/2609.02783 | abstract | Early outcome prediction with LightGBM classifiers. "eliminate 13%-26% of agent steps and up to 44.1% input tokens and 29.4% output tokens at 89%-97% prediction accuracy". |
| 5 | KaShun Shum, Binyuan Hui, Jiawei Chen, Lei Zhang, X. W., Jiaxi Yang, Yuzhen Huang, Junyang Lin, Junxian He. **SWE-RM: Execution-free Feedback For Software Engineering Agents.** arXiv:2512.21919. https://arxiv.org/abs/2512.21919 | abstract | A 30B MoE reward model for SWE agents that treats classification accuracy and calibration as crucial for RL: calibration of SWE reward models is not new. (Author "X. W." as listed on the page; confirm the full name from the PDF before printing.) |
| 6 | Binghai Wang, Chenlong Zhang, Dayiheng Liu, Jiajun Zhang, Jiawei Chen, Mingze Li, Mouxiang Chen, Rongyao Fang, Siyuan Zhang, Xuwu Wang, Yuheng Jing, Zeyao Ma, Zeyu Cui. **The Verification Horizon: No Silver Bullet for Coding Agent Rewards.** arXiv:2606.26300. https://arxiv.org/abs/2606.26300 | abstract | Motivation for adapting verifiers: "no fixed reward function can remain effective as policy capability continues to grow"; verification must co-evolve with the generator. |
| 7 | Ken Huang, Jerry Huang. **Audited Skill-Graph Self-Improvement for Agentic LLMs via Verifiable Rewards, Experience Synthesis, and Continual Memory.** arXiv:2512.23760 (28 Dec 2025). https://arxiv.org/abs/2512.23760 | abstract | Self-improvement with verifier-gated promotion: "promoted only after passing verifier-backed replay and contract checks". |
| 8 | Diandian Guo, Cong Cao, Fangfang Yuan, Yingqi Wang, Yueshan Wang, Dakui Wang. **Self-Authored Verification Is Unreliable in Heuristic Self-Improving Agents.** arXiv:2607.24300 (27 Jul 2026). https://arxiv.org/abs/2607.24300 | abstract | The verifier-deployment gap and SEAL, an external acceptance gate: "compares each candidate with the incumbent through a fixed harness-side audit". The closest prior work to our promotion gate; ours differs by using real outcomes and a repository-bootstrap interval on held-out log loss. |
| 9 | John C. Platt. **Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods.** In *Advances in Large Margin Classifiers*, MIT Press, 1999. | standard reference; confirm the page numbers before printing | Platt scaling, used by `OutcomeCalibrator`. |

## Data

| Dataset | Licence | Use |
|---|---|---|
| nebius/SWE-agent-trajectories | CC-BY-4.0 | training and evaluation (SWE-agent runs with `target`); `eval_logs` never read |
| nebius/SWE-rebench-openhands-trajectories | CC-BY-4.0 | the new agent for adaptation (planned) |
| SWE-bench/SWE-smith-trajectories | MIT | the third dataset for the leakage study (planned) |

## Not yet verified
- Full-text claims beyond the abstracts: for example, whether SWE-RM or SWE-Gym split by repository. Check the PDFs before writing "no prior work splits by repository".
