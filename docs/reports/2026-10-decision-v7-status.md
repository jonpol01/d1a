# Status on decision-v7, v0.2 era

The status table the README carried until October 2026: accuracy on the development partitions of decision-v7 for the first D1A checkpoints.

Early. On the development partitions of decision-v7:

| Model | Base | Accuracy: Trained Sources (dev) | Accuracy: New Sources (dev) |
|---|---|---|---|
| D1A-E4B (JohnP1/d1a-e4b), 2 epochs | Gemma-4-E4B | 0.853 | 0.680 |
| D1A-E2B v0.2 (JohnP1/d1a-e2b), 2 epochs | Gemma-4-E2B | 0.824 | 0.602 |
| D1A-E2B v0.1, 1 epoch | Gemma-4-E2B | 0.794 | 0.569 |

D1A-E4B is the most accurate so far, ahead of Jev (0.845) on sources it trained on and behind it (0.857) on new ones. Two epochs raised the calibration error from 0.040 to about 0.057 on both sizes, which is being looked at. E4B needed half E2B's learning rate (5e-5): at 1e-4 it diverged mid-epoch.
