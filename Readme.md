# SIRI-Agent: Visual Imitation Learning for CODM

SIRI (Swift Insight Reinforce-Imitate) Agent is an AI system that performs visual imitation learning in the mobile game CODM (Call of Duty: Mobile).

## Demos
### SIRI-Agent Gameplay Demonstrations
1. [Bilibili Video](https://www.bilibili.com/video/BV1usE4zdERM)

 [![Bilibili视频封面](https://i0.hdslb.com/bfs/archive/7ad194e6189a2d9f951f6f4a1ae42de45cd7a4ec.jpg@672w_378h_1c.avif)](https://www.bilibili.com/video/BV1usE4zdERM)


2. Combat Situation 1

<img src="assets/demo1_preview.gif" width="600"/>

3. Combat Situation 2 (short)

<img src="assets/demo2_preview.gif" width="600"/>

4. Combat Situation 3 (short)

<img src="assets/demo3_preview.gif" width="600"/>

### Rule-based Method Comparison
Below is a demonstration of a rule-based approach using behavior trees and MiDaS depth information for navigation. While this method provides basic functionality, it shows limitations compared to our imitation learning approach:

![Rule-based Navigation Demo](assets/rule-based_method_demo.gif)

## Introduction
SIRI-Agent addresses the challenge of implementing AI agents in mobile games without API access. Through pure visual imitation learning, it successfully replicates human gameplay in CODM while meeting real-time performance requirements.

## Dataset
![Dataset Overview](assets/PPT-Dataset.png)
Our comprehensive dataset includes:
- Total Size: 240 GiB
- Duration: ~8.15 hours
- Multiple scenarios and gameplay styles
- Carefully curated to ensure broad state coverage

## Model Architecture
We propose two network architectures:

### Lightweight Architecture
![Lightweight Network](assets/PPT-NN1.png)
Our lightweight architecture focuses on real-time performance while maintaining accurate predictions.

### Enhanced Architecture
![Enhanced Network with Feature Fusion](assets/PPT-NN2.png)
The enhanced architecture combines:
- EfficientNet for efficient feature extraction
- MiDaS for depth perception
- Feature fusion for improved understanding

## Features
- Pure vision-based imitation learning without API dependencies
- Real-time gameplay operation capabilities
- Lightweight model architecture for efficient processing
- Comprehensive state coverage through carefully curated datasets
- Mitigation of behavioral cloning cumulative error

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.