# USB-Rubber-Ducky
A Python-Based Emulator for Keystroke Injection and Liveness Detection Research

Overview

Keystroke dynamics offer a promising way to distinguish genuine human typing from potentially harmful automated activity. This project investigates keystroke injection attacks by mimicking human-like behavior using pseudorandom number generators (PRNGs) and generative adversarial networks (GANs).
The central part is a Python-based USB Rubber Ducky emulator that simulates keystroke sequences with realistic timing, following commands in a Ducky Script-like syntax. These synthetic sequences are evaluated by training various classifiers to detect machine input and testing their robustness against adversarial examples.



Objectives
- Emulate keystroke injection attacks with realistic delays and jitter
- Train machine learning classifiers to distinguish machine vs. human typing
- Use GANs to produce adversarial input that mimics human behavior
- Evaluate the resilience of keystroke-based liveness detection systems



Features
- USB Rubber Ducky emulator with customizable typing behavior
- Ducky Script-style commands
- 4 PRNG options for timing randomness: Philox, Xoroshiro128, Mersenne Twister, PCG64
- Real human keystroke dataset integration
- ML classifiers: Random Forest, XGBoost, SVM, k-NN
- GAN-based input generation for adversarial robustness testing



Research Impact

This work demonstrates that standard classifiers are effective at spotting naive injection attacks—but are vulnerable to adversarial imitation via GANs. It highlights the need for robust liveness detection methods capable of withstanding generative attacks in keystroke-based security systems.



Author

Martina Culpo

[🔗 GitHub](https://github.com/MartinaCulpo)



Disclaimer

This project is intended for academic research and ethical use only. Unauthorized use or deployment of input injection tools is strictly prohibited. The author assumes no responsibility for misuse.

