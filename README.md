# Bayesian-Holography
Simple python apps that show Bayesiann inference against a dynamic holographic file

Find the two programs you need in the Src folder.
Check out this blog post for detailed informaion: https://connectomic-ai.blogspot.com/2025/05/unveiling-dynamic-holographic-changes.html

The Python script acts as a continuous monitor for a file named holo.npy. It performs the following key functions:

File Watching: It periodically checks holo.npy for modifications.

Data Loading & Analysis: If the file changes, it reloads the entire dataset.

Bayesian Inference: It applies sequential Bayesian inference to estimate the underlying mean of the holographic data.

Change Detection: It heuristically identifies significant shifts in this inferred mean, indicating a change in the holographic file's characteristics.

Continuous Plotting: It updates a live plot, showing the raw data, the inferred mean, and its uncertainty, allowing for real-time visualization of the analysis.
