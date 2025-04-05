### Training

The idea: maybe we want to have our own gestures. We have 3 options:
1) Hard code the detection in MaxMSP
2) Send the OSC landmarks to Wekinator and classify there, sending the result to MaxMSP
3) Train our own model directly with MediaPipe.

The following is an initial attempt at (3). If it proves too time-consuming will fall back to other alternatives.

Following https://ai.google.dev/edge/mediapipe/solutions/customization/gesture_recognizer. Taking some inspiration from https://github.com/Thomas9363/How-to-Train-Custom-Hand-Gestures-Using-Mediapipe/tree/main

### Customizing the gesture model

I want to use the `mediapipe-model-maker` package to update the models to identify additional gestures. However I ran into a few problems.

### Problem 1: cmake

cmake is apparently a requirement for the model-maker. "pip install cmake" did not do the job. What ended up working:

```shell
pip uninstall cmake
arch -arm64 brew install cmake
pip install dlib
```

### Problem 2: New deps

The vanilla pip install doesn't work with the model-maker because it hasn't been released recently with an updated version, and it requires older version of multiple packages:

```
pip install mediapipe-model-maker
```
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
mediapipe-model-maker 0.2.1.4 requires opencv-python, which is not installed.
mediapipe-model-maker 0.2.1.4 requires tensorflow-addons, which is not installed.
mediapipe-model-maker 0.2.1.4 requires tensorflow<2.16,>=2.10, but you have tensorflow 2.18.1 which is incompatible.
mediapipe-model-maker 0.2.1.4 requires tensorflow-model-optimization<0.8.0, but you have tensorflow-model-optimization 0.8.0 which is incompatible.
mediapipe-model-maker 0.2.1.4 requires tf-models-official<2.16.0,>=2.13.2, but you have tf-models-official 2.18.0 which is incompatible.
```


Problem solved thanks to comment here: https://github.com/google-ai-edge/mediapipe/issues/5214#issuecomment-2334011576

```shell
pyenv install 3.9.16
pyenv local 3.9.16
pip install "pyyaml>6.0.0" "keras<3.0.0" "tensorflow<2.16" "tf-models-official<2.16" mediapipe-model-maker --no-deps
```
