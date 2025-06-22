# ProGuard rules for TensorFlow Lite
# Keep all TensorFlow Lite classes
-keep class org.tensorflow.** { *; }
# Do not warn about missing TensorFlow Lite classes
-dontwarn org.tensorflow.** 