<template>
  <div id="app">
    <h1>AI Webcam Prediction</h1>
    <CameraPredict />
  </div>
</template>

<script setup>
import CameraPredict from './components/CameraPredict.vue'
</script>

<style>
body {
  margin: 0;
  font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
  background: #f3f4f6;
  padding: 2rem;
}

#app {
  max-width: 1200px;
  margin: 0 auto;
}

h1 {
  text-align: center;
  color: #1f2937;
  font-size: 1.75rem;
  margin-bottom: 2rem;
}
</style>
