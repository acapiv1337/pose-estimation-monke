<template>
  <div id="app">
    <header>
      <h1>🖐️ Pose Classification</h1>
      <p class="subtitle">Real-time webcam pose estimation + XGBoost classifier</p>
    </header>
    <CameraPredict />
  </div>
</template>

<script setup>
import CameraPredict from './components/CameraPredict.vue'
</script>

<style>
* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  min-height: 100vh;
  padding: 2rem 1rem;
}

#app {
  max-width: 1200px;
  margin: 0 auto;
}

header {
  text-align: center;
  margin-bottom: 24px;
}

header h1 {
  color: white;
  font-size: 2rem;
  font-weight: 700;
  text-shadow: 0 2px 8px rgba(0,0,0,0.2);
}

.subtitle {
  color: rgba(255,255,255,0.85);
  font-size: 1rem;
  margin-top: 4px;
}
</style>
