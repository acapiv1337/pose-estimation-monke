<template>
  <div id="app">
    <CameraPredict />
  </div>
</template>

<script setup>
import CameraPredict from './components/CameraPredict.vue'
</script>

<style>
@import url('https://fonts.googleapis.com/css2?family=Comic+Neue:wght@400;600;700&display=swap');

body {
  margin: 0;
  padding: 0;
  font-family: 'Comic Neue', 'Comic Sans MS', cursive, sans-serif;
  overflow: hidden;
}

#app {
  width: 100%;
  height: 100vh;
  position: relative;
}

#app::before {
  content: '';
  position: fixed;
  inset: 0;
  background: url('/static/bg.jpeg') center/cover no-repeat;
  z-index: -1;
}
</style>
