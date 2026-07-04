<template>
  <div id="app" v-if="authenticated">
    <CameraPredict />
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import CameraPredict from './components/CameraPredict.vue'

const PORTFOLIO_PASSWORD = import.meta.env.VITE_PORTFOLIO_PASSWORD || 'monke2024'
const authenticated = ref(false)

onMounted(() => {
  const stored = sessionStorage.getItem('portfolio_auth')
  if (stored === 'true') {
    authenticated.value = true
    return
  }
  const input = prompt('Enter password to view this demo:')
  if (input === PORTFOLIO_PASSWORD) {
    sessionStorage.setItem('portfolio_auth', 'true')
    authenticated.value = true
  } else {
    document.body.innerHTML = '<pre>{\n  "error": "Access Denied",\n  "message": "Invalid password."\n}</pre>'
  }
})
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
