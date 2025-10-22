<template>
  <div>
    <video ref="video" autoplay playsinline width="640" height="480"></video>
    <canvas ref="canvas" width="640" height="480" style="display: none"></canvas>

    <div style="margin-top: 1rem;">
      <button @click="toggleCamera">{{ cameraOn ? 'Stop' : 'Start' }} Camera</button>
      <button v-if="cameraOn" @click="predict">Predict Frame</button>
    </div>

    <div v-if="prediction">
      <h3>Prediction Result</h3>
      <pre>{{ prediction }}</pre>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const video = ref(null)
const canvas = ref(null)
const prediction = ref(null)
const cameraOn = ref(false)

const toggleCamera = async () => {
  if (!cameraOn.value) {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true })
    video.value.srcObject = stream
    cameraOn.value = true
  } else {
    const stream = video.value.srcObject
    const tracks = stream.getTracks()
    tracks.forEach(track => track.stop())
    cameraOn.value = false
  }
}

const predict = async () => {
  const ctx = canvas.value.getContext('2d')
  ctx.drawImage(video.value, 0, 0, canvas.value.width, canvas.value.height)
  const blob = await new Promise(resolve => canvas.value.toBlob(resolve, 'image/jpeg'))

  const formData = new FormData()
  formData.append('frame', blob, 'frame.jpg')

  const res = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    body: formData
  })

  prediction.value = await res.json()
}
</script>
