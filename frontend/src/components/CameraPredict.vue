<template>
  <div class="p-4">
    <video ref="video" autoplay playsinline width="640" height="480"></video>
    <canvas ref="canvas" width="640" height="480" style="display: none;"></canvas>

    <button @click="toggleStream" class="px-4 py-2 bg-blue-500 text-white rounded mt-2">
      {{ streaming ? 'Stop' : 'Start' }} Pose Stream
    </button>

    <div v-if="frame" class="mt-4">
      <img :src="'data:image/jpeg;base64,' + frame" width="640" class="rounded shadow" />
    </div>

    <p v-if="error" class="text-red-500 mt-2">{{ error }}</p>
  </div>
</template>

<script setup>
import { ref, onBeforeUnmount } from 'vue'

const video = ref(null)
const canvas = ref(null)
let socket = null
const streaming = ref(false)
const frame = ref(null)
const error = ref(null)

let lastSent = 0
const SEND_INTERVAL = 100 // ms (~10 FPS)

const toggleStream = async () => {
  if (!streaming.value) {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true })
    video.value.srcObject = stream

    socket = new WebSocket("ws://localhost:8000/ws/pose") // change host if needed

    socket.onopen = () => {
      streaming.value = true
      error.value = null
      requestAnimationFrame(sendFrames)
    }

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data)
      frame.value = data.image
    }

    socket.onerror = (err) => {
      error.value = "WebSocket connection failed."
      console.error(err)
    }

    socket.onclose = () => {
      streaming.value = false
    }
  } else {
    socket.close()
    streaming.value = false
  }
}

const sendFrames = (timestamp) => {
  if (!streaming.value || !socket || socket.readyState !== WebSocket.OPEN) return

  if (timestamp - lastSent >= SEND_INTERVAL) {
    lastSent = timestamp

    const tempCanvas = canvas.value
    tempCanvas.width = 320 // reduce resolution
    tempCanvas.height = 240
    const ctx = tempCanvas.getContext("2d")
    ctx.drawImage(video.value, 0, 0, tempCanvas.width, tempCanvas.height)
    const dataUrl = tempCanvas.toDataURL("image/jpeg", 0.5) // lower quality
    socket.send(JSON.stringify({ image: dataUrl }))
  }

  requestAnimationFrame(sendFrames)
}

onBeforeUnmount(() => {
  if (socket) socket.close()
  const stream = video.value?.srcObject
  if (stream) stream.getTracks().forEach(track => track.stop())
})
</script>
