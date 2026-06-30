<template>
  <div class="camera-predict">
    <div class="main-layout">
      <!-- Left: Webcam + Controls -->
      <div class="panel">
        <h2>Live Webcam</h2>
        <div class="video-container">
          <div v-if="!streaming" class="placeholder">
            <span>Press Start to begin</span>
          </div>
          <div class="video-wrapper" :class="{ hidden: !streaming }">
            <video ref="video" autoplay playsinline muted class="webcam-feed"></video>
          </div>
        </div>

        <div class="controls">
          <button @click="toggleStream" :class="['btn', streaming ? 'btn-stop' : 'btn-start']">
            {{ streaming ? '⏹ Stop Stream' : '▶ Start Stream' }}
          </button>
        </div>
        <p v-if="error" class="error">{{ error }}</p>
      </div>

      <!-- Right: Classification Result -->
      <div class="panel">
        <h2>Pose Classification</h2>

        <div v-if="predictedClass" class="result-card">
          <div class="class-badge" :class="'class-' + predictedClass">
            <span class="class-icon">{{ classIcon }}</span>
            <span class="class-name">{{ predictedClass }}</span>
          </div>

          <div class="confidence-bar">
            <div class="confidence-label">Confidence</div>
            <div class="bar-track">
              <div class="bar-fill" :style="{ width: (confidence * 100) + '%' }"></div>
            </div>
            <div class="confidence-value">{{ (confidence * 100).toFixed(0) }}%</div>
          </div>

          <div class="reference-section">
            <div class="ref-label">Reference pose:</div>
            <img :src="referenceImage" class="reference-img" />
          </div>
        </div>

        <div v-else class="idle-state">
          <div class="idle-icon">🤸</div>
          <p>Waiting for pose detection...</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onBeforeUnmount } from 'vue'

const video = ref(null)
const canvas = document.createElement('canvas')
let socket = null
const streaming = ref(false)
const predictedClass = ref(null)
const confidence = ref(0)
const error = ref(null)

let lastSent = 0
const SEND_INTERVAL = 150 // ms (~6.5 FPS)

const classEmoji = {
  'heart-attack': '💔',
  'idea': '💡',
  'stand': '🧍',
  'think': '🤔'
}

const classIcon = computed(() => classEmoji[predictedClass.value] || '❓')

const referenceImage = computed(() => {
  if (!predictedClass.value) return ''
  return `/static/poses/${predictedClass.value}.jpeg`
})

const toggleStream = async () => {
  if (!streaming.value) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true })
      video.value.srcObject = stream
      video.value.play()

      socket = new WebSocket(`ws://${window.location.host}/ws/pose`)

      socket.onopen = () => {
        streaming.value = true
        error.value = null
        predictedClass.value = null
        requestAnimationFrame(sendFrames)
      }

      socket.onmessage = (event) => {
        const data = JSON.parse(event.data)
        if (data.class) {
          predictedClass.value = data.class
          confidence.value = data.confidence
        }
      }

      socket.onerror = () => {
        error.value = "WebSocket connection failed. Is the backend running?"
      }

      socket.onclose = () => {
        streaming.value = false
      }
    } catch (err) {
      error.value = "Camera access denied or unavailable."
    }
  } else {
    if (socket) socket.close()
    streaming.value = false
  }
}

const sendFrames = (timestamp) => {
  if (!streaming.value || !socket || socket.readyState !== WebSocket.OPEN) return

  if (timestamp - lastSent >= SEND_INTERVAL) {
    lastSent = timestamp
    canvas.width = 320
    canvas.height = 240
    const ctx = canvas.getContext("2d")
    ctx.drawImage(video.value, 0, 0, canvas.width, canvas.height)
    const dataUrl = canvas.toDataURL("image/jpeg", 0.5)
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

<style scoped>
.camera-predict {
  max-width: 1100px;
  margin: 0 auto;
  font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
}

.main-layout {
  display: flex;
  gap: 24px;
  justify-content: center;
  flex-wrap: wrap;
}

.panel {
  background: white;
  border-radius: 16px;
  padding: 24px;
  box-shadow: 0 4px 24px rgba(0,0,0,0.08);
  flex: 1;
  min-width: 340px;
  max-width: 520px;
}

.panel h2 {
  margin: 0 0 16px;
  font-size: 1.1rem;
  color: #374151;
  font-weight: 600;
}

.video-container {
  background: #1a1a2e;
  border-radius: 12px;
  overflow: hidden;
  aspect-ratio: 4/3;
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 240px;
}

.video-wrapper {
  width: 100%;
  height: 100%;
}

.webcam-feed {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
}

.video-wrapper.hidden {
  display: none;
}

.placeholder {
  color: #6b7280;
  font-size: 1rem;
}

.controls {
  margin-top: 12px;
}

.btn {
  padding: 12px 32px;
  border: none;
  border-radius: 10px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
  width: 100%;
}

.btn-start {
  background: #2563eb;
  color: white;
}
.btn-start:hover {
  background: #1d4ed8;
}

.btn-stop {
  background: #dc2626;
  color: white;
}
.btn-stop:hover {
  background: #b91c1c;
}

.error {
  color: #dc2626;
  margin-top: 8px;
  font-size: 0.9rem;
}

/* Classification Result */
.result-card {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.class-badge {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 20px;
  border-radius: 14px;
  font-size: 1.6rem;
  font-weight: 700;
  justify-content: center;
}

.class-icon {
  font-size: 2.2rem;
}

.class-name {
  text-transform: capitalize;
}

.class-heart-attack {
  background: #fef2f2;
  color: #991b1b;
  border: 2px solid #fecaca;
}

.class-idea {
  background: #fffbeb;
  color: #92400e;
  border: 2px solid #fde68a;
}

.class-stand {
  background: #f0fdf4;
  color: #166534;
  border: 2px solid #bbf7d0;
}

.class-think {
  background: #f3e8ff;
  color: #581c87;
  border: 2px solid #d8b4fe;
}

.confidence-bar {
  display: flex;
  align-items: center;
  gap: 12px;
}

.confidence-label {
  font-size: 0.85rem;
  color: #6b7280;
  min-width: 75px;
}

.bar-track {
  flex: 1;
  height: 10px;
  background: #e5e7eb;
  border-radius: 5px;
  overflow: hidden;
}

.bar-fill {
  height: 100%;
  background: linear-gradient(90deg, #22c55e, #16a34a);
  border-radius: 5px;
  transition: width 0.3s ease;
}

.confidence-value {
  font-size: 0.95rem;
  font-weight: 700;
  color: #374151;
  min-width: 48px;
  text-align: right;
}

.reference-section {
  text-align: center;
}

.ref-label {
  font-size: 0.85rem;
  color: #6b7280;
  margin-bottom: 8px;
}

.reference-img {
  width: 100%;
  max-height: 220px;
  object-fit: contain;
  border-radius: 10px;
  border: 2px solid #e5e7eb;
  background: #f9fafb;
}

.idle-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 60px 20px;
  color: #9ca3af;
}

.idle-icon {
  font-size: 4rem;
  margin-bottom: 12px;
}

.idle-state p {
  font-size: 1rem;
}
</style>
