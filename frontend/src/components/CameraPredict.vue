<template>
  <div class="camera-predict">
    <!-- Mode tabs -->
    <div class="mode-tabs">
      <button @click="mode = 'webcam'" :class="['tab', { active: mode === 'webcam' }]">
        📷 Webcam
      </button>
      <button @click="mode = 'upload'" :class="['tab', { active: mode === 'upload' }]">
        📁 Upload Image
      </button>
    </div>

    <div class="split">
      <!-- Left: Webcam or Uploaded Image -->
      <div class="cam-half">
        <!-- Webcam mode -->
        <template v-if="mode === 'webcam'">
          <div v-if="!streaming" class="placeholder">
            <div class="placeholder-content">
              <div class="placeholder-icon">📷</div>
              <span>Press Start to begin</span>
            </div>
          </div>
          <div class="video-wrapper" :class="{ hidden: !streaming }">
            <video ref="video" autoplay playsinline muted class="webcam-feed"></video>
          </div>
          <div class="controls-bar">
            <button @click="toggleStream" :class="['btn', streaming ? 'btn-stop' : 'btn-start']">
              {{ streaming ? '⏹ Stop Stream' : '▶ Start Stream' }}
            </button>
          </div>
        </template>

        <!-- Upload mode -->
        <template v-else>
          <div v-if="!uploadedImage" class="placeholder">
            <div class="placeholder-content">
              <div class="placeholder-icon">🖼️</div>
              <label class="upload-btn">
                Choose an image
                <input type="file" accept="image/*" @change="handleUpload" hidden />
              </label>
            </div>
          </div>
          <div v-else class="upload-preview">
            <img :src="uploadedImage" class="uploaded-img" />
            <div class="controls-bar">
              <label class="btn btn-start" style="display:inline-block;cursor:pointer">
                🔄 Choose another
                <input type="file" accept="image/*" @change="handleUpload" hidden />
              </label>
            </div>
          </div>
        </template>

        <p v-if="error" class="error">{{ error }}</p>
      </div>

      <!-- Right: Classification Result -->
      <div class="monkey-half" :class="{ active: predictedClass }">
        <div v-if="predictedClass" class="monkey-content">
          <div class="class-badge" :class="'class-' + predictedClass">
            <span class="class-icon">{{ classIcon }}</span>
            <span class="class-name">{{ predictedClass }}</span>
          </div>
          <div class="confidence-bar">
            <div class="bar-track">
              <div class="bar-fill" :style="{ width: (confidence * 100) + '%' }"></div>
            </div>
            <div class="confidence-value">{{ (confidence * 100).toFixed(0) }}%</div>
          </div>
          <div class="monkey-frame">
            <img :src="referenceImage" class="monkey-img" />
          </div>
        </div>
        <div v-else class="monkey-idle">
          <div class="idle-icon">🐒</div>
          <p>{{ mode === 'webcam' ? 'Waiting for pose detection...' : 'Upload an image to classify' }}</p>
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
const mode = ref('webcam')
const streaming = ref(false)
const predictedClass = ref(null)
const confidence = ref(0)
const error = ref(null)
const uploadedImage = ref(null)
const uploading = ref(false)

let lastSent = 0
const SEND_INTERVAL = 150

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

/* ─── Webcam ─── */
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

/* ─── Upload ─── */
const handleUpload = async (e) => {
  const file = e.target.files[0]
  if (!file) return

  // Show preview
  const reader = new FileReader()
  reader.onload = (ev) => {
    uploadedImage.value = ev.target.result
  }
  reader.readAsDataURL(file)

  // Send to backend
  if (uploading.value) return
  uploading.value = true
  error.value = null

  try {
    const formData = new FormData()
    formData.append('file', file)
    const res = await fetch('/predict', { method: 'POST', body: formData })
    const data = await res.json()
    if (data.class) {
      predictedClass.value = data.class
      confidence.value = data.confidence
    } else {
      predictedClass.value = null
      error.value = data.error || 'No person detected in the image'
    }
  } catch (err) {
    error.value = 'Failed to classify image. Is the backend running?'
  } finally {
    uploading.value = false
  }
}

onBeforeUnmount(() => {
  if (socket) socket.close()
  const stream = video.value?.srcObject
  if (stream) stream.getTracks().forEach(track => track.stop())
})
</script>

<style scoped>
@import url('https://fonts.googleapis.com/css2?family=Comic+Neue:wght@400;600;700&display=swap');

.camera-predict {
  display: flex;
  flex-direction: column;
  height: 100vh;
  width: 100%;
  font-family: 'Comic Neue', 'Comic Sans MS', cursive, sans-serif;
  background: transparent;
}

/* ─── Mode tabs ─── */
.mode-tabs {
  display: flex;
  gap: 0;
  background: rgba(0,0,0,0.4);
  border-bottom: 1px solid rgba(255,255,255,0.1);
}

.tab {
  flex: 1;
  padding: 14px 20px;
  border: none;
  background: transparent;
  color: #6b7280;
  font-family: inherit;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
}

.tab.active {
  background: rgba(0,0,0,0.5);
  color: #fff;
  border-bottom: 3px solid #2563eb;
}

.tab:hover {
  color: #d1d5db;
}

/* ─── Split layout ─── */
.split {
  display: flex;
  flex: 1;
  overflow: hidden;
}

/* ─── Left half ─── */
.cam-half {
  flex: 0.45;
  display: flex;
  flex-direction: column;
  background: transparent;
  position: relative;
  overflow: hidden;
}

.placeholder {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #6b7280;
}

.placeholder-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 20px;
}

.placeholder-icon {
  font-size: 4rem;
  opacity: 0.4;
}

.upload-btn {
  padding: 14px 36px;
  background: #2563eb;
  color: white;
  border-radius: 10px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.2s;
}

.upload-btn:hover {
  background: #1d4ed8;
}

.video-wrapper {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
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

.upload-preview {
  flex: 1;
  display: flex;
  flex-direction: column;
}

.uploaded-img {
  flex: 1;
  width: 100%;
  object-fit: contain;
  display: block;
}

.controls-bar {
  padding: 16px 24px;
  text-align: center;
  background: rgba(0,0,0,0.5);
}

.btn {
  padding: 12px 40px;
  border: none;
  border-radius: 10px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
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
  color: #ef4444;
  margin: 8px 24px;
  text-align: center;
  font-size: 0.85rem;
}

/* ─── Right half ─── */
.monkey-half {
  flex: 1.55;
  display: flex;
  align-items: center;
  justify-content: center;
  background: transparent;
  transition: background 0.4s;
}

.monkey-half.active {
  background: transparent;
}

.monkey-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 24px;
  padding: 32px;
  width: 100%;
  max-width: 640px;
}

.class-badge {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 18px 32px;
  border-radius: 14px;
  font-size: 1.5rem;
  font-weight: 700;
  justify-content: center;
  width: 100%;
}

.class-icon {
  font-size: 2rem;
}

.class-name {
  text-transform: capitalize;
}

.class-heart-attack {
  background: #2a1212;
  color: #fca5a5;
  border: 2px solid #7f1d1d;
}

.class-idea {
  background: #2a2412;
  color: #fde68a;
  border: 2px solid #7f6d1d;
}

.class-stand {
  background: #122a1a;
  color: #86efac;
  border: 2px solid #1d7f3d;
}

.class-think {
  background: #22123a;
  color: #d8b4fe;
  border: 2px solid #5b1d7f;
}

.confidence-bar {
  display: flex;
  align-items: center;
  gap: 12px;
  width: 100%;
}

.bar-track {
  flex: 1;
  height: 10px;
  background: #1f2937;
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
  font-size: 1rem;
  font-weight: 700;
  color: #22c55e;
  min-width: 48px;
  text-align: right;
}

.monkey-frame {
  width: 100%;
  max-width: 600px;
  aspect-ratio: 1;
  border-radius: 20px;
  overflow: hidden;
  border: 4px solid rgba(255,255,255,0.15);
  box-shadow: 0 12px 48px rgba(0,0,0,0.6);
  background: #1a1a2e;
}

.monkey-img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.monkey-idle {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16px;
  color: #4b5563;
}

.idle-icon {
  font-size: 5rem;
}

.monkey-idle p {
  font-size: 1.1rem;
  margin: 0;
}
</style>
