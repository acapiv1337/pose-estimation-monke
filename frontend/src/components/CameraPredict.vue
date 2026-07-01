<template>
  <div class="camera-predict">
    <!-- ═══ LOGIN OVERLAY ═══ -->
    <div v-if="!loggedIn" class="login-overlay">
      <div class="login-modal">
        <div class="login-icon">🐵</div>
        <h2 class="login-title">Monkey Pose Detection</h2>
        <p class="login-subtitle">Enter password to access the app</p>
        <div class="login-input-wrap">
          <input v-model="passwordInput" type="password" class="login-input" placeholder="••••••••" @keyup.enter="checkLogin" autofocus />
          <button class="login-btn" @click="checkLogin">Enter →</button>
        </div>
        <p v-if="loginError" class="login-error-msg">Incorrect password. Try again.</p>
      </div>
    </div>
    <template v-if="loggedIn">
      <!-- ═══ HEADER ═══ -->
    <header class="header">
      <h1 class="header-title">MONKEY POSE DETECTION</h1>
      <div class="header-actions">
        <!-- Info button -->
        <div class="info-wrap">
          <button class="icon-btn info-btn" @mouseenter="showInfo = true" @mouseleave="showInfo = false"
            @focus="showInfo = true" @blur="showInfo = false">
            ⓘ
          </button>
          <Transition name="pop">
            <div v-if="showInfo" class="info-popover">
              <div class="popover-arrow"></div>
              <h3>How it works</h3>
              <p><strong>YOLOv8s-pose</strong> extracts 11 upper-body keypoints from each frame, then an <strong>XGBoost
                  classifier</strong> predicts one of 4 gestures:</p>
              <div class="pose-grid">
                <div class="pose-card" v-for="p in poseList" :key="p.class">
                  <img :src="p.image" :alt="p.class" class="pose-thumb" />
                  <span class="pose-label">{{ p.emoji }} {{ p.label }}</span>
                </div>
              </div>
              <p class="popover-hint">Try to mimic one of these poses 🙈</p>
              <p class="popover-current" v-if="predictedClass">
                Currently detecting: <strong>{{ predictedClass }}</strong>
              </p>
            </div>
          </Transition>
        </div>
        <!-- Upload button -->
        <div class="upload-wrap">
          <button class="icon-btn upload-btn-header" @click="$refs.fileInput.click()">
            📁
          </button>
          <span class="upload-tooltip">Upload an image</span>
          <input ref="fileInput" type="file" accept="image/*" @change="handleUpload" hidden />
        </div>
      </div>
    </header>

    <!-- ═══ MAIN CONTENT ═══ -->
    <div class="main-area">
      <!-- LEFT: Webcam -->
      <div class="panel panel-left">
        <div class="panel-header-row">
          <span class="panel-label">WEBCAM</span>
          <button v-if="!streaming" class="start-btn" @click="toggleStream">▶ Start Webcam</button>
          <button v-else class="stop-btn" @click="toggleStream">■ Stop</button>
        </div>
        <div class="panel-body">
          <!-- Placeholder -->
          <div v-if="!streaming && !uploadedImage" class="placeholder-content">
            <div class="placeholder-icon">🐵</div>
            <p>Press <strong>Start Webcam</strong> to begin</p>
          </div>
          <!-- Video feed -->
          <div v-show="streaming" class="video-wrapper">
            <video ref="video" autoplay playsinline muted class="webcam-feed"></video>
          </div>
          <!-- Uploaded image preview -->
          <div v-if="uploadedImage && !streaming" class="upload-preview">
            <img :src="uploadedImage" class="uploaded-img" />
          </div>
        </div>
      </div>

      <!-- RIGHT: Monkey Display -->
      <div class="panel panel-right">
        <div class="panel-header-row">
          <span class="panel-label">MONKEY DISPLAY</span>
        </div>
        <div class="panel-body monkey-body">
          <div v-if="predictedClass" class="monkey-content">
            <div class="monkey-frame">
              <img :src="referenceImage" :alt="predictedClass" class="monkey-img" />
            </div>
          </div>
          <div v-else class="monkey-idle">
            <div class="idle-avatar">🐒</div>
            <p class="idle-text">Waiting for pose...</p>
          </div>
        </div>
      </div>
    </div>

    <p v-if="error" class="error">{{ error }}</p>

    <!-- ═══ FOOTER ═══ -->
    <footer class="footer">
      <div class="label-bar">
        <span class="label-value" v-if="predictedClass" :class="'label-' + predictedClass">
          {{ predictedClass.replace('-', ' ') }}
        </span>
        <span class="label-value label-none" v-else>—</span>
      </div>
      <div class="confident-box">
        <span class="confident-pct" v-if="predictedClass">{{ (confidence * 100).toFixed(0) }}%</span>
        <span class="confident-pct pct-none" v-else>—</span>
        <span class="confident-caption">PERCENT<br />CONFIDENT</span>
      </div>
    </footer>
    </template>
  </div>
</template>

<script setup>
import { ref, computed, onBeforeUnmount, nextTick } from 'vue'

const video = ref(null)
const fileInput = ref(null)
const canvas = document.createElement('canvas')
let socket = null

const streaming = ref(false)
const predictedClass = ref(null)
const confidence = ref(0)
const error = ref(null)
const uploadedImage = ref(null)
const uploading = ref(false)
const showInfo = ref(false)

/* ─── Login state ─── */
const loggedIn = ref(false)
const passwordInput = ref('')
const loginError = ref(false)

const checkLogin = () => {
  if (passwordInput.value === 'test123') {
    loggedIn.value = true
    loginError.value = false
  } else {
    loginError.value = true
    passwordInput.value = ''
    nextTick(() => {
      const input = document.querySelector('.login-input')
      if (input) input.focus()
    })
  }
}

let lastSent = 0
const SEND_INTERVAL = 150

const classEmoji = {
  'heart-attack': '💔',
  'idea': '💡',
  'stand': '🧍',
  'think': '🤔'
}

const referenceImage = computed(() => {
  if (!predictedClass.value) return ''
  return `/static/poses/${predictedClass.value}.jpeg`
})

const classes = ['heart-attack', 'idea', 'stand', 'think']
const poseList = classes.map(c => ({
  class: c,
  label: c.replace('-', ' '),
  emoji: classEmoji[c],
  image: `/static/poses/${c}.jpeg`
}))

/* ─── Webcam ─── */
const toggleStream = async () => {
  if (!streaming.value) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true })
      video.value.srcObject = stream
      video.value.play()

      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      socket = new WebSocket(`${wsProtocol}//${window.location.host}/ws/pose`)

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
    if (socket) {
      socket.close()
    }
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

<style>
/* ─── Global reset for full-page layout ─── */
.camera-predict {
  display: flex;
  flex-direction: column;
  height: 100vh;
  width: 100%;
  font-family: 'Comic Neue', 'Comic Sans MS', cursive, sans-serif;
  color: #e0e0e0;
  background: transparent;
}

/* ═══ HEADER ═══ */
.header {
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  padding: 16px 24px;
  border-bottom: 2px solid rgba(255, 255, 255, 0.15);
  background: rgba(0, 0, 0, 0.55);
  backdrop-filter: blur(6px);
  flex-shrink: 0;
  overflow: visible;
  z-index: 20;
}

.header-title {
  margin: 0;
  font-size: 1.9rem;
  font-weight: 800;
  letter-spacing: 2px;
  color: #fff;
  text-shadow: 0 2px 8px rgba(0, 0, 0, 0.6);
}

.header-actions {
  position: absolute;
  right: 20px;
  top: 50%;
  transform: translateY(-50%);
  display: flex;
  gap: 8px;
  overflow: visible;
  align-items: flex-start;
}

.icon-btn {
  border: 2px solid rgba(255, 255, 255, 0.25);
  border-radius: 8px;
  background: rgba(0, 0, 0, 0.45);
  color: #fff;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
  line-height: 1;
}

.icon-btn:hover {
  background: rgba(255, 255, 255, 0.15);
  border-color: rgba(255, 255, 255, 0.5);
}

/* Header icon buttons — same size square */
.info-btn,
.upload-btn-header {
  width: 40px;
  height: 40px;
  font-size: 1.2rem;
  align-self: flex-start;
}

/* ─── Info popover ─── */
.info-wrap {
  position: relative;
}

.info-popover {
  position: absolute;
  top: calc(100% + 10px);
  right: 0;
  width: 320px;
  background: #1e1e2e;
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 12px;
  padding: 20px 24px;
  z-index: 100;
  box-shadow: 0 12px 40px rgba(0, 0, 0, 0.7);
  font-size: 0.95rem;
  line-height: 1.6;
}

.popover-arrow {
  position: absolute;
  top: -8px;
  right: 14px;
  width: 14px;
  height: 14px;
  background: #1e1e2e;
  border-left: 1px solid rgba(255, 255, 255, 0.2);
  border-top: 1px solid rgba(255, 255, 255, 0.2);
  transform: rotate(45deg);
}

.info-popover h3 {
  margin: 0 0 8px;
  font-size: 1rem;
  color: #fbbf24;
}

.info-popover ul {
  margin: 8px 0;
  padding-left: 20px;
}

.info-popover li {
  margin-bottom: 2px;
}

.popover-current {
  margin: 8px 0 0;
  padding-top: 8px;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
  color: #86efac;
}

/* ─── Pose preview grid in info popover ─── */
.pose-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin: 10px 0 4px;
}

.pose-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 8px;
  padding: 6px;
  transition: background 0.2s;
}

.pose-card:hover {
  background: rgba(255, 255, 255, 0.1);
}

.pose-thumb {
  width: 100%;
  aspect-ratio: 1 / 1;
  object-fit: cover;
  border-radius: 4px;
  display: block;
}

.pose-label {
  font-size: 0.75rem;
  font-weight: 600;
  color: #cbd5e1;
  text-transform: capitalize;
  white-space: nowrap;
}

.popover-hint {
  margin: 6px 0 0;
  font-size: 0.85rem;
  color: #fbbf24;
  text-align: center;
}

/* Popover transition */
.pop-enter-active,
.pop-leave-active {
  transition: opacity 0.2s ease, transform 0.2s ease;
}
.pop-enter-from,
.pop-leave-to {
  opacity: 0;
  transform: translateY(-6px);
}

/* ─── Upload tooltip ─── */
.upload-wrap {
  position: relative;
}

.upload-tooltip {
  position: absolute;
  top: calc(100% + 8px);
  right: 0;
  white-space: nowrap;
  padding: 6px 14px;
  background: #1e1e2e;
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 8px;
  font-size: 0.8rem;
  color: #cbd5e1;
  pointer-events: none;
  opacity: 0;
  transition: opacity 0.2s ease;
  z-index: 100;
}

.upload-wrap:hover .upload-tooltip {
  opacity: 1;
}

/* ═══ MAIN AREA ═══ */
.main-area {
  flex: 1;
  display: flex;
  gap: 20px;
  padding: 20px;
  min-height: 0;
  align-items: center;
}

.panel {
  flex: 1;
  display: flex;
  flex-direction: column;
  border: 2px solid rgba(255, 255, 255, 0.15);
  border-radius: 14px;
  overflow: hidden;
  background: rgba(0, 0, 0, 0.35);
  backdrop-filter: blur(4px);
  min-width: 0;
  aspect-ratio: 16 / 9.2;
}

.panel-header-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 16px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  background: rgba(0, 0, 0, 0.3);
  flex-shrink: 0;
  min-height: 44px;
}

.panel-label {
  font-size: 1rem;
  font-weight: 700;
  letter-spacing: 1.5px;
  color: rgba(255, 255, 255, 0.5);
}

.panel-body {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  position: relative;
}

/* ─── Start / Stop buttons ─── */
.start-btn {
  padding: 6px 18px;
  border: none;
  border-radius: 8px;
  background: #2563eb;
  color: #fff;
  font-family: inherit;
  font-size: 0.85rem;
  font-weight: 700;
  cursor: pointer;
  transition: background 0.2s;
}

.start-btn:hover {
  background: #1d4ed8;
}

.stop-btn {
  padding: 6px 18px;
  border: none;
  border-radius: 8px;
  background: #dc2626;
  color: #fff;
  font-family: inherit;
  font-size: 0.85rem;
  font-weight: 700;
  cursor: pointer;
  transition: background 0.2s;
}

.stop-btn:hover {
  background: #b91c1c;
}

/* ─── Placeholder ─── */
.placeholder-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  color: rgba(255, 255, 255, 0.4);
}

.placeholder-icon {
  font-size: 4rem;
}

.placeholder-content p {
  margin: 0;
  font-size: 0.95rem;
}

/* ─── Video ─── */
.video-wrapper {
  width: 100%;
  height: 100%;
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

/* ─── Upload preview ─── */
.upload-preview {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}

.uploaded-img {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
}

/* ═══ RIGHT PANEL — Monkey Display ═══ */
.monkey-body {
  padding: 0;
}

.monkey-content {
  width: 100%;
  height: 100%;
}

.monkey-frame {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}

.monkey-img {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
}

.monkey-pose-name {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 18px;
  border-radius: 20px;
  font-size: 0.9rem;
  font-weight: 700;
  text-transform: capitalize;
  flex-shrink: 0;
}

.pose-emoji {
  font-size: 1.2rem;
}

.pose-heart-attack {
  background: rgba(127, 29, 29, 0.4);
  color: #fca5a5;
  border: 1px solid #7f1d1d;
}

.pose-idea {
  background: rgba(127, 109, 29, 0.4);
  color: #fde68a;
  border: 1px solid #7f6d1d;
}

.pose-stand {
  background: rgba(29, 127, 61, 0.4);
  color: #86efac;
  border: 1px solid #1d7f3d;
}

.pose-think {
  background: rgba(91, 29, 127, 0.4);
  color: #d8b4fe;
  border: 1px solid #5b1d7f;
}

/* ─── Monkey idle ─── */
.monkey-idle {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
}

.idle-avatar {
  font-size: 5rem;
  filter: grayscale(0.5);
  opacity: 0.6;
}

.idle-text {
  margin: 0;
  font-size: 0.95rem;
  color: rgba(255, 255, 255, 0.35);
  font-weight: 600;
}

/* ═══ ERROR MESSAGE ═══ */
.error {
  color: #ef4444;
  margin: 0 20px 8px;
  text-align: center;
  font-size: 0.85rem;
  flex-shrink: 0;
}

/* ═══ FOOTER ═══ */
.footer {
  display: flex;
  gap: 12px;
  padding: 14px 20px;
  border-top: 8px solid rgba(255, 255, 255, 0.2);
  background: rgba(0, 0, 0, 0.55);
  backdrop-filter: blur(6px);
  flex-shrink: 0;
  align-items: flex-start;
}

.label-bar {
  flex: 1;
  border: 2px solid rgba(255, 255, 255, 0.15);
  border-radius: 10px;
  padding: 8px 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(0, 0, 0, 0.3);
  height: 56px;
  box-sizing: border-box;
  position: relative;
}

.label-value {
  font-size: 1.8rem;
  font-weight: 800;
  text-transform: capitalize;
  transition: color 0.3s;
}

.label-none {
  color: rgba(255, 255, 255, 0.2);
}

.label-heart-attack {
  color: #fca5a5;
}

.label-idea {
  color: #fde68a;
}

.label-stand {
  color: #86efac;
}

.label-think {
  color: #d8b4fe;
}

.confident-box {
  width: 100px;
  min-width: 100px;
  height: 72px;
  border: 2px solid rgba(255, 255, 255, 0.15);
  border-radius: 10px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 4px 8px;
  gap: 1px;
  background: rgba(0, 0, 0, 0.3);
  box-sizing: border-box;
  align-self: flex-start;
}

.confident-pct {
  font-size: 1.5rem;
  font-weight: 800;
  color: #22c55e;
  line-height: 1.2;
}

.pct-none {
  color: rgba(255, 255, 255, 0.2);
}

.confident-caption {
  font-size: 0.55rem;
  font-weight: 700;
  letter-spacing: 0.5px;
  color: rgba(255, 255, 255, 0.35);
  text-align: center;
  line-height: 1.3;
}

/* ═══ LOGIN OVERLAY ═══ */
.login-overlay {
  position: fixed;
  inset: 0;
  z-index: 9999;
  background: rgba(0, 0, 0, 0.75);
  backdrop-filter: blur(6px);
  display: flex;
  align-items: center;
  justify-content: center;
}

.login-modal {
  background: #1e1e2e;
  border: 1px solid rgba(255, 255, 255, 0.15);
  border-radius: 20px;
  padding: 40px 44px;
  width: 360px;
  max-width: 90vw;
  text-align: center;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.6);
  animation: login-enter 0.3s ease;
}

@keyframes login-enter {
  from {
    opacity: 0;
    transform: scale(0.95) translateY(10px);
  }
  to {
    opacity: 1;
    transform: scale(1) translateY(0);
  }
}

.login-icon {
  font-size: 4rem;
  margin-bottom: 8px;
}

.login-title {
  margin: 0 0 4px;
  font-size: 1.6rem;
  font-weight: 800;
  color: #fff;
}

.login-subtitle {
  margin: 0 0 24px;
  font-size: 0.9rem;
  color: rgba(255, 255, 255, 0.5);
}

.login-input-wrap {
  display: flex;
  gap: 8px;
  align-items: stretch;
}

.login-input {
  flex: 1;
  padding: 12px 16px;
  border: 2px solid rgba(255, 255, 255, 0.15);
  border-radius: 10px;
  background: rgba(0, 0, 0, 0.4);
  color: #fff;
  font-family: inherit;
  font-size: 1rem;
  outline: none;
  transition: border-color 0.2s;
}

.login-input:focus {
  border-color: #2563eb;
}

.login-input::placeholder {
  color: rgba(255, 255, 255, 0.25);
}

.login-btn {
  padding: 12px 20px;
  border: none;
  border-radius: 10px;
  background: #2563eb;
  color: #fff;
  font-family: inherit;
  font-size: 0.95rem;
  font-weight: 700;
  cursor: pointer;
  white-space: nowrap;
  transition: background 0.2s;
}

.login-btn:hover {
  background: #1d4ed8;
}

.login-error-msg {
  margin: 14px 0 0;
  font-size: 0.85rem;
  color: #ef4444;
  font-weight: 600;
}
</style>
