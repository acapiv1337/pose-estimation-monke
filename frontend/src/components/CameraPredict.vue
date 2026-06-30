<template>
  <div class="camera-predict">
    <div class="main-layout">
      <!-- Left: Webcam -->
      <div class="panel">
        <h2>Live Webcam</h2>
        <div class="video-container">
          <div v-if="!streaming" class="placeholder">
            <span>Press Start to begin</span>
          </div>
          <div v-if="streaming" class="video-wrapper">
            <video ref="video" autoplay playsinline muted class="webcam-feed"></video>
            <div class="class-overlay" v-if="predictedClass">
              <span class="overlay-badge" :class="'badge-' + predictedClass">
                {{ classIcon }} {{ predictedClass }}
              </span>
              <span class="overlay-conf">{{ (confidence * 100).toFixed(0) }}%</span>
            </div>
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
            <span class="conf-label">Confidence</span>
            <div class="bar-track">
              <div class="bar-fill" :style="{ width: (confidence * 100) + '%' }"></div>
            </div>
            <span class="conf-value">{{ (confidence * 100).toFixed(0) }}%</span>
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
let pc = null
let dataChannel = null
let localStream = null
let videoTrack = null
const streaming = ref(false)
const predictedClass = ref(null)
const confidence = ref(0)
const error = ref(null)

const ICE_SERVERS = []  // localhost only; add STUN/TURN for remote

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

function cleanup() {
  if (dataChannel) {
    dataChannel.close()
    dataChannel = null
  }
  if (pc) {
    pc.close()
    pc = null
  }
  if (localStream) {
    localStream.getTracks().forEach(t => t.stop())
    localStream = null
  }
  streaming.value = false
}

async function toggleStream() {
  if (streaming.value) {
    cleanup()
    return
  }

  try {
    // 1. Get webcam
    localStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, frameRate: 30 }
    })
    video.value.srcObject = localStream
    videoTrack = localStream.getVideoTracks()[0]

    // 2. Create peer connection
    pc = new RTCPeerConnection({ iceServers: ICE_SERVERS })
    pc.addTrack(videoTrack, localStream)

    // 3. Handle incoming DataChannel from server
    pc.ondatachannel = (event) => {
      dataChannel = event.channel
      dataChannel.onmessage = (e) => {
        const data = JSON.parse(e.data)
        predictedClass.value = data.class
        confidence.value = data.confidence
      }
    }

    // 4. Handle ICE connection state
    pc.oniceconnectionstatechange = () => {
      if (pc.iceConnectionState === 'disconnected' ||
          pc.iceConnectionState === 'failed') {
        error.value = 'WebRTC connection lost'
        cleanup()
      }
    }

    // 5. Create SDP offer and send to server
    const offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    const resp = await fetch('/offer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        sdp: pc.localDescription.sdp,
        type: pc.localDescription.type,
      }),
    })

    if (!resp.ok) {
      throw new Error(`Server returned ${resp.status}`)
    }

    const answer = await resp.json()
    await pc.setRemoteDescription(
      new RTCSessionDescription({ sdp: answer.sdp, type: answer.type })
    )

    streaming.value = true
    error.value = null
    predictedClass.value = null
  } catch (err) {
    cleanup()
    error.value = err.message || 'Failed to connect'
  }
}

onBeforeUnmount(() => {
  cleanup()
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

.placeholder {
  color: #6b7280;
  font-size: 1rem;
}

.video-wrapper {
  position: relative;
  width: 100%;
  height: 100%;
}

.webcam-feed {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
}

.class-overlay {
  position: absolute;
  bottom: 16px;
  left: 16px;
  display: flex;
  gap: 8px;
  align-items: center;
}

.overlay-badge {
  padding: 6px 16px;
  border-radius: 20px;
  font-size: 0.95rem;
  font-weight: 700;
  text-transform: capitalize;
}

.overlay-conf {
  padding: 6px 12px;
  border-radius: 20px;
  background: rgba(0,0,0,0.65);
  color: white;
  font-size: 0.9rem;
  font-weight: 600;
}

.badge-heart-attack {
  background: rgba(239, 68, 68, 0.85);
  color: white;
}

.badge-idea {
  background: rgba(234, 179, 8, 0.85);
  color: #1a1a1a;
}

.badge-stand {
  background: rgba(34, 197, 94, 0.85);
  color: white;
}

.badge-think {
  background: rgba(168, 85, 247, 0.85);
  color: white;
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

/* Right panel */
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

.class-icon { font-size: 2.2rem; }
.class-name { text-transform: capitalize; }

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

.conf-label {
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

.conf-value {
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
