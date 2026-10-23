const tabs = document.querySelectorAll('.tab');

const panels = {
  upload: document.getElementById('upload-panel'),
  camera: document.getElementById('camera-panel')
};

const imageInput = document.getElementById('image-input');
const browseBtn = document.getElementById('browse-btn');
const dropZone = document.getElementById('drop-zone');
const imagePreview = document.getElementById('image-preview');
const predictUploadBtn = document.getElementById('predict-upload-btn');
const uploadStatus = document.getElementById('upload-status');
const uploadResultSection = document.getElementById('upload-result-section');

let selectedFile = null;
let cameraStream = null;
let cameraTimer = null;
let predictingFrame = false;


// ======================================================
// LIVE PREDICTION STABILIZATION
// ======================================================

let predictionHistory = [];

const HISTORY_SIZE = 5;
const MIN_STABLE_COUNT = 2;
const MIN_CONFIDENCE = 0.15;


// ======================================================
// MODE SWITCHING
// ======================================================

function switchMode(mode) {

  tabs.forEach(tab => {
    tab.classList.toggle(
      'active',
      tab.dataset.mode === mode
    );
  });

  Object.entries(panels).forEach(([key, panel]) => {
    panel.classList.toggle(
      'active',
      key === mode
    );
  });

  if (mode !== 'camera') {
    stopCamera();
  }
}

tabs.forEach(tab => {
  tab.addEventListener('click', () => {
    switchMode(tab.dataset.mode);
  });
});


// ======================================================
// IMAGE UPLOAD
// ======================================================

browseBtn.addEventListener('click', () => {
  imageInput.click();
});

imageInput.addEventListener('change', () => {

  if (imageInput.files.length) {
    setImage(imageInput.files[0]);
  }

});


// ======================================================
// DRAG AND DROP
// ======================================================

['dragenter', 'dragover'].forEach(eventName => {

  dropZone.addEventListener(eventName, event => {

    event.preventDefault();
    dropZone.classList.add('drag');

  });

});


['dragleave', 'drop'].forEach(eventName => {

  dropZone.addEventListener(eventName, event => {

    event.preventDefault();
    dropZone.classList.remove('drag');

  });

});


dropZone.addEventListener('drop', event => {

  if (event.dataTransfer.files.length) {

    setImage(
      event.dataTransfer.files[0]
    );

  }

});


// ======================================================
// SET UP UPLOADED IMAGE
// ======================================================

function setImage(file) {

  if (!file.type.startsWith('image/')) {

    uploadStatus.textContent = 'Invalid file';
    return;

  }

  selectedFile = file;

  const reader = new FileReader();

  reader.onload = event => {

    imagePreview.innerHTML = `
      <img
        src="${event.target.result}"
        alt="Selected gesture"
      >
    `;

    uploadStatus.textContent =
      file.name.length > 22
        ? file.name.slice(0, 22) + '…'
        : file.name;

    predictUploadBtn.disabled = false;

    uploadResultSection.hidden = true;

  };

  reader.readAsDataURL(file);

}


// ======================================================
// UPLOAD IMAGE PREDICTION
// ======================================================

predictUploadBtn.addEventListener('click', async () => {

  if (!selectedFile) {
    return;
  }

  predictUploadBtn.disabled = true;

  predictUploadBtn.innerHTML =
    'Analyzing… <span>⌛</span>';

  uploadStatus.textContent =
    'Processing';

  const form = new FormData();

  form.append(
    'image',
    selectedFile
  );

  try {

    const response = await fetch(
      '/api/predict-upload',
      {
        method: 'POST',
        body: form
      }
    );

    const result =
      await response.json();

    if (!result.success) {

      throw new Error(
        result.message
      );

    }

    showUploadResult(result);

    uploadStatus.textContent =
      'Prediction complete';

  }

  catch (err) {

    uploadStatus.textContent =
      err.message ||
      'Prediction failed';

  }

  finally {

    predictUploadBtn.disabled = false;

    predictUploadBtn.innerHTML =
      'Analyze Gesture <span>→</span>';

  }

});


// ======================================================
// SHOW UPLOAD RESULT
// ======================================================

function showUploadResult(result) {

  uploadResultSection.hidden = false;

  document.getElementById(
    'upload-prediction'
  ).textContent =
    result.prediction;

  document.getElementById(
    'upload-result-title'
  ).textContent =
    `Gesture ${result.prediction}`;

  document.getElementById(
    'upload-confidence'
  ).textContent =
    result.confidence == null
      ? '—'
      : `${(result.confidence * 100).toFixed(1)}%`;

  uploadResultSection.scrollIntoView({
    behavior: 'smooth',
    block: 'nearest'
  });

}


// ======================================================
// CAMERA ELEMENTS
// ======================================================

const video =
  document.getElementById('video');

const canvas =
  document.getElementById('canvas');

const cameraStatus =
  document.getElementById('camera-status');

const cameraOverlay =
  document.getElementById('camera-overlay');

const cameraPrediction =
  document.getElementById('camera-prediction');

const cameraConfidence =
  document.getElementById('camera-confidence');

const cameraProgress =
  document.getElementById('camera-progress');

const cameraTopList =
  document.getElementById('camera-top-list');

const detectionStatus =
  document.getElementById('detection-status');

const startCameraBtn =
  document.getElementById('start-camera');

const stopCameraBtn =
  document.getElementById('stop-camera');


// ======================================================
// CAMERA BUTTONS
// ======================================================

startCameraBtn.addEventListener(
  'click',
  startCamera
);

stopCameraBtn.addEventListener(
  'click',
  stopCamera
);


// ======================================================
// START CAMERA
// ======================================================

async function startCamera() {

  try {

    cameraStream =
      await navigator.mediaDevices.getUserMedia({

        video: {

          width: {
            ideal: 960
          },

          height: {
            ideal: 540
          },

          facingMode: 'user'

        },

        audio: false

      });


    video.srcObject =
      cameraStream;

    await video.play();


    // Reset previous predictions
    predictionHistory = [];

    // Hide previous upload result
    uploadResultSection.hidden = true;


    cameraOverlay.style.display =
      'none';


    document
      .querySelector('.camera-frame')
      .classList.add('camera-running');


    cameraStatus.textContent =
      'Camera active';

    cameraStatus.style.color =
      '#4ee0a8';


    detectionStatus.textContent =
      '● SCANNING';

    detectionStatus.style.color =
      '#e0b85c';


    cameraPrediction.textContent =
      '—';

    cameraConfidence.textContent =
      '—';

    cameraProgress.style.width =
      '0%';


    cameraTopList.innerHTML =
      '';


    startCameraBtn.disabled =
      true;

    stopCameraBtn.disabled =
      false;


    // Predict approximately every 400 ms
    cameraTimer =
      setInterval(
        sendFrame,
        400
      );

  }

  catch (err) {

    console.error(
      'Camera error:',
      err
    );

    cameraStatus.textContent =
      'Camera permission denied';

    detectionStatus.textContent =
      '● CAMERA ERROR';

    detectionStatus.style.color =
      '#ff6b86';

  }

}


// ======================================================
// STOP CAMERA
// ======================================================

function stopCamera() {

  if (cameraTimer) {

    clearInterval(
      cameraTimer
    );

    cameraTimer = null;

  }


  if (cameraStream) {

    cameraStream
      .getTracks()
      .forEach(track => {
        track.stop();
      });

    cameraStream = null;

  }


  video.srcObject = null;


  // Clear prediction history
  predictionHistory = [];


  cameraOverlay.style.display =
    'grid';


  document
    .querySelector('.camera-frame')
    .classList.remove('camera-running');


  cameraStatus.textContent =
    'Camera stopped';

  cameraStatus.style.color =
    '';


  detectionStatus.textContent =
    '● READY';

  detectionStatus.style.color =
    '';


  cameraPrediction.textContent =
    '—';

  cameraConfidence.textContent =
    '—';

  cameraProgress.style.width =
    '0%';


  cameraTopList.innerHTML =
    '';


  startCameraBtn.disabled =
    false;

  stopCameraBtn.disabled =
    true;

}


// ======================================================
// STABILIZE PREDICTION
// ======================================================

function getStablePrediction(
  prediction,
  confidence
) {

  // Ignore low confidence predictions
  if (
    confidence < MIN_CONFIDENCE
  ) {

    return null;

  }


  // Add latest prediction
  predictionHistory.push(
    prediction
  );


  // Keep only recent predictions
  if (
    predictionHistory.length >
    HISTORY_SIZE
  ) {

    predictionHistory.shift();

  }


  // Count predictions
  const counts = {};


  predictionHistory.forEach(
    label => {

      counts[label] =
        (counts[label] || 0) + 1;

    }
  );


  let bestLabel = null;
  let bestCount = 0;


  // Find most frequent prediction
  for (
    const [label, count]
    of Object.entries(counts)
  ) {

    if (
      count > bestCount
    ) {

      bestLabel = label;
      bestCount = count;

    }

  }


  // Require repeated agreement
  if (
    bestCount >= MIN_STABLE_COUNT
  ) {

    return bestLabel;

  }


  return null;

}


// ======================================================
// SHOW TOP CAMERA PREDICTIONS
// ======================================================

function updateCameraTopPredictions(
  predictions
) {

  if (!cameraTopList) {
    return;
  }

  cameraTopList.innerHTML = '';

  predictions
    .slice(0, 3)
    .forEach((item, index) => {

      const label =
        item.prediction ||
        item.label ||
        '';

      const confidence =
        item.confidence == null
          ? 0
          : item.confidence;

      const percentage =
        Math.min(Math.max(confidence * 100, 0), 100).toFixed(1);

      const row =
        document.createElement('div');

      row.className =
        'prediction-row';

      row.innerHTML = `
        <strong>${index + 1}</strong>
        <strong>${label}</strong>
        <div class="prediction-bar"><i style="width: ${percentage}%"></i></div>
      `;

      cameraTopList.appendChild(row);

    });

}

// ======================================================
// SEND CAMERA FRAME
// ======================================================

async function sendFrame() {

  if (
    predictingFrame ||
    !cameraStream ||
    video.readyState < 2
  ) {

    return;

  }

  predictingFrame = true;


  try {

    // ==================================================
    // IMPORTANT:
    // Use the complete camera frame.
    // Frame is flipped horizontally in backend for prediction.
    // ==================================================

    canvas.width =
      video.videoWidth;

    canvas.height =
      video.videoHeight;


    const ctx =
      canvas.getContext('2d');


    ctx.drawImage(
      video,
      0,
      0,
      video.videoWidth,
      video.videoHeight
    );


    // Convert camera frame to JPEG
    const image =
      canvas.toDataURL(
        'image/jpeg',
        0.90
      );


    // Send frame to Flask
    const response =
      await fetch(
        '/api/predict-frame',
        {
          method: 'POST',

          headers: {
            'Content-Type':
              'application/json'
          },

          body: JSON.stringify({
            image: image
          })
        }
      );


    if (!response.ok) {

      throw new Error(
        `HTTP ${response.status}`
      );

    }


    const result =
      await response.json();


    // ==================================================
    // PREDICTION SUCCESS
    // ==================================================

    if (result.success) {

      const confidence =
        result.confidence || 0;


      // Stabilize prediction
      const stablePrediction =
        getStablePrediction(
          result.prediction,
          confidence
        );


      // Update main prediction
      if (stablePrediction) {

        cameraPrediction.textContent =
          stablePrediction;

        detectionStatus.textContent =
          '● STABLE HAND DETECTED';

        detectionStatus.style.color =
          '#4ee0a8';

      } else {

        cameraPrediction.textContent =
          result.prediction;

        detectionStatus.textContent =
          '● HAND DETECTED — HOLD';

        detectionStatus.style.color =
          '#e0b85c';

      }


      // Show confidence
      cameraConfidence.textContent =
        `${(confidence * 100).toFixed(1)}%`;


      // Confidence bar
      cameraProgress.style.width =
        `${Math.min(
          confidence * 100,
          100
        )}%`;


      // Show top predictions
      if (
        Array.isArray(
          result.top_predictions
        )
      ) {

        updateCameraTopPredictions(
          result.top_predictions
        );

      }


      cameraStatus.textContent =
        'Camera active';

      cameraStatus.style.color =
        '#4ee0a8';

    }


    // ==================================================
    // NO HAND DETECTED
    // ==================================================

    else {

      predictionHistory = [];

      cameraPrediction.textContent =
        '—';

      cameraConfidence.textContent =
        '—';

      cameraProgress.style.width =
        '0%';

      cameraTopList.innerHTML =
        '<div class="muted">Show your hand clearly to recognize.</div>';

      cameraStatus.textContent =
        'Camera active';

      cameraStatus.style.color =
        '#4ee0a8';

      detectionStatus.textContent =
        '● SCANNING';

      detectionStatus.style.color =
        '#e0b85c';

    }

  }


  catch (error) {

    console.error(
      'Camera prediction error:',
      error
    );

    cameraStatus.textContent =
      'CONNECTION ERROR';

  }


  finally {

    predictingFrame = false;

  }

}


// ======================================================
// CLEAN UP WHEN PAGE IS CLOSED
// ======================================================

window.addEventListener(
  'beforeunload',
  stopCamera
);