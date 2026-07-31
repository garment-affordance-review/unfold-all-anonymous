const realInferenceData = {
  scene_1: [
    "case_001.mp4",
    "case_002.mp4",
    "case_005.mp4",
    "case_006.mp4",
    "case_007.mp4",
    "case_008.mp4",
    "case_009.mp4",
    "case_010.mp4",
    "case_011.mp4",
    "case_012.mp4",
    "case_013.mp4",
    "case_014.mp4",
    "case_015.mp4",
    "case_016.mp4",
    "case_017.mp4",
    "case_018.mp4",
    "case_019.mp4",
    "case_020.mp4",
    "case_021.mp4",
    "case_022.mp4",
    "case_023.mp4",
    "case_024.mp4",
    "case_025.mp4",
    "case_026.mp4"
  ],
  scene_2: [
    "case_001.mp4",
    "case_002.mp4",
    "case_003.mp4",
    "case_004.mp4",
    "case_005.mp4",
    "case_006.mp4",
    "case_007.mp4",
    "case_008.mp4",
    "case_009.mp4",
    "case_010.mp4",
    "case_011.mp4",
    "case_012.mp4",
    "case_013.mp4",
    "case_014.mp4",
    "case_015.mp4",
    "case_016.mp4"
  ],
  scene_3: [
    "case_001.mp4",
    "case_002.mp4",
    "case_003.mp4",
    "case_004.mp4",
    "case_005.mp4",
    "case_006.mp4",
    "case_007.mp4",
    "case_008.mp4",
    "case_009.mp4",
    "case_010.mp4",
    "case_011.mp4",
    "case_012.mp4",
    "case_013.mp4",
    "case_014.mp4",
    "case_015.mp4",
    "case_016.mp4",
    "case_017.mp4"
  ],
  scene_4: [
    "case_001.mp4",
    "case_002.mp4",
    "case_003.mp4",
    "case_004.mp4",
    "case_005.mp4",
    "case_006.mp4",
    "case_007.mp4",
    "case_008.mp4",
    "case_009.mp4",
    "case_010.mp4",
    "case_011.mp4",
    "case_012.mp4",
    "case_013.mp4"
  ]
};

document.addEventListener("DOMContentLoaded", () => {
  const video = document.getElementById("realInferenceVideo");
  const preview = document.getElementById("realInferencePreview");
  const instancePrev = document.getElementById("realInstancePrev");
  const instanceNext = document.getElementById("realInstanceNext");
  const caption = document.getElementById("realInferenceCaption");
  const stage = document.querySelector(".real-video-stage");
  const playButton = document.getElementById("realInferencePlay");
  const tabs = Array.from(document.querySelectorAll(".real-inference-controls .scene-tab"));

  if (!video || !preview || !instancePrev || !instanceNext || !caption || !stage || !playButton || tabs.length === 0) return;

  let currentScene = "scene_1";
  let pendingIndex = 0;
  let loadedScene = "";
  let loadedIndex = -1;
  const preloadedScenes = new Set();
  const frameCache = new Map();
  const frameVersion = "unified-grasp-v1";
  const videoVersion = "unified-grasp-v1";

  function sceneLabel(scene) {
    return scene.replace("scene_", "Scene ");
  }

  function framePath(scene, file) {
    return `static/images/real_inference_frames/${scene}/${file.replace(/\.mp4$/i, ".jpg")}?v=${frameVersion}`;
  }

  function frameKey(scene, index) {
    return `${scene}:${index}`;
  }

  function updateLabel(scene, index) {
    const files = realInferenceData[scene] || [];
    caption.textContent = `${sceneLabel(scene)} - Instance ${String(index + 1).padStart(2, "0")} / ${files.length}`;
    instancePrev.disabled = index === 0;
    instanceNext.disabled = index === files.length - 1;
  }

  function setPreview(scene, index) {
    const files = realInferenceData[scene] || [];
    const file = files[index] || files[0];
    if (!file) return;

    const cached = frameCache.get(frameKey(scene, index));
    preview.src = cached?.src || framePath(scene, file);
    stage.classList.add("is-previewing");
    stage.classList.remove("is-paused");
    updateLabel(scene, index);
  }

  function cacheFrame(scene, index) {
    const files = realInferenceData[scene] || [];
    const file = files[index];
    if (!file) return Promise.resolve();

    const key = frameKey(scene, index);
    const cached = frameCache.get(key);
    if (cached) return cached.ready;

    const image = new Image();
    image.decoding = "async";
    image.loading = "eager";
    image.src = framePath(scene, file);
    image.ready = (image.decode ? image.decode() : new Promise((resolve) => {
      image.onload = resolve;
      image.onerror = resolve;
    })).catch(() => {});
    frameCache.set(key, image);
    return image.ready;
  }

  function preloadSceneFrames(scene, priorityIndex = 0) {
    if (preloadedScenes.has(scene)) return;
    preloadedScenes.add(scene);
    const files = realInferenceData[scene] || [];
    cacheFrame(scene, priorityIndex);
    files.forEach((_, index) => {
      if (index !== priorityIndex) cacheFrame(scene, index);
    });
  }

  function warmAllScenes() {
    Object.keys(realInferenceData).forEach((scene) => preloadSceneFrames(scene));
  }

  function setVideo(scene, index) {
    const files = realInferenceData[scene] || [];
    const file = files[index] || files[0];
    if (!file) return;

    video.pause();
    video.src = `static/videos/real_inference/${scene}/${file}?v=${videoVersion}`;
    video.load();
    loadedScene = scene;
    loadedIndex = index;
    setPreview(scene, index);
  }

  function playSelectedVideo() {
    if (loadedScene !== currentScene || loadedIndex !== pendingIndex) {
      setVideo(currentScene, pendingIndex);
    }
    const playPromise = video.play();
    if (playPromise) {
      playPromise.catch(() => {
        stage.classList.add("is-previewing");
      });
    }
  }

  function resetInstance() {
    pendingIndex = 0;
  }

  function selectInstance(index, loadVideo = true) {
    const files = realInferenceData[currentScene] || [];
    pendingIndex = Math.max(0, Math.min(index, files.length - 1));
    video.pause();
    cacheFrame(currentScene, pendingIndex);
    setPreview(currentScene, pendingIndex);
    if (loadVideo) setVideo(currentScene, pendingIndex);
  }

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      currentScene = tab.dataset.scene || "scene_1";
      tabs.forEach((item) => item.classList.toggle("is-active", item === tab));
      resetInstance();
      preloadSceneFrames(currentScene, 0);
      setVideo(currentScene, 0);
    });
  });

  instancePrev.addEventListener("click", () => {
    selectInstance(pendingIndex - 1);
  });

  instanceNext.addEventListener("click", () => {
    selectInstance(pendingIndex + 1);
  });

  [instancePrev, instanceNext].forEach((button) => {
    button.addEventListener("pointerdown", (event) => {
      event.stopPropagation();
    });
  });

  playButton.addEventListener("click", playSelectedVideo);

  preview.addEventListener("click", playSelectedVideo);

  video.addEventListener("play", () => {
    stage.classList.remove("is-previewing");
    stage.classList.remove("is-paused");
  });

  video.addEventListener("pause", () => {
    if (!video.ended) stage.classList.add("is-paused");
  });

  video.addEventListener("ended", () => {
    stage.classList.add("is-previewing");
    stage.classList.remove("is-paused");
  });

  resetInstance();
  preloadSceneFrames(currentScene, 0);
  if ("requestIdleCallback" in window) {
    window.requestIdleCallback(warmAllScenes);
  } else {
    window.setTimeout(warmAllScenes, 600);
  }
  setVideo(currentScene, 0);
});
