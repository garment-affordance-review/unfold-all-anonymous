const artfData = {
  categories: [
    { id: "towels", label: "Towels" },
    { id: "tshirts", label: "T-shirts" },
    { id: "shorts", label: "Shorts" }
  ],
  scenes: [
    { id: "location_2", label: "Scene 1" },
    { id: "location_3", label: "Scene 2" },
    { id: "location_4", label: "Scene 3" },
    { id: "location_5", label: "Scene 4" },
    { id: "location_6", label: "Scene 5" },
    { id: "location_7", label: "Scene 6" },
    { id: "location_8", label: "Scene 7" },
    { id: "location_9", label: "Scene 8" }
  ],
  sampleCount: 8
};

document.addEventListener("DOMContentLoaded", () => {
  const oursImage = document.getElementById("artfOursPreview");
  const clothMateImage = document.getElementById("artfClothMatePreview");
  const sceneSlider = document.getElementById("artfSceneSlider");
  const sceneValue = document.getElementById("artfSceneValue");
  const samplePrev = document.getElementById("artfSamplePrev");
  const sampleNext = document.getElementById("artfSampleNext");
  const caption = document.getElementById("artfCaption");
  const tabs = Array.from(document.querySelectorAll(".artf-category-tab"));

  if (!oursImage || !clothMateImage || !sceneSlider || !sceneValue || !samplePrev || !sampleNext || !caption || tabs.length === 0) return;

  let currentCategory = "towels";
  let currentSceneIndex = 0;
  let currentSampleIndex = 0;
  const imageCache = new Map();
  const imageVersion = "artf-hires-v1";

  function sampleName(index) {
    return `sample_${String(index + 1).padStart(2, "0")}.jpg`;
  }

  function imagePath(method, category, sceneIndex, sampleIndex) {
    const scene = artfData.scenes[sceneIndex] || artfData.scenes[0];
    return `static/images/artf_visualization/${method}/${category}/${scene.id}/${sampleName(sampleIndex)}?v=${imageVersion}`;
  }

  function cacheKey(method, category, sceneIndex, sampleIndex) {
    return `${method}:${category}:${sceneIndex}:${sampleIndex}`;
  }

  function categoryLabel(category) {
    const item = artfData.categories.find((entry) => entry.id === category);
    return item ? item.label : category;
  }

  function updateLabels() {
    const scene = artfData.scenes[currentSceneIndex] || artfData.scenes[0];
    const sampleLabel = `${String(currentSampleIndex + 1).padStart(2, "0")} / ${String(artfData.sampleCount).padStart(2, "0")}`;
    sceneValue.textContent = `${scene.label} / ${artfData.scenes.length}`;
    caption.textContent = `${categoryLabel(currentCategory)} - ${scene.label} - Sample ${sampleLabel}`;
    samplePrev.disabled = currentSampleIndex === 0;
    sampleNext.disabled = currentSampleIndex === artfData.sampleCount - 1;
  }

  function cacheImage(method, category, sceneIndex, sampleIndex) {
    const key = cacheKey(method, category, sceneIndex, sampleIndex);
    const cached = imageCache.get(key);
    if (cached) return cached.ready;

    const preview = new Image();
    preview.decoding = "async";
    preview.loading = "eager";
    preview.src = imagePath(method, category, sceneIndex, sampleIndex);
    preview.ready = (preview.decode ? preview.decode() : new Promise((resolve) => {
      preview.onload = resolve;
      preview.onerror = resolve;
    })).catch(() => {});
    imageCache.set(key, preview);
    return preview.ready;
  }

  function setImage() {
    const oursCached = imageCache.get(cacheKey("ours", currentCategory, currentSceneIndex, currentSampleIndex));
    const clothMateCached = imageCache.get(cacheKey("clothmate", currentCategory, currentSceneIndex, currentSampleIndex));
    oursImage.src = oursCached?.src || imagePath("ours", currentCategory, currentSceneIndex, currentSampleIndex);
    clothMateImage.src = clothMateCached?.src || imagePath("clothmate", currentCategory, currentSceneIndex, currentSampleIndex);
    updateLabels();
  }

  function preloadCurrentScene() {
    for (let sampleIndex = 0; sampleIndex < artfData.sampleCount; sampleIndex += 1) {
      cacheImage("ours", currentCategory, currentSceneIndex, sampleIndex);
      cacheImage("clothmate", currentCategory, currentSceneIndex, sampleIndex);
    }
  }

  function warmNearbyScenes() {
    const sceneIndexes = [
      currentSceneIndex - 1,
      currentSceneIndex,
      currentSceneIndex + 1
    ].filter((index) => index >= 0 && index < artfData.scenes.length);

    sceneIndexes.forEach((sceneIndex) => {
      for (let sampleIndex = 0; sampleIndex < artfData.sampleCount; sampleIndex += 1) {
        cacheImage("ours", currentCategory, sceneIndex, sampleIndex);
        cacheImage("clothmate", currentCategory, sceneIndex, sampleIndex);
      }
    });
  }

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      currentCategory = tab.dataset.category || "towels";
      currentSceneIndex = 0;
      currentSampleIndex = 0;
      sceneSlider.value = "1";
      tabs.forEach((item) => item.classList.toggle("is-active", item === tab));
      preloadCurrentScene();
      setImage();
      window.setTimeout(warmNearbyScenes, 150);
    });
  });

  sceneSlider.addEventListener("input", () => {
    currentSceneIndex = Math.max(0, Number(sceneSlider.value || 1) - 1);
    currentSampleIndex = 0;
    preloadCurrentScene();
    setImage();
  });

  function setSampleIndex(index) {
    currentSampleIndex = Math.max(0, Math.min(index, artfData.sampleCount - 1));
    cacheImage("ours", currentCategory, currentSceneIndex, currentSampleIndex);
    cacheImage("clothmate", currentCategory, currentSceneIndex, currentSampleIndex);
    setImage();
  }

  samplePrev.addEventListener("click", () => {
    setSampleIndex(currentSampleIndex - 1);
  });

  sampleNext.addEventListener("click", () => {
    setSampleIndex(currentSampleIndex + 1);
  });

  [samplePrev, sampleNext].forEach((button) => {
    button.addEventListener("pointerdown", (event) => {
      event.stopPropagation();
    });
  });

  sceneSlider.max = String(artfData.scenes.length);
  preloadCurrentScene();
  if ("requestIdleCallback" in window) {
    window.requestIdleCallback(warmNearbyScenes);
  } else {
    window.setTimeout(warmNearbyScenes, 600);
  }
  setImage();
});
