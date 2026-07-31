(function () {
  const DATA_URL = 'static/data/pair_value_teacher_demo.json?v=method-assets-8-v1';
  const canvas = document.getElementById('pairTeacherCanvas');
  if (!canvas) return;

  const stage = canvas.closest('.pair-teacher-stage');
  const tabs = Array.from(document.querySelectorAll('.pair-asset-tab'));
  const prevButton = document.getElementById('pairAnchorPrev');
  const nextButton = document.getElementById('pairAnchorNext');
  const assetName = document.getElementById('pairTeacherAssetName');
  const anchorName = document.getElementById('pairTeacherAnchorName');
  const rewardRange = document.getElementById('pairTeacherRewardRange');
  const caption = document.getElementById('pairTeacherCaption');

  const state = {
    data: null,
    assetIndex: 0,
    anchorIndex: 0,
    cameraZ: 2.8,
    dragging: false,
    moved: false,
    lastX: 0,
    lastY: 0,
    scene: null,
    camera: null,
    renderer: null,
    group: null,
    pointsObject: null,
    raycaster: null,
    pointer: null,
    pendingAssetIndex: null,
    markerMeshes: [],
    lineObject: null
  };

  function currentAsset() {
    return state.data.assets[state.assetIndex];
  }

  function currentAnchor() {
    return currentAsset().anchors[state.anchorIndex];
  }

  function setStageMessage(message) {
    if (!stage) return;
    stage.dataset.message = message;
    stage.classList.toggle('is-loading', Boolean(message));
  }

  function colorFor(t) {
    const clamped = Math.max(0, Math.min(1, t));
    const stops = [
      [37, 99, 235],
      [20, 184, 166],
      [250, 204, 21],
      [239, 68, 68]
    ];
    const scaled = clamped * (stops.length - 1);
    const i = Math.min(stops.length - 2, Math.floor(scaled));
    const f = scaled - i;
    const a = stops[i];
    const b = stops[i + 1];
    return [
      (a[0] + (b[0] - a[0]) * f) / 255,
      (a[1] + (b[1] - a[1]) * f) / 255,
      (a[2] + (b[2] - a[2]) * f) / 255
    ];
  }

  function normalizeScores(scores) {
    const min = Math.min(...scores);
    const max = Math.max(...scores);
    return scores.map((v) => (v - min) / (max - min + 1e-6));
  }

  function initThree() {
    if (!window.THREE) {
      setStageMessage('Three.js failed to load. Refresh the page or check network access.');
      return false;
    }
    state.scene = new THREE.Scene();
    state.scene.background = new THREE.Color(0xffffff);
    state.camera = new THREE.PerspectiveCamera(30, 1, 0.01, 20);
    state.camera.position.set(0, 0, state.cameraZ);
    state.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    state.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    state.group = new THREE.Group();
    state.group.rotation.x = 0;
    state.group.rotation.z = 0;
    state.scene.add(state.group);
    state.raycaster = new THREE.Raycaster();
    state.raycaster.params.Points.threshold = 0.035;
    state.pointer = new THREE.Vector2();

    const grid = new THREE.GridHelper(1.35, 14, 0xd1d5db, 0xe5e7eb);
    grid.rotation.x = Math.PI / 2;
    grid.position.z = -0.34;
    state.scene.add(grid);

    const axes = new THREE.AxesHelper(0.42);
    axes.position.set(-0.62, -0.46, -0.28);
    state.scene.add(axes);
    resize();
    return true;
  }

  function resize() {
    if (!state.renderer || !state.camera) return;
    const rect = stage.getBoundingClientRect();
    if (rect.width < 2 || rect.height < 2) return;
    state.camera.aspect = rect.width / rect.height;
    state.camera.updateProjectionMatrix();
    state.renderer.setSize(rect.width, rect.height, false);
    render();
  }

  function clearGroup() {
    if (!state.group) return;
    while (state.group.children.length > 0) {
      const child = state.group.children.pop();
      if (child.geometry) child.geometry.dispose();
      if (child.material) child.material.dispose();
    }
    state.markerMeshes = [];
    state.pointsObject = null;
    state.lineObject = null;
  }

  function updateScene() {
    if (!state.data || !state.group) return;
    const asset = currentAsset();
    const anchor = currentAnchor();
    const norm = normalizeScores(anchor.scores);

    clearGroup();
    const positions = new Float32Array(asset.points.length * 3);
    const colors = new Float32Array(asset.points.length * 3);
    asset.points.forEach((point, i) => {
      positions.set(point, i * 3);
      colors.set(colorFor(norm[i]), i * 3);
    });
    const pointGeometry = new THREE.BufferGeometry();
    pointGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    pointGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    const pointMaterial = new THREE.PointsMaterial({
      size: 0.064,
      vertexColors: true,
      sizeAttenuation: true
    });
    state.pointsObject = new THREE.Points(pointGeometry, pointMaterial);
    state.group.add(state.pointsObject);

    const anchorPoint = asset.points[anchor.index];
    const linePositions = [];
    anchor.topPairs.forEach((pair) => {
      const target = asset.points[pair.j];
      linePositions.push(anchorPoint[0], anchorPoint[1], anchorPoint[2], target[0], target[1], target[2]);
    });
    const lineGeometry = new THREE.BufferGeometry();
    lineGeometry.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));
    state.lineObject = new THREE.LineSegments(
      lineGeometry,
      new THREE.LineBasicMaterial({ color: 0x2563eb, transparent: true, opacity: 0.72 })
    );
    state.group.add(state.lineObject);

    addMarker(anchorPoint, 0xdc2626, 0.032);
    anchor.topPairs.slice(0, 6).forEach((pair, rank) => {
      addMarker(asset.points[pair.j], rank === 0 ? 0xf97316 : 0x2563eb, rank === 0 ? 0.027 : 0.022);
    });
    render();
  }

  function addMarker(point, color, radius) {
    const geometry = new THREE.SphereGeometry(radius, 16, 12);
    const material = new THREE.MeshBasicMaterial({ color });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(point[0], point[1], point[2]);
    state.group.add(mesh);
    state.markerMeshes.push(mesh);
  }

  function render() {
    if (state.renderer && state.scene && state.camera) {
      state.camera.position.z = state.cameraZ;
      state.renderer.render(state.scene, state.camera);
    }
  }

  function updateLabels() {
    const asset = currentAsset();
    const anchor = currentAnchor();
    const best = anchor.topPairs[0];
    const range = asset.rewardRange || [-0.42, -0.06];
    if (assetName) assetName.textContent = `${asset.name} · ${asset.category}`;
    if (anchorName) anchorName.textContent = `${anchor.label} (${state.anchorIndex + 1} / ${asset.anchors.length})`;
    if (rewardRange) rewardRange.textContent = `Reward ${range[0].toFixed(2)} to ${range[1].toFixed(2)}`;
    if (caption && best) {
      caption.textContent = `Current best predicted pair: x1 point ${anchor.index}, x2 point ${best.j}, reward ${best.score.toFixed(3)}.`;
    }
  }

  function setAsset(index, options) {
    const count = state.data.assets.length;
    const nextIndex = (index + count) % count;
    state.assetIndex = nextIndex;
    state.anchorIndex = 0;
    tabs.forEach((tab, i) => tab.classList.toggle('is-active', i === nextIndex));
    updateLabels();
    updateScene();
    if (!options || !options.silent) {
      window.dispatchEvent(new CustomEvent('methodAssetChange', {
        detail: { index: nextIndex, source: 'teacher' }
      }));
    }
  }

  function setAnchor(index) {
    const count = currentAsset().anchors.length;
    state.anchorIndex = (index + count) % count;
    updateLabels();
    updateScene();
  }

  function pickPoint(event) {
    if (!state.pointsObject || !state.raycaster || !state.pointer || !state.camera) return -1;
    const rect = canvas.getBoundingClientRect();
    state.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    state.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    state.raycaster.setFromCamera(state.pointer, state.camera);
    const hits = state.raycaster.intersectObject(state.pointsObject, false);
    return hits.length ? hits[0].index : -1;
  }

  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => setAsset(index));
  });
  prevButton.addEventListener('click', () => {
    if (!state.data) return;
    const count = state.data.assets.length;
    setAsset((state.assetIndex - 1 + count) % count);
  });
  nextButton.addEventListener('click', () => {
    if (!state.data) return;
    const count = state.data.assets.length;
    setAsset((state.assetIndex + 1) % count);
  });
  window.addEventListener('methodAssetChange', (event) => {
    if (!event.detail || event.detail.source === 'teacher') return;
    if (!state.data) {
      state.pendingAssetIndex = Number(event.detail.index) || 0;
      return;
    }
    setAsset(Number(event.detail.index) || 0, { silent: true });
  });

  canvas.addEventListener('pointerdown', (event) => {
    state.dragging = true;
    state.moved = false;
    state.lastX = event.clientX;
    state.lastY = event.clientY;
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener('pointermove', (event) => {
    if (state.dragging) {
      const dx = event.clientX - state.lastX;
      const dy = event.clientY - state.lastY;
      if (Math.abs(dx) + Math.abs(dy) > 2) state.moved = true;
      state.group.rotation.z += dx * 0.008;
      state.group.rotation.x += dy * 0.008;
      state.group.rotation.x = Math.max(-1.45, Math.min(1.1, state.group.rotation.x));
      state.lastX = event.clientX;
      state.lastY = event.clientY;
      render();
    }
  });
  canvas.addEventListener('pointerup', (event) => {
    state.dragging = false;
    if (state.moved) return;
    const point = pickPoint(event);
    if (point >= 0) {
      let anchorIndex = currentAsset().anchors.findIndex((a) => a.index === point);
      if (anchorIndex < 0) {
        const clicked = currentAsset().points[point];
        let bestDist = Infinity;
        currentAsset().anchors.forEach((anchor, index) => {
          const p = currentAsset().points[anchor.index];
          const d = (p[0] - clicked[0]) ** 2 + (p[1] - clicked[1]) ** 2 + (p[2] - clicked[2]) ** 2;
          if (d < bestDist) {
            bestDist = d;
            anchorIndex = index;
          }
        });
      }
      setAnchor(anchorIndex);
    }
  });
  canvas.addEventListener('pointerleave', () => {
    state.dragging = false;
  });
  canvas.addEventListener('wheel', (event) => {
    event.preventDefault();
    state.cameraZ *= event.deltaY > 0 ? 1.08 : 0.92;
    state.cameraZ = Math.max(1.25, Math.min(3.5, state.cameraZ));
    render();
  }, { passive: false });

  setStageMessage('Loading pair-value field...');
  const canRender = initThree();
  if (!canRender) return;

  fetch(DATA_URL)
    .then((response) => response.json())
    .then((data) => {
      state.data = data;
      setStageMessage('');
      if (state.pendingAssetIndex !== null) {
        setAsset(state.pendingAssetIndex, { silent: true });
        state.pendingAssetIndex = null;
      } else {
        updateLabels();
        updateScene();
      }
      if (window.ResizeObserver && stage) {
        new ResizeObserver(resize).observe(stage);
      } else {
        window.addEventListener('resize', resize);
      }
    })
    .catch(() => {
      if (caption) caption.textContent = 'Interactive teacher data failed to load.';
      setStageMessage('Interactive teacher data failed to load.');
    });
})();
