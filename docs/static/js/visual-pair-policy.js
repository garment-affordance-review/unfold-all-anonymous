(function () {
  const video = document.getElementById('visualPolicyVideo');
  const prev = document.getElementById('visualPolicyPrev');
  const next = document.getElementById('visualPolicyNext');
  const caption = document.getElementById('visualPolicyCaption');
  if (!video || !prev || !next || !caption) return;

  const assets = [
    {
      label: 'Asset 1 / 8',
      src: 'static/videos/visual_pair_policy/a1_a2_asset_1.mp4?v=method-assets-8-v1'
    },
    {
      label: 'Asset 2 / 8',
      src: 'static/videos/visual_pair_policy/a1_a2_asset_2.mp4?v=method-assets-8-v1'
    },
    {
      label: 'Asset 3 / 8',
      src: 'static/videos/visual_pair_policy/a1_a2_asset_3.mp4?v=method-assets-8-v1'
    },
    {
      label: 'Asset 4 / 8',
      src: 'static/videos/visual_pair_policy/a1_a2_asset_4.mp4?v=method-assets-8-v1'
    },
    {
      label: 'Asset 5 / 8',
      src: 'static/videos/visual_pair_policy/a1_a2_asset_5.mp4?v=method-assets-8-v1'
    },
    {
      label: 'Asset 6 / 8',
      src: 'static/videos/visual_pair_policy/a1_a2_asset_6.mp4?v=method-assets-8-v1'
    },
    {
      label: 'Asset 7 / 8',
      src: 'static/videos/visual_pair_policy/a1_a2_asset_7.mp4?v=method-assets-8-v1'
    },
    {
      label: 'Asset 8 / 8',
      src: 'static/videos/visual_pair_policy/a1_a2_asset_8.mp4?v=method-assets-8-v1'
    }
  ];
  let index = 0;

  function setAsset(nextIndex, options) {
    index = (nextIndex + assets.length) % assets.length;
    video.pause();
    video.src = assets[index].src;
    video.load();
    video.play().catch(function () {});
    caption.textContent = `${assets[index].label}. Top: A1 overlay on a synthetic rendered garment. Bottom: A2 overlay conditioned on the current x1.`;
    if (!options || !options.silent) {
      window.dispatchEvent(new CustomEvent('methodAssetChange', {
        detail: { index, source: 'visual-policy' }
      }));
    }
  }

  prev.addEventListener('click', function () {
    setAsset(index - 1);
  });
  next.addEventListener('click', function () {
    setAsset(index + 1);
  });
  window.addEventListener('methodAssetChange', function (event) {
    if (!event.detail || event.detail.source === 'visual-policy') return;
    setAsset(Number(event.detail.index) || 0, { silent: true });
  });
})();
