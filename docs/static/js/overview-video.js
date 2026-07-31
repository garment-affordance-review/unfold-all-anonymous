(function () {
  const stage = document.getElementById('overviewVideoStage');
  const video = document.getElementById('overviewVideo');
  const overlay = document.getElementById('overviewVideoPlay');
  if (!stage || !video || !overlay) return;

  overlay.addEventListener('click', function () {
    video.play().catch(function () {});
  });
  video.addEventListener('play', function () {
    stage.classList.add('is-playing');
  });
  video.addEventListener('pause', function () {
    stage.classList.remove('is-playing');
  });
})();
