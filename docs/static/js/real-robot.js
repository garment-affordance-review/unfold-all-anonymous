(function () {
  const stage = document.getElementById('realRobotStage');
  const video = document.getElementById('realRobotVideo');
  const overlay = document.getElementById('realRobotPlay');
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
