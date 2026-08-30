(function () {
  var slide = document.getElementById("slide");
  if (!slide) return;

  function fit() {
    var w = 1920;
    var h = 1080;
    var s = Math.min(window.innerWidth / w, window.innerHeight / h);
    slide.style.transform = "scale(" + s + ")";
  }

  fit();
  window.addEventListener("resize", fit);
})();
