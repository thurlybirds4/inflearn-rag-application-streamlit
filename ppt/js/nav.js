(function () {
  var slides = [
    "01.html", "02.html", "03.html", "04.html",
    "06.html", "07.html", "08.html", "09.html", "10.html",
    "11.html", "12.html", "13.html", "14.html", "15.html",
    "16.html", "17.html", "18.html", "19.html", "20.html",
    "21.html", "22.html", "23.html", "24.html", "25.html"
  ];

  var file = decodeURIComponent(location.pathname.split("/").pop() || "");
  var i = slides.indexOf(file);
  if (i < 0) return;

  var hint = document.createElement("div");
  hint.id = "nav-hint";
  hint.innerHTML =
    '<svg class="nav-left" viewBox="0 0 24 24" fill="none" aria-hidden="true">' +
      '<path d="M14.5 6.5L9 12l5.5 5.5" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>' +
    "</svg>" +
    '<svg class="nav-right" viewBox="0 0 24 24" fill="none" aria-hidden="true">' +
      '<path d="M9.5 6.5L15 12l-5.5 5.5" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>' +
    "</svg>";
  document.body.appendChild(hint);

  var side = null;

  function update(e) {
    var x = e.clientX / window.innerWidth;
    var next = x <= 0.3 ? "left" : x >= 0.7 ? "right" : null;
    if (next === "left" && i === 0) next = null;
    if (next === "right" && i === slides.length - 1) next = null;
    side = next;
    hint.className = side ? "is-" + side : "is-off";
    hint.style.transform =
      "translate(" + (e.clientX + (side === "left" ? -52 : 20)) + "px," +
      (e.clientY - 22) + "px)";
    document.body.style.cursor = side ? "pointer" : "default";
  }

  document.addEventListener("mousemove", update);
  var moving = false;

  function go(href) {
    if (moving) return;
    if ((file === "06.html" && href === "07.html") ||
        (file === "11.html" && href === "12.html") ||
        (file === "17.html" && href === "18.html")) {
      var slide = document.getElementById("slide");
      if (slide && !slide.classList.contains("is-leaving")) {
        moving = true;
        if (file === "11.html") {
          var board = slide.querySelector(".board");
          var lastTitle = board && board.querySelector(".col:last-child .h");
          if (lastTitle) lastTitle.textContent = "Refactoring";
          if (board && board.children.length === 4) {
            var fifth = document.createElement("div");
            fifth.className = "card col";
            fifth.innerHTML = '<div class="head"><div class="step">05</div><div class="h">Cloud Deploy</div></div>';
            board.appendChild(fifth);
          }
        }
        slide.classList.add("is-leaving");
        setTimeout(function () { location.href = href; }, 460);
        return;
      }
    }
    location.href = href;
  }

  document.addEventListener("click", function (e) {
    if (e.target.closest("a")) return;
    if (side === "right") go(slides[i + 1]);
    if (side === "left") go(slides[i - 1]);
  });
})();
