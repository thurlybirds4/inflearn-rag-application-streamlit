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

  var slide = document.getElementById("slide");
  if (!slide) return;

  var moving = false;

  function go(href) {
    if (!href || moving) return;
    if ((file === "06.html" && href === "07.html") ||
        (file === "11.html" && href === "12.html") ||
        (file === "17.html" && href === "18.html")) {
      if (!slide.classList.contains("is-leaving")) {
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

  function makeBtn(cls, label, svgPath) {
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "nav-btn " + cls;
    btn.setAttribute("aria-label", label);
    btn.innerHTML =
      '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true">' +
        '<path d="' + svgPath + '" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>' +
      "</svg>";
    return btn;
  }

  if (i > 0) {
    var prev = makeBtn(
      "nav-prev",
      "이전 슬라이드",
      "M14.5 6.5L9 12l5.5 5.5"
    );
    prev.addEventListener("click", function (e) {
      e.stopPropagation();
      go(slides[i - 1]);
    });
    slide.appendChild(prev);
  }

  if (i < slides.length - 1) {
    var next = makeBtn(
      "nav-next",
      "다음 슬라이드",
      "M9.5 6.5L15 12l-5.5 5.5"
    );
    if (i === 0) next.classList.add("is-cover");
    next.addEventListener("click", function (e) {
      e.stopPropagation();
      go(slides[i + 1]);
    });
    slide.appendChild(next);
  }
})();
