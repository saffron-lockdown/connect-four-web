$(document).ready(function () {
  function postMove(move) {
    $.post("/postmethod", { move }, function (err, req, resp) {
      var board = resp["responseJSON"]["board"];
      // extend the board
      var boardE = board.map((col) => {
        while (col.length < 4) {
          col.push("_");
        }
        return col;
      });

      // Transpose the board
      boardT = boardE[0].map((_, colIndex) =>
        board.map((row) => row[colIndex])
      );

      boardText = boardT.reverse().map(function (x) {
        return x.join("");
      });

      document.getElementById("board").innerHTML = boardText.join("<br>");
    });
  }

  $("#clearButton").click(function () {
    clearCanvas();
  });

  $("#move1").click(function () {
    postMove(0);
  });
  $("#move2").click(function () {
    postMove(1);
  });
});
