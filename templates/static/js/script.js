$(document).ready(function () {
  function postMove(move) {
    $.post("/postmethod", { move }, function (err, req, resp) {
      var board_text = make_board_text(resp["responseJSON"]["board"]);
      var new_board_text = make_board_text(resp["responseJSON"]["new_board"]);
      var winner = resp["responseJSON"]["winner"];

      display_board_text(board_text);
      setTimeout(() => {
        display_board_text(new_board_text);
        display_winner(winner);
      }, 200);
    });
  }

  function display_board_text(board_text) {
    document.getElementById("board").innerHTML = board_text;
  }

  function display_winner(winner) {
    if (winner == 0) {
      document.getElementById("msg").innerHTML = "You Win!";
    } else if (winner == 1) {
      document.getElementById("msg").innerHTML = "You Lose!";
    }
  }

  function make_board_text(board) {
    // Return a text representation of the board

    // extend arrays to full length
    boardE = board.map((col) => {
      while (col.length < 4) {
        col.push("_");
      }
      return col;
    });

    // Transpose the board
    boardT = boardE[0].map((_, colIndex) => board.map((row) => row[colIndex]));

    return boardT
      .reverse()
      .map(function (x) {
        return x.join("");
      })
      .join("<br>");
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
  $("#move3").click(function () {
    postMove(2);
  });
  $("#move4").click(function () {
    postMove(3);
  });
});
