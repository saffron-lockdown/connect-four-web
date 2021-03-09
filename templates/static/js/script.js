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

  function create_tile(val) {
    const tile = document.createElement("div");
    tile.className = "tile";
    if ([0, 1].includes(val)) {
      const counter = document.createElement("div");
      counter.className = `counter counter-${val}`;
      tile.appendChild(counter);
    }
    return tile;
  }

  function display_board_text(board_text) {
    const board_el = document.getElementById("board");
    board_el.innerHTML = "";
    board_text.forEach((row) => {
      const row_el = document.createElement("div");
      row_el.className = "row";
      row.forEach((col) => {
        const tile = create_tile(col);
        row_el.appendChild(tile);
      });
      board_el.appendChild(row_el);
    });
  }

  function display_winner(winner) {
    if (winner == 0) {
      document.getElementById("msg").innerHTML = "You Win!";
    } else if (winner == 1) {
      document.getElementById("msg").innerHTML = "You Lose!";
    } else {
      document.getElementById("msg").innerHTML = "";
    }
  }

  function make_board_text(board) {
    // Return a text representation of the board

    // extend arrays to full length
    boardE = board.map((col) => {
      while (col.length < 4) {
        col.push(null);
      }
      return col;
    });

    // Transpose the board
    boardT = boardE[0].map((_, colIndex) => board.map((row) => row[colIndex]));

    return boardT.reverse();
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
  $("#restart").click(function () {
    $.post("/postmethod/restart", function (err, req, resp) {
      display_board_text(initial_board());
      display_winner(null);
    });
  });

  function initial_board() {
    const initial_board = [];
    for (let i = 0; i < 4; i += 1) {
      initial_board.push(new Array(4).fill(null));
    }
    return initial_board;
  }

  display_board_text(initial_board());
});
