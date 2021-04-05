import {
  Component,
  createElement as h,
  render,
} from "https://unpkg.com/preact@latest?module";

import htm from "https://unpkg.com/htm?module";
import { useState } from "https://unpkg.com/preact@latest/hooks/dist/hooks.module.js?module";

const html = htm.bind(h);
const BOARD_WIDTH = 7;
const BOARD_HEIGHT = 6;

function make_board(board) {
  // extend arrays to full length
  const boardE = board.map((col) => {
    while (col.length < BOARD_HEIGHT) {
      col.push(null);
    }
    return col;
  });

  return boardE;
}

function initial_board() {
  const initial_board = [];
  for (let i = 0; i < BOARD_WIDTH; i += 1) {
    initial_board.push(new Array(BOARD_HEIGHT).fill(null));
  }
  return initial_board;
}

function App(props) {
  const [board, setBoard] = useState(initial_board());
  const [winner, setWinner] = useState();
  let winnerText;
  if (winner === 0) {
    winnerText = "You Win!";
  } else if (winner === 1) {
    winnerText = "You Lose!";
  } else {
    winnerText = "";
  }
  return html`
    <div class="row board">
      ${[
        board.map((col, colIndex) => {
          const ghostRowIndex = col.findIndex((x) => x === null);
          return html`
            <div class="col">
              ${col.map(
                (row, rowIndex) => html`
                  <div
                    class="tile"
                    onClick=${() => {
                      $.post(
                        "/postmethod",
                        { move: colIndex },
                        function (err, req, resp) {
                          var board = make_board(resp["responseJSON"]["board"]);
                          var new_board = make_board(
                            resp["responseJSON"]["new_board"]
                          );
                          var winner = resp["responseJSON"]["winner"];

                          setBoard(board);
                          setTimeout(() => {
                            setBoard(new_board);
                            setWinner(winner);
                          }, 200);
                        }
                      );
                    }}
                  >
                    ${typeof row === "number" &&
                    html`<div class="counter counter-${row}"></div>`}
                    ${rowIndex === ghostRowIndex &&
                    html`<div class="counter ghost"></div>`}
                  </div>
                `
              )}
            </div>
          `;
        }),
      ]}
    </div>
    <div class="row">
      <button
        class="btn btn-primary btn-block"
        onClick=${() => {
          $.post("/postmethod/restart", function (err, req, resp) {
            setBoard(initial_board());
            setWinner(null);
          });
        }}
      >
        Restart Game
      </button>
    </div>
    <div class="row"><p>${winnerText}</p></div>
  `;
}

render(html`<${App} name="World" />`, document.querySelector("#app"));
