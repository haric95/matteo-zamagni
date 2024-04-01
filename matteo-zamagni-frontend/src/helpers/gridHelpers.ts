import { Grid, Pos2D } from "@/types/global";

export const clearGrid = (grid: Grid) => {
  const updated: Grid = grid.map((currentRow, currentRowIndex) => {
    return currentRow.map((currentValue, currentColIndex) => {
      return false;
    });
  });
  return updated;
};

export const lightPixel = (grid: Grid, x: number, y: number) => {
  const clonedGrid = [...grid];
  const clonedRow = [...clonedGrid[y]];
  clonedRow[x] = true;
  clonedGrid[y] = clonedRow;
  return clonedGrid;
};

export const drawVerticalLine = (
  grid: Grid,
  col: number,
  rowStart: number,
  rowEnd: number
) => {
  const updated: Grid = grid.map((currentRow, currentRowIndex) => {
    return currentRow.map((currentValue, currentColIndex) => {
      if (
        currentColIndex === col &&
        currentRowIndex >= rowStart &&
        currentRowIndex < rowEnd
      ) {
        return true;
      } else {
        return currentValue;
      }
    });
  });
  return updated;
};

// From https://stackoverflow.com/a/38796691
export const tronPath = (a: Pos2D, b: Pos2D) => {
  var path = [a];
  var x = a.x,
    y = a.y; // starting cell
  var dx = a.x == b.x ? 0 : b.x > a.x ? 1 : -1; // right or left
  var dy = a.y == b.y ? 0 : b.y > a.y ? 1 : -1; // up or down
  if (dx == 0 || dy == 0) {
    // STRAIGHT LINE ...
  } else if (Math.abs(b.x - a.x) > Math.abs(b.y - a.y)) {
    // MAINLY HORIZONTAL
    var tan = (b.y - a.y) / (b.x - a.x); // tangent
    var max = (1 - Math.abs(tan)) / 2; // distance threshold
    while (x != b.x || y != b.y) {
      // while target not reached
      var ideal = a.y + (x - a.x) * tan; // y of ideal line at x
      if ((ideal - y) * dy >= max) y += dy; // move vertically
      else x += dx; // move horizontally
      path.push({ x: x, y: y }); // add cell to path
    }
  } else {
    // MAINLY VERTICAL
    var cotan = (b.x - a.x) / (b.y - a.y); // cotangent
    var max = (1 - Math.abs(cotan)) / 2; // distance threshold
    while (x != b.x || y != b.y) {
      // while target not reached
      var ideal = a.x + (y - a.y) * cotan; // x of ideal line at y
      if ((ideal - x) * dx >= max) x += dx; // move horizontally
      else y += dy; // move vertically
      path.push({ x: x, y: y }); // add cell to path
    }
  }
  return path;
};

export const findNearestCornerOfRect = (
  point: Pos2D,
  rect: Pos2D & { width: number; height: number }
) => {
  const nearestCornerX =
    Math.abs(point.x - rect.x) < Math.abs(point.x - (rect.x + rect.width))
      ? rect.x
      : rect.x + rect.width;
  const nearestCornerY =
    Math.abs(point.y - rect.y) < Math.abs(point.y - (rect.y + rect.height))
      ? rect.y
      : rect.y + rect.height;

  return { x: nearestCornerX, y: nearestCornerY };
};
