import { Grid } from "@/types/global";

export const clearGrid = (grid: Grid) => {
  const updated: Grid = grid.map((currentRow, currentRowIndex) => {
    return currentRow.map((currentValue, currentColIndex) => {
      return false;
    });
  });
  return updated;
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
