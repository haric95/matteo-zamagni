import { lightPixel, dimPixels } from "@/helpers/gridHelpers";
import {
  useGlobalContext,
  useGlobalContextDispatch,
} from "@/state/GlobalStore";
import { Pos2D } from "@/types/global";
import { MutableRefObject, useCallback, useEffect, useState } from "react";

type useLEDScrollbarArgs = {
  rowStart: number;
  rowEnd: number;
  element: HTMLDivElement;
};

export const useLEDScrollbar = (
  rowStart: number,
  rowEnd: number,
  column: number,
  element: MutableRefObject<HTMLDivElement | null>
) => {
  const dispatch = useGlobalContextDispatch();
  const { grid } = useGlobalContext();
  const [litPixelRow, setLitPixelRow] = useState<number>(rowEnd - rowStart);
  const [uiLitPixelRow, setUILitPixelRow] = useState<number | null>(null);
  const scrollHandler = useCallback(() => {
    const numLEDs = rowEnd - rowStart;
    if (element.current) {
      const litLED =
        Math.floor(
          (element.current.scrollTop /
            (element.current.scrollHeight - element.current.offsetHeight)) *
            numLEDs
        ) + rowStart;
      setLitPixelRow(litLED);
    }
  }, [element, rowStart, rowEnd]);

  useEffect(() => {
    if (litPixelRow !== uiLitPixelRow) {
      if (dispatch && grid) {
        const pixelsToDim = new Array(rowEnd - rowStart + 1)
          .fill(null)
          .map<Pos2D>((_, index) => ({
            x: column,
            y: rowStart + index,
          }));
        const dimmedGrid = dimPixels(grid, pixelsToDim);
        const updatedGrid = lightPixel(dimmedGrid, column, litPixelRow);
        dispatch({ type: "UPDATE_GRID", grid: updatedGrid });
        setUILitPixelRow(litPixelRow);
      }
    }
  }, [
    dispatch,
    grid,
    column,
    litPixelRow,
    uiLitPixelRow,
    setUILitPixelRow,
    rowEnd,
    rowStart,
  ]);

  useEffect(() => {
    const div = element.current;
    if (div) {
      div.addEventListener("scroll", scrollHandler);
    }

    return () => {
      if (div) {
        div.removeEventListener("scroll", scrollHandler);
      }
    };
  }, [element, scrollHandler]);
};
