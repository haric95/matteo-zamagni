import { useCallback } from "react";
import { useGridLineAnimation } from "./useGridLineAnimation";
import { Pos2D } from "@/types/global";

export const useGridRectAnimation = () => {
  const { startAnimation: startAnimation1, clearLine: clearLine1 } =
    useGridLineAnimation();
  const { startAnimation: startAnimation2, clearLine: clearLine2 } =
    useGridLineAnimation();
  const { startAnimation: startAnimation3, clearLine: clearLine3 } =
    useGridLineAnimation();
  const { startAnimation: startAnimation4, clearLine: clearLine4 } =
    useGridLineAnimation();

  const startRectAnimation = useCallback(
    (x: number, y: number, width: number, height: number) => {
      const topPoints: Pos2D[] = new Array(width - 1)
        .fill(null)
        .map((_, index) => {
          return { x: x + index, y: y };
        });
      const rightPoints: Pos2D[] = new Array(height - 1)
        .fill(null)
        .map((_, index) => {
          return { x: x + width, y: y + index };
        });
      const bottomPoints: Pos2D[] = new Array(width - 1)
        .fill(null)
        .map((_, index) => {
          return { x: x + width - index, y: y + height };
        });
      const leftPoints: Pos2D[] = new Array(height - 1)
        .fill(null)
        .map((_, index) => {
          return { x: x, y: y + height - index };
        });

      // startAnimation1(topPoints);
      // startAnimation2(rightPoints);
      // startAnimation3(bottomPoints);
      startAnimation4(leftPoints);
    },
    [startAnimation1, startAnimation2, startAnimation3, startAnimation4]
  );

  const clearRect = useCallback(() => {
    clearLine1();
    clearLine2();
    clearLine3();
    clearLine4();
  }, [clearLine1, clearLine2, clearLine3, clearLine4]);

  return { startRectAnimation, clearRect };
};
