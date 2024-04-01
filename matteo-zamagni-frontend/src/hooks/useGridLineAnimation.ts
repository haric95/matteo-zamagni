import { clearGrid, lightPixel } from "@/helpers/gridHelpers";
import {
  useGlobalContext,
  useGlobalContextDispatch,
} from "@/state/GlobalStore";
import { Grid, Pos2D } from "@/types/global";
import { useCallback, useEffect, useState } from "react";

// Using an variable outside of the hook so that the ref to the grid is not specific
// to each instance of the hook. This means multiple instances of the hook can mutate
// the grid simultaneously.
let outerGridRef: Grid | null = null;

export const useGridLineAnimation = () => {
  const { grid } = useGlobalContext();
  const dispatch = useGlobalContextDispatch();
  const [currentTimeout, setCurrentTimeout] = useState<NodeJS.Timeout | null>(
    null
  );

  const getUpdatedGrid = useCallback((pos: Pos2D) => {
    if (outerGridRef) {
      return lightPixel(outerGridRef, pos.x, pos.y);
    }
    return null;
  }, []);

  useEffect(() => {
    if (grid) {
      outerGridRef = grid;
    }
  }, [grid]);

  const onAnimationEnd = useCallback(() => {
    // if (gridRef.current) {
    //   const updatedGrid = clearGrid(gridRef.current);
    //   gridRef.current = updatedGrid;
    // }
  }, []);

  const clearLine = useCallback(() => {
    // TODO: Find a way to make this not clear whole grid
    if (outerGridRef && dispatch) {
      const updatedGrid = clearGrid(outerGridRef);
      outerGridRef = updatedGrid;
      dispatch({ type: "CLEAR_GRID" });
      if (currentTimeout) {
        clearTimeout(currentTimeout);
      }
    }
  }, [dispatch, currentTimeout]);

  const onTimeout = useCallback(
    async (points: Pos2D[], duration: number) => {
      if (dispatch) {
        if (points.length === 0) {
          // Cleanup
          onAnimationEnd();
          return true;
        }
        const updatedGrid = getUpdatedGrid(points[0]);
        if (updatedGrid) {
          dispatch({ type: "UPDATE_GRID", grid: updatedGrid });
          outerGridRef = updatedGrid;
          const updatedPointsArray = points.splice(1);
          const timeout = setTimeout(() => {
            onTimeout(updatedPointsArray, duration);
          }, duration);
          setCurrentTimeout(timeout);
        }
      }
    },
    [dispatch, getUpdatedGrid, onAnimationEnd]
  );

  const startAnimationLoop = useCallback(
    async (points: Pos2D[], totalDuration: number) => {
      if (dispatch) {
        const timeoutDuration = totalDuration / points.length;
        const timeout = setTimeout(() => {
          onTimeout(points, timeoutDuration);
        }, timeoutDuration);
        setCurrentTimeout(timeout);
      }
    },
    [dispatch, onTimeout]
  );

  const startAnimation = useCallback(
    async (points: Pos2D[], duration: number = 1000) => {
      startAnimationLoop(points, duration);
    },
    [startAnimationLoop]
  );

  const cancelAnimation = useCallback(() => {
    if (currentTimeout) {
      clearTimeout(currentTimeout);
    }
  }, [currentTimeout]);
  return { startAnimation, cancelAnimation, clearLine };
};
