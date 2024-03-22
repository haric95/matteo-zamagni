"use client";
import { useGlobalContextDispatch } from "@/state/GlobalStore";
import { Dim2D } from "@/types/global";
import { useCallback, useEffect } from "react";

const SCALE_FACTOR = 1; // scales grid
const TARGET_CELL_SIZE = 24; //px will determine how many LEDs will fit

const getGridDim = (screenDim: Dim2D): Dim2D => {
  // Ensure grid dimensions are always even numbers
  const gridWidth =
    Math.floor(screenDim.x / (TARGET_CELL_SIZE * 2)) * 2 * SCALE_FACTOR;
  const gridHeight =
    Math.floor(screenDim.y / (TARGET_CELL_SIZE * 2)) * 2 * SCALE_FACTOR;

  return { x: gridWidth, y: gridHeight };
};

export const useUpdateScreenDim = () => {
  const dispatch = useGlobalContextDispatch();

  const updateDims = useCallback(() => {
    if (dispatch && window) {
      const dim = { x: window.innerWidth, y: window.innerHeight };
      dispatch({
        type: "SET_SCREEN_DIM",
        dim,
      });
      const gridDim = getGridDim(dim);
      dispatch({ type: "SET_GRID_DIM", dim: gridDim });
    }
  }, [dispatch]);

  useEffect(() => {
    updateDims();
    window.addEventListener("resize", updateDims);

    return () => window.removeEventListener("resize", updateDims);
  }, [updateDims]);
};
