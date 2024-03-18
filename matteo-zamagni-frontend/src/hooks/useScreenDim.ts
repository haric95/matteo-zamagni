"use client";
import { useGlobalContextDispatch } from "@/state/GlobalStore";
import { useCallback, useEffect } from "react";

export const useScreenDim = () => {
  const dispatch = useGlobalContextDispatch();

  const updateScreenDim = useCallback(() => {
    if (dispatch && window) {
      dispatch({
        type: "SET_DIM",
        dim: { x: window.innerWidth, y: window.innerHeight },
      });
    }
  }, [dispatch]);

  useEffect(() => {
    updateScreenDim();
    window.addEventListener("resize", updateScreenDim);

    return () => window.removeEventListener("resize", updateScreenDim);
  }, [updateScreenDim]);
};
