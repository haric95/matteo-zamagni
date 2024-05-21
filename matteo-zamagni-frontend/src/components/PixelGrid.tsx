"use client";
import { useSetDarkThemeClass } from "@/hooks/useSetDarkThemeClass";
import {
  useGlobalContext,
  useGlobalContextDispatch,
} from "@/state/GlobalStore";
import React, {
  PropsWithChildren,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

const GRID_PIXEL_SIZE = 2; // LED pixel size

const PIXEL_TRANSITION_DURATION = 500;

export const PixelGrid: React.FC<PropsWithChildren> = ({ children }) => {
  const { gridDim, grid } = useGlobalContext();
  const dispatch = useGlobalContextDispatch();

  const handleThemeTransitionEnd = useCallback(() => {
    if (dispatch) {
      console.log("tranitionend");
      dispatch({
        type: "END_THEME_TRANSITION",
      });
    }
  }, [dispatch]);

  return (
    <main className="fixed relative w-screen h-screen">
      <div
        className="absolute w-full h-full bg-background_Light dark:bg-background_Dark transition-[background-color] duration-500"
        onTransitionEnd={handleThemeTransitionEnd}
      />
      <div
        className={`w-full h-full absolute grid pointer-events-none`}
        style={{
          gridTemplateColumns: `repeat(${gridDim?.x}, minmax(0, 1fr))`,
          gridTemplateRows: `repeat(${gridDim?.y}, minmax(0, 1fr))`,
        }}
      >
        {grid &&
          grid.map((columns, rowIndex) => {
            return columns.map((pixelIsLit, columnIndex) => {
              return (
                <div
                  className="flex justify-center items-center"
                  key={`${rowIndex}-${columnIndex}`}
                >
                  <div
                    className={`transition-all ${
                      pixelIsLit
                        ? "bg-ledActive_Light dark:bg-ledActive_Dark"
                        : "bg-ledInactive_Light dark:bg-ledInactive_Dark"
                    }`}
                    style={{
                      width: GRID_PIXEL_SIZE,
                      height: GRID_PIXEL_SIZE,
                      transitionDuration: `${PIXEL_TRANSITION_DURATION}ms`,
                    }}
                  ></div>
                </div>
              );
            });
          })}
      </div>
      <div
        className={`w-full h-full absolute grid`}
        style={{
          gridTemplateColumns: `repeat(${gridDim?.x}, minmax(0, 1fr))`,
          gridTemplateRows: `repeat(${gridDim?.y}, minmax(0, 1fr))`,
        }}
      >
        {children}
      </div>
    </main>
  );
};
