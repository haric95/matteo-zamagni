"use client";
import { TARGET_CELL_SIZE } from "@/hooks/useScreenDim";
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

const GRID_PIXEL_SIZE = 1.5; // LED pixel size

const PIXEL_TRANSITION_DURATION = 500;

export const PixelGrid: React.FC<PropsWithChildren> = ({ children }) => {
  const { gridDim, grid } = useGlobalContext();
  const dispatch = useGlobalContextDispatch();
  const [cell, setCell] = useState<HTMLDivElement | null>(null);

  const handleThemeTransitionEnd = useCallback(() => {
    if (dispatch) {
      dispatch({
        type: "END_THEME_TRANSITION",
      });
    }
  }, [dispatch]);

  const cellRef = useRef<HTMLDivElement | null>(null);

  const updateCellSize = useCallback(() => {
    if (cellRef.current && dispatch) {
      const rect = cellRef.current.getBoundingClientRect();
      dispatch({
        type: "SET_CELL_SIZE",
        dim: {
          x: rect.width,
          y: rect.height,
        },
      });
    }
  }, [cellRef, dispatch]);

  useEffect(() => {
    // TODO(HC): Find a way to not use timeout here.
    // This is a hack for now, as the loading screen will cover the layout effects of this timeout.
    setTimeout(() => {
      if (cellRef) {
        updateCellSize();
      }
    }, 100);
  }, [cellRef, updateCellSize]);

  useEffect(() => {
    window.addEventListener("resize", updateCellSize);

    return () => {
      window.removeEventListener("resize", updateCellSize);
    };
  }, [updateCellSize]);

  return (
    <main className="fixed relative w-screen h-[calc(100dvh)] flex justify-center items-center">
      <div
        className="absolute w-full h-full bg-background_Light dark:bg-background_Dark transition-[background-color] duration-500"
        onTransitionEnd={handleThemeTransitionEnd}
      />
      <div
        className={`absolute grid pointer-events-none`}
        style={{
          gridTemplateColumns: `repeat(${gridDim?.x}, minmax(0, 1fr))`,
          gridTemplateRows: `repeat(${gridDim?.y}, minmax(0, 1fr))`,
          width: gridDim ? `${gridDim.x * TARGET_CELL_SIZE}px` : "100%",
          height: gridDim ? `${gridDim.y * TARGET_CELL_SIZE}px` : "100%",
        }}
      >
        <div className="flex justify-center items-center" ref={cellRef}>
          <div
            className={`transition-all ${
              grid && grid[0][0]
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
        {grid &&
          grid.map((columns, rowIndex) => {
            return columns.map((pixelIsLit, columnIndex) => {
              if (rowIndex === 0 && columnIndex === 0) {
                return;
              }
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
        className={`absolute grid`}
        style={{
          gridTemplateColumns: `repeat(${gridDim?.x}, minmax(0, 1fr))`,
          gridTemplateRows: `repeat(${gridDim?.y}, minmax(0, 1fr))`,
          width: gridDim ? `${gridDim.x * TARGET_CELL_SIZE}px` : "100%",
          height: gridDim ? `${gridDim.y * TARGET_CELL_SIZE}px` : "100%",
        }}
      >
        {children}
      </div>
    </main>
  );
};
