"use client";
import { useGlobalContext } from "@/state/GlobalStore";
import React, { PropsWithChildren } from "react";

const GRID_PIXEL_SIZE = 1.5; // LED pixel size

const PIXEL_TRANSITION_DURATION = 1000;

export const PixelGrid: React.FC<PropsWithChildren> = ({ children }) => {
  const { gridDim, grid } = useGlobalContext();

  return (
    <main className="fixed relative w-screen h-screen">
      <div
        className={`bg-background_Light dark:bg-background_Dark w-full h-full absolute grid pointer-events-none transition-all`}
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
