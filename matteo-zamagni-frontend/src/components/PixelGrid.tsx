"use client";
import { useGlobalContext } from "@/state/GlobalStore";
import React, { PropsWithChildren } from "react";

const GRID_PIXEL_SIZE = 1.5; // LED pixel size

enum LEDColors {
  INACTIVE = "#333333",
  ACTIVE = "#ABABAB",
}

export const PixelGrid: React.FC<PropsWithChildren> = ({ children }) => {
  const { gridDim, grid } = useGlobalContext();

  return (
    <main className="fixed relative w-screen h-screen">
      <div
        className={`bg-background_Light dark:bg-background_Dark w-full h-full absolute grid pointer-events-none transition-all duration-1000`}
        style={{
          gridTemplateColumns: `repeat(${gridDim?.x}, minmax(0, 1fr))`,
          gridTemplateRows: `repeat(${gridDim?.y}, minmax(0, 1fr))`,
        }}
      >
        {grid &&
          grid.map((columns, rowIndex) => {
            return columns.map((pixelLit, columnIndex) => {
              return (
                <div className="flex justify-center items-center">
                  <div
                    className="transition-all duration-500"
                    style={{
                      width: GRID_PIXEL_SIZE,
                      height: GRID_PIXEL_SIZE,
                      backgroundColor: pixelLit
                        ? LEDColors.ACTIVE
                        : LEDColors.INACTIVE,
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
