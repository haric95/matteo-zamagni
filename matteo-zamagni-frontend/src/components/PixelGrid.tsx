"use client";
import { useGlobalContext } from "@/state/GlobalStore";
import { Dim2D } from "@/types/global";
import React, { PropsWithChildren, useMemo, useState } from "react";

const GRID_PIXEL_SIZE = 1.5; // LED pixel size

enum LEDColors {
  INACTIVE = "#333333",
  ACTIVE = "#ABABAB",
}

export const PixelGrid: React.FC<PropsWithChildren> = ({ children }) => {
  const { screenDim, gridDim } = useGlobalContext();

  const mapArray = useMemo<null[][] | null>(() => {
    return gridDim
      ? (new Array(gridDim.y).fill(new Array(gridDim.x).fill(null)) as null[][])
      : null;
  }, [screenDim]);

  return (
    <main className="fixed relative w-screen h-screen">
      <div
        className={`w-full h-full absolute grid pointer-events-none`}
        style={{
          gridTemplateColumns: `repeat(${gridDim?.x}, minmax(0, 1fr))`,
          gridTemplateRows: `repeat(${gridDim?.y}, minmax(0, 1fr))`,
        }}
      >
        {mapArray &&
          gridDim &&
          mapArray.map((columns, rowIndex) =>
            columns.map((_, columnIndex) => {
              return (
                <div className="flex justify-center items-center">
                  <div
                    style={{
                      width: GRID_PIXEL_SIZE,
                      height: GRID_PIXEL_SIZE,
                      backgroundColor: LEDColors.INACTIVE,
                    }}
                  ></div>
                </div>
              );
            })
          )}
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
