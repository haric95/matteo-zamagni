"use client";
import { useGlobalContext } from "@/state/GlobalStore";
import React, { PropsWithChildren } from "react";

const GRID_PIXEL_SIZE = 2; // LED pixel size

const PIXEL_TRANSITION_DURATION = 500;

export const PixelGrid: React.FC<PropsWithChildren> = ({ children }) => {
  const { gridDim, grid } = useGlobalContext();

  return (
    <main className="fixed relative w-screen h-screen">
      <div className="absolute h-full w-full">
        <img className="h-full w-full" src="/othergif.gif" />
      </div>
      <div
        className={`w-full h-full absolute grid pointer-events-none transition-all duration-500`}
        style={{
          gridTemplateColumns: `repeat(${gridDim?.x}, minmax(0, 1fr))`,
          gridTemplateRows: `repeat(${gridDim?.y}, minmax(0, 1fr))`,
        }}
      >
        {grid &&
          grid.map((columns, rowIndex) => {
            return columns.map((pixelIsLit, columnIndex) => {
              return (
                <div className="relative" key={`${rowIndex}-${columnIndex}`}>
                  <div className="absolute h-full w-full bg-white opacity-25" />
                  <div
                    className="absolute h-full w-full"
                    style={{ backgroundImage: 'url("/pixel.png")' }}
                  />
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
